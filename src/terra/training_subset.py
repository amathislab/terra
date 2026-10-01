"""Deterministic, stratified subsets of versioned training selections."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from terra._files import atomic_write
from terra.training_segments import selection_segment
from terra.training_split import canonical_motion_type, split_identity

DEFAULT_SUBSET_SEED = "terra-universal-medium-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(seed: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def _stratum(row: Mapping[str, str]) -> tuple[str, str]:
    dataset = row.get("dataset", "").strip().casefold()
    motion_type = canonical_motion_type(row)
    if not dataset or motion_type == "unspecified":
        raise ValueError(f"dataset or motion type is unavailable for {row.get('motion', '')!r}")
    return dataset, motion_type


def _source_motion(row: Mapping[str, str]) -> str:
    segment = selection_segment(row)
    return row.get("motion", "").strip() if segment is None else segment.source_motion


def _largest_remainder_quotas(
    counts: Counter[tuple[str, str]],
    target_entries: int,
) -> dict[tuple[str, str], int]:
    total = sum(counts.values())
    if target_entries < len(counts):
        raise ValueError(
            f"target_entries ({target_entries}) cannot represent all {len(counts)} dataset/motion-type strata"
        )
    quotas = {stratum: target_entries * count // total for stratum, count in counts.items()}
    for stratum in counts:
        quotas[stratum] = max(1, quotas[stratum])
    remainder = target_entries - sum(quotas.values())
    if remainder < 0:
        raise ValueError("target_entries is too small after preserving every stratum")
    ordered = sorted(
        counts,
        key=lambda stratum: (
            -(target_entries * counts[stratum] / total - quotas[stratum]),
            stratum,
        ),
    )
    for stratum in ordered[:remainder]:
        quotas[stratum] += 1
    return quotas


def select_stratified_training_subset(
    rows: Sequence[Mapping[str, str]],
    *,
    target_entries: int,
    seed: str = DEFAULT_SUBSET_SEED,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Select exactly ``target_entries`` train rows without splitting source clips.

    Existing ``test`` and ``evaluation`` rows are retained verbatim. Training
    quotas use largest-remainder proportional allocation over
    ``(dataset, motion_type)`` strata. A derived temporal segment is atomic with
    every sibling segment from the same source clip.
    """

    if isinstance(target_entries, bool) or not isinstance(target_entries, int) or target_entries < 1:
        raise ValueError("target_entries must be a positive integer")
    if not seed:
        raise ValueError("seed must be non-empty")
    if not rows:
        raise ValueError("cannot subset an empty selection")

    copied = [{str(key): str(value) for key, value in row.items()} for row in rows]
    seen_motions: set[str] = set()
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    group_splits: dict[str, str] = {}
    group_strata: dict[str, tuple[str, str]] = {}
    segment_indices: dict[str, set[int]] = defaultdict(set)
    segment_counts: dict[str, int] = {}

    for row in copied:
        motion = row.get("motion", "").strip()
        split = row.get("split", "train").strip() or "train"
        if not motion or motion in seen_motions:
            raise ValueError(f"selection contains an empty or duplicate motion: {motion!r}")
        if split not in {"train", "test", "evaluation"}:
            raise ValueError(f"invalid split {split!r} for {motion!r}")
        seen_motions.add(motion)
        source = _source_motion(row)
        groups[source].append(row)
        if group_splits.setdefault(source, split) != split:
            raise ValueError(f"source motion {source!r} crosses split boundaries")
        stratum = _stratum(row)
        if group_strata.setdefault(source, stratum) != stratum:
            raise ValueError(f"source motion {source!r} crosses dataset/motion-type strata")
        segment = selection_segment(row)
        if segment is not None:
            segment_indices[source].add(segment.index)
            segment_counts[source] = segment.count

    for source, indices in segment_indices.items():
        expected = set(range(1, segment_counts[source] + 1))
        if indices != expected or len(groups[source]) != segment_counts[source]:
            raise ValueError(f"segmented source motion {source!r} is missing sibling intervals")

    training_groups = {source: members for source, members in groups.items() if group_splits[source] == "train"}
    full_train_entries = sum(len(members) for members in training_groups.values())
    if target_entries > full_train_entries:
        raise ValueError(f"target_entries ({target_entries}) exceeds available train entries ({full_train_entries})")

    full_counts: Counter[tuple[str, str]] = Counter()
    for source, members in training_groups.items():
        stratum = group_strata[source]
        full_counts[stratum] += len(members)
    quotas = _largest_remainder_quotas(full_counts, target_entries)

    stable_keys = {source: _stable_key(seed, source) for source in training_groups}
    # Seed every stratum with its smallest atomic group. This guarantees rare
    # types remain represented even when one long source clip has more derived
    # entries than its proportional quota.
    selected_sources = {
        min(
            (source for source in training_groups if group_strata[source] == stratum),
            key=lambda source: (len(training_groups[source]), stable_keys[source]),
        )
        for stratum in full_counts
    }
    selected_counts: Counter[tuple[str, str]] = Counter()
    for source in selected_sources:
        selected_counts[group_strata[source]] += len(training_groups[source])

    remaining = target_entries - sum(selected_counts.values())
    if remaining < 0:
        raise ValueError(
            f"target_entries ({target_entries}) is smaller than the {sum(selected_counts.values())} entries "
            "required to preserve every dataset/motion-type stratum and complete segment group"
        )
    while remaining:
        candidates = [
            source
            for source, members in training_groups.items()
            if source not in selected_sources and len(members) <= remaining
        ]
        if not candidates:
            raise ValueError(
                f"cannot reach exactly {target_entries} entries while retaining complete segment groups; "
                f"{remaining} entries remain"
            )

        def candidate_key(source: str) -> tuple[float, bytes]:
            stratum = group_strata[source]
            weight = len(training_groups[source])
            quota = quotas[stratum]
            before = (selected_counts[stratum] - quota) / max(quota, 1)
            after = (selected_counts[stratum] + weight - quota) / max(quota, 1)
            return after * after - before * before, stable_keys[source]

        chosen = min(candidates, key=candidate_key)
        selected_sources.add(chosen)
        selected_counts[group_strata[chosen]] += len(training_groups[chosen])
        remaining -= len(training_groups[chosen])

    selected_rows = [
        row
        for row in copied
        if (row.get("split", "train").strip() or "train") != "train" or _source_motion(row) in selected_sources
    ]
    retained_counts = Counter((row.get("split", "train").strip() or "train") for row in selected_rows)
    selected_identity_count = len({split_identity(source) for source in selected_sources})
    stratum_records = [
        {
            "dataset": dataset,
            "motion_type": motion_type,
            "full_train_entries": full_counts[(dataset, motion_type)],
            "target_entries": quotas[(dataset, motion_type)],
            "selected_train_entries": selected_counts[(dataset, motion_type)],
            "selected_source_motions": sum(
                source in selected_sources
                for source in training_groups
                if group_strata[source] == (dataset, motion_type)
            ),
        }
        for dataset, motion_type in sorted(full_counts)
    ]
    audit: dict[str, object] = {
        "schema_version": 1,
        "seed": seed,
        "target_train_entries": target_entries,
        "full_train_entries": full_train_entries,
        "selected_train_entries": retained_counts["train"],
        "full_train_source_motions": len(training_groups),
        "selected_train_source_motions": len(selected_sources),
        "selected_train_identities": selected_identity_count,
        "retained_test_entries": retained_counts["test"],
        "retained_evaluation_entries": retained_counts["evaluation"],
        "complete_segment_groups": True,
        "strata": stratum_records,
    }
    return selected_rows, audit


def publish_training_subset(
    source_manifest: Path,
    destination_manifest: Path,
    audit_path: Path,
    rows: Sequence[Mapping[str, str]],
    audit: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Atomically publish a training subset and its provenance audit."""

    source_reference = source_manifest.as_posix()
    destination_reference = destination_manifest.as_posix()
    source = source_manifest.expanduser().resolve()
    destination = destination_manifest.expanduser().resolve()
    resolved_audit = audit_path.expanduser().resolve()
    if destination.suffix.casefold() != ".csv":
        raise ValueError("training subset must use a .csv extension")
    if resolved_audit.suffix.casefold() != ".json":
        raise ValueError("training subset audit must use a .json extension")
    if len({source, destination, resolved_audit}) != 3:
        raise ValueError("source selection, subset selection, and audit paths must differ")
    collisions = [path for path in (destination, resolved_audit) if path.exists()]
    if collisions and not overwrite:
        raise FileExistsError(f"refusing to replace existing subset output: {collisions[0]}")
    if not rows:
        raise ValueError("training subset must be non-empty")

    fieldnames = list(rows[0])

    def write_csv(path: Path) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    atomic_write(destination, write_csv)
    payload = dict(audit) | {
        "source_selection": source_reference,
        "source_selection_sha256": _sha256(source),
        "selection": destination_reference,
        "selection_sha256": _sha256(destination),
    }
    atomic_write(
        resolved_audit,
        lambda temporary: temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"),
    )
    return payload


__all__ = [
    "DEFAULT_SUBSET_SEED",
    "publish_training_subset",
    "select_stratified_training_subset",
]
