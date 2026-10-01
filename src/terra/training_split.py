"""Deterministic, identity-disjoint splits for mixed policy-training cohorts."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

DEFAULT_TEST_FRACTION = 0.15
DEFAULT_SPLIT_SEED = "terra-universal-tracking-split-v1"

_GAIT120_MOTION_TYPES = {
    "LevelWalking": "flat_locomotion",
    "SlopeAscent": "ramp_up",
    "SlopeDescent": "ramp_down",
    "StairAscent": "stairs_up",
    "StairDescent": "stairs_down",
    "SitToStand": "chair_sit_to_stand",
    "StandToSit": "chair_stand_to_sit",
}


def _stable_key(seed: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def canonical_motion_type(row: Mapping[str, str]) -> str:
    """Return a manuscript-readable motion type from catalog metadata or identity."""
    motion = row.get("motion", "").strip()
    parts = motion.split("/")
    if len(parts) >= 3 and parts[0] == "Gait120":
        movement_type = _GAIT120_MOTION_TYPES.get(parts[2])
        if movement_type is not None:
            return movement_type

    explicit = (row.get("motion_type") or row.get("terrain_class") or "").strip().casefold()
    if explicit == "flat":
        return "flat_locomotion"
    if explicit:
        return explicit
    if row.get("dataset", "").strip().casefold() == "amass-locomotion":
        return "flat_locomotion"
    return "unspecified"


def split_identity(motion: str) -> tuple[str, str]:
    """Return the split domain and recorded person/session identity.

    All Gait120 motion families share a subject identity even when their artifacts
    came from separate flat, chair, and non-flat runs. Most AMASS collections put
    the actor in the second path component (for example ``KIT/513``), while some
    repeat the collection name before it (for example ``CMU/CMU/106``). Account
    for both layouts so clips from one recorded person always remain together.
    """
    parts = motion.strip().split("/")
    if len(parts) < 2 or not all(parts[:2]):
        raise ValueError(f"motion has no dataset/person identity: {motion!r}")
    source = parts[0]
    domain = source if source in {"Gait120", "PRISM", "Vielemeyer", "Darmstadt"} else "AMASS"
    normalized_source = "".join(character for character in source.casefold() if character.isalnum())
    normalized_parent = "".join(character for character in parts[1].casefold() if character.isalnum())
    identity_depth = 3 if len(parts) >= 4 and normalized_source == normalized_parent else 2
    return domain, "/".join(parts[:identity_depth])


def _stratum(row: Mapping[str, str]) -> tuple[str, str]:
    dataset = row.get("dataset", "").strip().casefold()
    if not dataset:
        raise ValueError(f"dataset is unavailable for {row.get('motion', '')!r}")
    return dataset, canonical_motion_type(row)


def _score(
    counts: Counter[tuple[str, str]],
    totals: Counter[tuple[str, str]],
    stratum_group_counts: Counter[tuple[str, str]],
    holdout_fraction: float,
) -> float:
    score = 0.0
    for stratum, total in totals.items():
        target = holdout_fraction * total
        observed = counts[stratum]
        score += ((observed - target) / max(target, 1.0)) ** 2
        # Every type must remain represented in training. When two or more
        # identities exist, require representation in the test set as well.
        if observed == total or (stratum_group_counts[stratum] >= 2 and observed == 0):
            score += 1_000_000.0
    target_total = holdout_fraction * sum(totals.values())
    score += ((sum(counts.values()) - target_total) / max(target_total, 1.0)) ** 2
    return score


def _domain_holdout_groups(
    rows: Sequence[Mapping[str, str]],
    group_rows: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    holdout_fraction: float,
    seed: str,
) -> set[str]:
    group_counts = {group: Counter(_stratum(row) for row in members) for group, members in group_rows.items()}
    totals = Counter(_stratum(row) for row in rows)
    stratum_group_counts = Counter()
    for counts in group_counts.values():
        stratum_group_counts.update(counts.keys())

    group_target = min(
        len(group_rows) - 1,
        max(1, round(holdout_fraction * len(group_rows))),
    )
    selected: set[str] = set()
    selected_counts: Counter[tuple[str, str]] = Counter()
    ordered = sorted(group_rows, key=lambda group: _stable_key(seed, group))
    while len(selected) < group_target:
        chosen = min(
            (group for group in ordered if group not in selected),
            key=lambda group: (
                _score(
                    selected_counts + group_counts[group],
                    totals,
                    stratum_group_counts,
                    holdout_fraction,
                ),
                _stable_key(seed, group),
            ),
        )
        selected.add(chosen)
        selected_counts.update(group_counts[chosen])

    # A deterministic one-for-one local search removes most residual imbalance
    # without changing the held-out identity count.
    current_score = _score(selected_counts, totals, stratum_group_counts, holdout_fraction)
    for _ in range(100):
        best: tuple[float, bytes, str, str, Counter[tuple[str, str]]] | None = None
        for removed in selected:
            without = selected_counts - group_counts[removed]
            for added in ordered:
                if added in selected:
                    continue
                candidate_counts = without + group_counts[added]
                candidate_score = _score(
                    candidate_counts,
                    totals,
                    stratum_group_counts,
                    holdout_fraction,
                )
                tie = _stable_key(seed, f"{removed}->{added}")
                candidate = (candidate_score, tie, removed, added, candidate_counts)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
        if best is None or best[0] >= current_score - 1e-12:
            break
        current_score, _tie, removed, added, selected_counts = best
        selected.remove(removed)
        selected.add(added)
    return selected


def assign_stratified_splits(
    rows: Sequence[Mapping[str, str]],
    *,
    holdout_fraction: float = DEFAULT_TEST_FRACTION,
    seed: str = DEFAULT_SPLIT_SEED,
) -> list[dict[str, str]]:
    """Assign identity-disjoint train/test labels balanced by dataset and type."""
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be greater than 0 and less than 0.5")
    if not seed:
        raise ValueError("split seed must be non-empty")
    if not rows:
        raise ValueError("cannot split an empty motion cohort")

    domains: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    groups_by_domain: dict[str, dict[str, list[Mapping[str, str]]]] = defaultdict(lambda: defaultdict(list))
    seen: set[str] = set()
    for row in rows:
        motion = row.get("motion", "").strip()
        if not motion or motion in seen:
            raise ValueError(f"cohort contains an empty or duplicate motion: {motion!r}")
        seen.add(motion)
        if canonical_motion_type(row) == "unspecified":
            raise ValueError(f"motion type is unavailable for {motion!r}")
        domain, identity = split_identity(motion)
        domains[domain].append(row)
        groups_by_domain[domain][identity].append(row)

    test_groups: set[tuple[str, str]] = set()
    for domain, domain_rows in domains.items():
        domain_groups = groups_by_domain[domain]
        if len(domain_groups) < 2:
            raise ValueError(f"split domain {domain!r} has fewer than two identities")
        selected = _domain_holdout_groups(
            domain_rows,
            domain_groups,
            holdout_fraction=holdout_fraction,
            seed=f"{seed}:{domain}",
        )
        test_groups.update((domain, identity) for identity in selected)

    output = []
    for row in rows:
        item = dict(row)
        item["motion_type"] = canonical_motion_type(row)
        item["split"] = "test" if split_identity(item["motion"]) in test_groups else "train"
        output.append(item)
    return output


def split_audit(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    """Summarize a published split and fail on identity leakage."""
    identities: dict[tuple[str, str], set[str]] = defaultdict(set)
    strata: Counter[tuple[str, str, str]] = Counter()
    split_counts: Counter[str] = Counter()
    for row in rows:
        split = row.get("split", "train").strip() or "train"
        if split not in {"train", "evaluation", "test"}:
            raise ValueError(f"invalid split {split!r} for {row.get('motion', '')!r}")
        identities[split_identity(row["motion"])].add(split)
        dataset, motion_type = _stratum(row)
        strata[(dataset, motion_type, split)] += 1
        split_counts[split] += 1
    leaked = [identity for identity, splits in identities.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"train/evaluation identity leakage: {leaked[0][1]!r}")
    return {
        "motions": len(rows),
        "identities": len(identities),
        "splits": dict(sorted(split_counts.items())),
        "strata": [
            {
                "dataset": dataset,
                "motion_type": motion_type,
                "split": split,
                "motions": count,
            }
            for (dataset, motion_type, split), count in sorted(strata.items())
        ],
    }


__all__ = [
    "DEFAULT_SPLIT_SEED",
    "DEFAULT_TEST_FRACTION",
    "assign_stratified_splits",
    "canonical_motion_type",
    "split_audit",
    "split_identity",
]
