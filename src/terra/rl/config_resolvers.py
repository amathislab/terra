"""OmegaConf resolvers used by TERRA reinforcement-learning configs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from omegaconf import OmegaConf

MOTION_SELECTION_RECORD_ENV = "TERRA_MOTION_SELECTION_RECORD"
_LEGACY_MOTION_ENV = {
    "train": "TERRA_MOTIONS",
    "validation": "TERRA_VALIDATION_MOTIONS",
}


def _validated_motion_names(value: object, *, source: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source} must contain a non-empty JSON list of motion names")
    if any(not isinstance(name, str) or not name for name in value):
        raise ValueError(f"{source} motion names must be non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{source} contains duplicate motion names")
    return list(value)


def _verify_count(environment_variable: str, actual: int) -> None:
    expected = os.environ.get(environment_variable)
    if expected is not None and int(expected) != actual:
        raise ValueError(f"{environment_variable}={expected} does not match the materialization record count {actual}")


def _motions_from_record(role: str, record_value: str) -> list[str]:
    record_path = Path(record_value).expanduser().resolve()
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"materialization record is not valid JSON: {record_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"materialization record must contain a JSON object: {record_path}")
    rows = payload.get("motions")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"materialization record must contain at least one motion: {record_path}")

    split_names: dict[str, list[str]] = {"train": [], "evaluation": [], "test": []}
    all_names: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"materialization motion row {index} must be an object")
        motion = row.get("motion")
        if not isinstance(motion, str) or not motion:
            raise ValueError(f"materialization motion row {index} has an empty or missing motion")
        split = str(row.get("split", "train"))
        if split not in split_names:
            raise ValueError(
                f"materialization motion row {index} has invalid split {split!r}; "
                "expected 'train', 'evaluation', or 'test'"
            )
        if motion in all_names:
            raise ValueError(f"materialization record contains duplicate motion: {motion}")
        all_names.add(motion)
        split_names[split].append(motion)

    train_names = _validated_motion_names(split_names["train"], source=str(record_path))
    validation_names = split_names["evaluation"] or train_names
    _verify_count("TERRA_SELECTION_SIZE", len(train_names))
    _verify_count("TERRA_VALIDATION_SIZE", len(validation_names))
    _verify_count("TERRA_TEST_SIZE", len(split_names["test"]))
    return list(train_names if role == "train" else validation_names)


def motion_selection(role: str, record_value: str = "") -> list[str]:
    """Resolve ordered train or validation motions without placing them in the environment."""

    normalized_role = str(role).strip().lower()
    if normalized_role not in _LEGACY_MOTION_ENV:
        raise ValueError("motion selection role must be 'train' or 'validation'")
    record_value = str(record_value).strip()
    if record_value:
        return _motions_from_record(normalized_role, record_value)

    environment_variable = _LEGACY_MOTION_ENV[normalized_role]
    encoded = os.environ.get(environment_variable)
    if encoded is None and normalized_role == "validation":
        environment_variable = _LEGACY_MOTION_ENV["train"]
        encoded = os.environ.get(environment_variable)
    if encoded is None:
        raise ValueError(
            f"set {MOTION_SELECTION_RECORD_ENV} to a materialization record or "
            f"set the legacy {environment_variable} variable"
        )
    try:
        decoded = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError(f"{environment_variable} is not valid JSON") from error
    return _validated_motion_names(decoded, source=environment_variable)


def motion_selection_hash(role: str, record_value: str = "") -> str:
    """Hash one resolved ordered cohort for persistent trajectory caching."""
    names = motion_selection(role, record_value)
    identity: object = names
    record_value = str(record_value).strip()
    if record_value:
        record_path = Path(record_value).expanduser().resolve()
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        rows = payload["motions"]
        requested_split = "train" if str(role).strip().lower() == "train" else "evaluation"
        selected = [row for row in rows if str(row.get("split", "train")) == requested_split]
        if requested_split == "evaluation" and not selected:
            selected = [row for row in rows if str(row.get("split", "train")) == "train"]
        identity = {
            "retargeting_method": payload.get("retargeting_method", "terra"),
            "source_caches": payload.get("source_caches"),
            "terrain_mode": payload.get("terrain_mode", "nonflat"),
            "motions": [
                {
                    "motion": row["motion"],
                    "dataset": row.get("dataset"),
                    "source_motion": row.get("source_motion"),
                    "segment_start_frame": row.get("segment_start_frame"),
                    "segment_end_frame_exclusive": row.get("segment_end_frame_exclusive"),
                }
                for row in selected
            ],
        }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def register_config_resolvers() -> None:
    """Register TERRA's config resolvers before Hydra composes a config."""

    OmegaConf.register_new_resolver("terra.motion_selection", motion_selection, replace=True)
    OmegaConf.register_new_resolver("terra.motion_selection_hash", motion_selection_hash, replace=True)


__all__ = [
    "MOTION_SELECTION_RECORD_ENV",
    "motion_selection",
    "motion_selection_hash",
    "register_config_resolvers",
]
