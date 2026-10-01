"""Explicit self-collision profiles for controlled backend comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import musclemimic_models

SELF_COLLISION_MODES = ("all", "lower_body_only", "none")

_LOWER_BODY_NAMES = frozenset(
    {
        "pelvis",
        "femur_r",
        "femur_l",
        "tibia_r",
        "tibia_l",
        "talus_r",
        "talus_l",
        "calcn_r",
        "calcn_l",
        "toes_r",
        "toes_l",
    }
)


def _pair_key(pair: Sequence[str]) -> tuple[str, str]:
    if isinstance(pair, (str, bytes)) or len(pair) != 2:
        raise ValueError("contact pairs must contain exactly two geometry names")
    first, second = (str(value).strip() for value in pair)
    if not first or not second or first == second:
        raise ValueError("contact pairs must contain two distinct non-empty geometry names")
    return tuple(sorted((first, second)))


def _disabled_pairs_for_spec(spec: mujoco.MjSpec, mode: str) -> list[list[str]]:
    """Return explicit pairs to delete while leaving terrain collisions intact."""
    if mode not in SELF_COLLISION_MODES:
        choices = ", ".join(SELF_COLLISION_MODES)
        raise ValueError(f"self_collision_mode must be one of {choices}; got {mode!r}")
    if mode == "all":
        return []

    geom_body = {
        geom.name: geom.parent.name
        for geom in spec.geoms
        if geom.name and geom.parent is not None
    }
    disabled = []
    for pair in spec.pairs:
        names = [pair.geomname1, pair.geomname2]
        if mode == "none" or not all(geom_body.get(name) in _LOWER_BODY_NAMES for name in names):
            disabled.append(names)
    return disabled


def disabled_self_collision_pairs(mode: str) -> list[list[str]]:
    """Resolve a profile against the maintained MyoFullBody collision model."""
    path = musclemimic_models.get_xml_path("myofullbody")
    return _disabled_pairs_for_spec(mujoco.MjSpec.from_file(str(path)), mode)


def merge_disabled_contact_pairs(
    configured: Sequence[Sequence[str]],
    *,
    self_collision_mode: str,
) -> list[list[str]]:
    """Combine targeted removals with a profile, without duplicate pairs."""
    merged: dict[tuple[str, str], list[str]] = {}
    for pair in [*configured, *disabled_self_collision_pairs(self_collision_mode)]:
        key = _pair_key(pair)
        merged.setdefault(key, list(pair))
    return list(merged.values())


__all__ = [
    "SELF_COLLISION_MODES",
    "disabled_self_collision_pairs",
    "merge_disabled_contact_pairs",
]
