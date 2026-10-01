"""Contribution-free baselines distributed and benchmarked with TERRA."""

from types import MappingProxyType

from terra.baselines._spec import BaselineSpec
from terra.baselines.gmr import GMR_BASELINE
from terra.baselines.gmr import fit_motion as fit_gmr_baseline
from terra.baselines.omniretarget import (
    OMNIRETARGET_BASELINE,
)
from terra.baselines.omniretarget import (
    fit_motion as fit_omniretarget_baseline,
)
from terra.baselines.smpl import SMPL_BASELINE
from terra.baselines.smpl import fit_motion as fit_smpl_baseline

BASELINES = MappingProxyType(
    {
        spec.key: spec
        for spec in (
            OMNIRETARGET_BASELINE,
            GMR_BASELINE,
            SMPL_BASELINE,
        )
    }
)


def baseline_spec(method: str) -> BaselineSpec:
    """Return the immutable definition for a named baseline."""
    try:
        return BASELINES[method]
    except KeyError as exc:
        raise ValueError(f"unknown baseline {method!r}; known: {sorted(BASELINES)}") from exc


__all__ = [
    "BASELINES",
    "GMR_BASELINE",
    "OMNIRETARGET_BASELINE",
    "SMPL_BASELINE",
    "BaselineSpec",
    "baseline_spec",
    "fit_gmr_baseline",
    "fit_omniretarget_baseline",
    "fit_smpl_baseline",
]
