"""Comparison retargeting methods distributed with TERRA."""

from terra.baselines.gmr import GMR_DEFAULTS
from terra.baselines.gmr import fit_motion as fit_gmr_baseline
from terra.baselines.omniretarget import fit_motion as fit_omniretarget_baseline
from terra.baselines.smpl import SMPL_DEFAULTS
from terra.baselines.smpl import fit_motion as fit_smpl_baseline

__all__ = ["GMR_DEFAULTS", "SMPL_DEFAULTS", "fit_gmr_baseline", "fit_omniretarget_baseline", "fit_smpl_baseline"]
