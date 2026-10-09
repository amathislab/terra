"""Reconstruct and validate terrain from source joint motion.

The subpackage detects stance and seated support, fits per-level boxes,
staircases, ramps, and seats, and returns a
:class:`~loco_mujoco.core.terrain.TerrainSpec` in the solver coordinate frame.
"""

from terra.terrain.family import (
    FAMILY_EVIDENCE_HEIGHT_ONLY,
    FAMILY_EVIDENCE_MODES,
    FAMILY_EVIDENCE_PHYSICAL,
    SMPLH_NEUTRAL_FOOT_PITCH_DEG,
    STAIR_ASCENT_MIDPOINT_CLEARANCE,
    STAIR_DESCENT_EARLY_LATE_CLEARANCE,
    SURFACE_NORMAL_RESOLUTION_DEG,
    calibrate_neutral_foot_pitch,
    classify_terrain_family,
    surface_normal_evidence,
    swing_clearance_evidence,
)
from terra.terrain.fitting import fit_terrain_from_motion
from terra.terrain.metadata import TerrainMetadata
from terra.terrain.ramps import fit_ramp
from terra.terrain.reconstruction_profiles import (
    TERRA_FULL_PROFILE,
    TERRA_NO_PHYSICAL_CUES_PROFILE,
    TERRA_RECONSTRUCTION_PROFILES,
    TerraReconstructionProfile,
    resolve_terra_reconstruction_profile,
)
from terra.terrain.seats import (
    PELVIS_SEAT_OFFSET,
    SEAT_GEOM_PREFIX,
    SEAT_SPLIT_GAP,
    SeatRest,
    detect_seat_rests,
    drop_seated_contacts,
    fit_seat,
)
from terra.terrain.shapes import DEFAULT_CONTACT_MARGIN
from terra.terrain.stairs import STAIR_SOLE_HALF, fit_stair_flight
from terra.terrain.stance import (
    DEFAULT_CONTACT_JOINTS,
    DEFAULT_LEVEL_TOL,
    DEFAULT_STANCE_SPEED,
    StanceEvent,
    cluster_levels,
    detect_stance_events,
    joint_surface_offsets,
    paired_sole_offsets,
)
from terra.terrain.validation import validate_terrain

__all__ = [
    "DEFAULT_CONTACT_JOINTS",
    "DEFAULT_CONTACT_MARGIN",
    "DEFAULT_LEVEL_TOL",
    "DEFAULT_STANCE_SPEED",
    "FAMILY_EVIDENCE_HEIGHT_ONLY",
    "FAMILY_EVIDENCE_MODES",
    "FAMILY_EVIDENCE_PHYSICAL",
    "PELVIS_SEAT_OFFSET",
    "SEAT_GEOM_PREFIX",
    "SEAT_SPLIT_GAP",
    "SMPLH_NEUTRAL_FOOT_PITCH_DEG",
    "STAIR_ASCENT_MIDPOINT_CLEARANCE",
    "STAIR_DESCENT_EARLY_LATE_CLEARANCE",
    "STAIR_SOLE_HALF",
    "SURFACE_NORMAL_RESOLUTION_DEG",
    "TERRA_FULL_PROFILE",
    "TERRA_NO_PHYSICAL_CUES_PROFILE",
    "TERRA_RECONSTRUCTION_PROFILES",
    "SeatRest",
    "StanceEvent",
    "TerraReconstructionProfile",
    "TerrainMetadata",
    "calibrate_neutral_foot_pitch",
    "classify_terrain_family",
    "cluster_levels",
    "detect_seat_rests",
    "detect_stance_events",
    "drop_seated_contacts",
    "fit_ramp",
    "fit_seat",
    "fit_stair_flight",
    "fit_terrain_from_motion",
    "joint_surface_offsets",
    "paired_sole_offsets",
    "resolve_terra_reconstruction_profile",
    "surface_normal_evidence",
    "swing_clearance_evidence",
    "validate_terrain",
]
