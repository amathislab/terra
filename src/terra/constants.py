"""Define MyoFullBody landmark mappings and robot joint groups."""

from __future__ import annotations

#: Supported environments mapped to ``musclemimic_models`` robot names.
ENV_TO_OMNIRETARGET_ROBOT = {
    "MyoFullBody": "myofullbody",
    "MjxMyoFullBody": "myofullbody",
}


#: Ordered calibration schema stored in ``shape_optimized.pkl`` for MyoFullBody.
#:
#: The shape cache predates named fields, so its position and rotation arrays follow
#: ``MyoFullBody.sites_for_mimic`` order.  Keep that order explicit at the integration
#: boundary and validate it in tests against the environment definition.  Each entry is
#: ``(mimic_site, SMPL-H joint, MyoFullBody body)``.
MYOFULLBODY_SITE_CALIBRATION = (
    ("pelvis_mimic", "Pelvis", "pelvis"),
    ("upper_body_mimic", "Spine", "lumbar1"),
    ("head_mimic", "Head", "head"),
    ("left_shoulder_mimic", "L_Shoulder", "humerus_l"),
    ("left_elbow_mimic", "L_Elbow", "ulna_l"),
    ("left_hand_mimic", "L_Wrist", "lunate_l"),
    ("right_shoulder_mimic", "R_Shoulder", "humerus_r"),
    ("right_elbow_mimic", "R_Elbow", "ulna_r"),
    ("right_hand_mimic", "R_Wrist", "lunate_r"),
    ("left_hip_mimic", "L_Hip", "femur_l"),
    ("left_knee_mimic", "L_Knee", "tibia_l"),
    ("left_ankle_mimic", "L_Ankle", "talus_l"),
    ("left_toes_mimic", "L_Toe", "toes_l"),
    ("right_hip_mimic", "R_Hip", "femur_r"),
    ("right_knee_mimic", "R_Knee", "tibia_r"),
    ("right_ankle_mimic", "R_Ankle", "talus_r"),
    ("right_toes_mimic", "R_Toe", "toes_r"),
)

#: Version recorded in new trajectory/terrain analyses.  This is deliberately independent
#: of dataset names: the correction describes the SMPL-H-to-robot landmark convention.
SITE_CALIBRATION_VERSION = "myofullbody_smplh_sites_v1"


#: SMPL-H joint names mapped to MyoFullBody body names.
SMPLH_TO_MYOFULLBODY = {
    "Pelvis": "pelvis",
    "L_Hip": "femur_l",
    "R_Hip": "femur_r",
    "L_Knee": "tibia_l",
    "R_Knee": "tibia_r",
    "L_Ankle": "talus_l",
    "R_Ankle": "talus_r",
    "L_Toe": "toes_l",
    "R_Toe": "toes_r",
    "L_Shoulder": "humerus_l",
    "R_Shoulder": "humerus_r",
    "L_Elbow": "ulna_l",
    "R_Elbow": "ulna_r",
    "L_Wrist": "lunate_l",
    "R_Wrist": "lunate_r",
    "Spine": "lumbar1",
    "Head": "head",
}


#: Joint-name prefixes used for trunk smoothing and regularization.
TRUNK_JOINT_PREFIXES = (
    "flex_extension",
    "lat_bending",
    "axial_rotation",
    "Abs_",
    "L1_",
    "L2_",
    "L3_",
    "L4_",
    "L5_",
)


#: Axial-rotation joints that receive dedicated temporal damping.
AXIAL_ROTATION_JOINTS = (
    "pro_sup_l",
    "pro_sup_r",
    "shoulder_rot_l",
    "shoulder_rot_r",
    "elv_angle_l",
    "elv_angle_r",
    "hip_rotation_l",
    "hip_rotation_r",
    "subtalar_angle_l",
    "subtalar_angle_r",
)


#: Metatarsophalangeal joints that receive dedicated temporal damping.
MTP_JOINTS = ("mtp_angle_l", "mtp_angle_r")


#: Foot labels mapped to bodies used by the foot-sticking constraint.
MYOFULLBODY_FOOT_LINKS = {"left_foot": "calcn_l", "right_foot": "calcn_r"}

#: Foot labels mapped to sites used for horizontal velocity control.
MYOFULLBODY_FOOT_SITES = {"left_foot": "LTOE", "right_foot": "RTOE"}


#: Collision-geom prefixes that make up each foot sole.
MYOFULLBODY_SOLE_GEOMS = {
    "l": ("l_foot_col", "l_bofoot_col"),
    "r": ("r_foot_col", "r_bofoot_col"),
}


#: Rigid collision ellipsoids representing the MyoFullBody gluteal support surface.
MYOFULLBODY_SEAT_GEOMS = ("r_pelvis_col", "l_pelvis_col")


#: Source foot landmarks mapped to left and right side labels.
FOOT_LANDMARK_SIDES = {"L_Toe": "l", "R_Toe": "r", "L_Ankle": "l", "R_Ankle": "r"}


#: Robot foot mimic sites mapped to left and right side labels.
FOOT_MIMIC_SITES = {
    "left_ankle_mimic": "l",
    "left_toes_mimic": "l",
    "right_ankle_mimic": "r",
    "right_toes_mimic": "r",
}


#: Segment names used to construct inter-leg collision pairs.
LEG_SEGMENTS = ("femur", "tibia", "calcn", "toes")


MYOFULLBODY_SELF_COLLISION_PAIRS = tuple((f"{a}_l", f"{b}_r") for a in LEG_SEGMENTS for b in LEG_SEGMENTS)


#: Left/right foot-body pairs checked by postprocessing collision repair.
MYOFULLBODY_POSTHOC_SELF_COLLISION_PAIRS = (
    ("calcn_l", "calcn_r"),
    ("toes_l", "toes_r"),
)


#: Infinite threshold used to disable OmniRetarget mat-height preprocessing.
NO_MAT_HEIGHT = float("inf")


#: Minimum support height considered raised at the start of a motion, in meters.
RAISED_START_HEIGHT = 0.04
