"""Define default weights, thresholds, and limits for TERRA contributions."""

from __future__ import annotations

import numpy as np

DEFAULT_TRUNK_SMOOTH_WEIGHT = 2.0  # base smoothing is 0.2


DEFAULT_TRUNK_Q_DIAG = 1e-3


DEFAULT_AXIAL_SMOOTH_WEIGHT = 10.0  # None leaves them at the base weight


DEFAULT_MTP_SMOOTH_WEIGHT = 10.0


#: Weight for each mimic-site orientation residual.
DEFAULT_ORIENT_WEIGHT = 0.5


#: Foot-site orientation weight used for flat-ground motions.
DEFAULT_FLAT_FOOT_ORIENT_WEIGHT = 0.1


#: Default stance constraint: ``"anchored"``, ``"omniretarget"``, or ``"off"``.
DEFAULT_FOOT_MODE = "anchored"


DEFAULT_CONTACT_SPEED = 0.30  # m/s, horizontal toe speed below which a foot may be planted


DEFAULT_CONTACT_HEIGHT = 0.06  # m above the supporting surface, likewise


DEFAULT_FOOT_ANCHOR_WEIGHT = 100.0


#: Weight for horizontal velocity above the stance-foot speed limit.
DEFAULT_FOOT_VELOCITY_WEIGHT = 150.0


#: Weight for horizontal displacement during sticking intervals.
DEFAULT_FOOT_VELOCITY_TRACKING_WEIGHT = 200.0


#: Horizontal stance-foot speed limit in meters per second.
DEFAULT_FOOT_VELOCITY_LIMIT = 0.25  # m/s


DEFAULT_FOOT_RAMP_FRAMES = 8  # ~80 ms at 100 Hz


#: Direct stance-height authority helps align the robot sole with source support.
DEFAULT_STANCE_HEIGHT_WEIGHT = 400.0
DEFAULT_STANCE_HEIGHT_MAX_RECOVERY_PER_ITER = 0.004


#: Chair-only morphology calibration.  The reconstructed seat follows the source body's
#: actual posterior surface, while MyoFullBody's rigid pelvis ellipsoids have a different
#: pelvis-to-glute distance.  A direct geom-distance target during detected seated rests
#: closes that representation gap without moving the reconstructed chair or affecting any
#: terrain that has no seat geom.
DEFAULT_SEAT_CONTACT_WEIGHT = 2000.0
DEFAULT_SEAT_CONTACT_MAX_RECOVERY_PER_ITER = 0.01
DEFAULT_SEAT_CONTACT_RAMP_FRAMES = 12
DEFAULT_SEAT_CONTACT_CLEARANCE = 0.0


#: Contact schedule used to activate foot anchoring.
DEFAULT_ANCHOR_GATE = "strict"


#: Weight for polynomial joint-coupler residuals.
DEFAULT_COUPLER_WEIGHT = 0.0


#: Static source frames prepended before solving.
DEFAULT_WARMUP_FRAMES = 30

#: QP backend used by default.
DEFAULT_SOLVER_BACKEND = "native_clarabel"


#: Finite SQP trust radius allowed only while solving the first source frame. The ordinary
#: 0.2 radius can be too small to reach the model's constrained articulated manifold from
#: its generic zero pose; the historical fallback removed the trust region altogether.
DEFAULT_INITIAL_STEP_SIZE = 1.0


#: Defensive trajectory-envelope gates. These are deliberately far outside ordinary gait
#: kinematics: they reject a numerically escaped solve without imposing a motion prior on
#: TERRA or changing any accepted solution.
DEFAULT_MAX_ROOT_STEP = 0.75  # metres in one source frame
DEFAULT_MAX_ROOT_SOURCE_DEVIATION = 5.0  # metres from the source pelvis


DEFAULT_SELF_COLLISION_TOLERANCE = 0.002  # m of separation to maintain


#: Inter-leg collision weight for flat-ground motions.
DEFAULT_FLAT_SELF_COLLISION_WEIGHT = 1000.0


#: Inter-leg collision weight for terrain motions.
DEFAULT_SELF_COLLISION_WEIGHT = 20000.0


#: Terrain non-penetration tolerance in meters.
DEFAULT_TERRAIN_PENETRATION_TOLERANCE = 0.0009


#: Maximum self-collision recovery per SQP iteration, in meters.
DEFAULT_SELF_COLLISION_MAX_RECOVERY_PER_ITER = 0.002


#: Fraction of source sole clearance requested during swing.
DEFAULT_CLEARANCE_FRACTION = 0.7


#: Maximum requested sole clearance above a surface, in meters.
DEFAULT_CLEARANCE_CAP = 0.15


DEFAULT_CLEARANCE_WEIGHT = 1500.0  # on a shortfall in metres


#: Maximum clearance recovery per SQP iteration, in meters.
DEFAULT_CLEARANCE_MAX_RECOVERY_PER_ITER = 0.01


#: Horizontal reach used for swing-clearance surface queries, in meters.
DEFAULT_CLEARANCE_LOOKAHEAD = 0.12


#: Minimum sole clearance during swing, in meters.
DEFAULT_MIN_SWING_CLEARANCE = 0.0


#: Frames used to taper minimum clearance at swing boundaries.
DEFAULT_CLEARANCE_RAMP_FRAMES = 12


#: Half-width of surface-height smoothing in seconds.
DEFAULT_CLEARANCE_SURFACE_RAMP_SECONDS = 0.05


#: Maximum swing-route landmark lift in meters.
DEFAULT_SWING_TARGET_LIFT = 0.06


#: Frames used to taper swing-route targets.
DEFAULT_SWING_TARGET_RAMP_FRAMES = 12


#: Weight for each active ankle or toe swing-route target.
DEFAULT_SWING_TARGET_WEIGHT = 1500.0


#: Maximum swing-route recovery per SQP iteration, in meters.
DEFAULT_SWING_TARGET_MAX_RECOVERY_PER_ITER = 0.004


#: Maximum penetration accepted by collision repair, in meters.
DEFAULT_POSTHOC_PEN_THRESHOLD = 0.005


#: Maximum collision run length eligible for interpolation.
DEFAULT_POSTHOC_MAX_RUN = 3


#: Relative tendon-jump threshold used by bounded repair.
DEFAULT_POSTHOC_TENDON_THRESHOLD = 0.05


#: Maximum full-pose tendon-repair span in frames.
DEFAULT_POSTHOC_TENDON_MAX_FRAMES = 15


#: Maximum tendon-local repair span in frames.
DEFAULT_POSTHOC_TENDON_MAX_LOCAL_FRAMES = 30


#: Allowed per-frame mean landmark-error regression in meters.
DEFAULT_POSTHOC_TRACKING_MEAN_REGRESSION = 0.01
#: Allowed per-frame maximum landmark-error regression in meters.
DEFAULT_POSTHOC_TRACKING_MAX_REGRESSION = 0.025


#: Maximum absolute sole-offset correction in meters.
MAX_SOLE_OFFSET = 0.05  # m


#: Maximum landmark-to-support distance for flat stance, in meters.
FLAT_STANCE_TOL = 0.02


#: Maximum accepted foot-orientation correction in radians.
MAX_FOOT_ORIENT_OFFSET = np.radians(30.0)
