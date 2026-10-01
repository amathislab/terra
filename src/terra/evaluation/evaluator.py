"""Single-pass terrain-aware retargeting evaluator.

One :func:`evaluate_method_motion` call loads one method-motion trajectory, measures terrain
interaction once, and derives every scalar, phase record, and per-frame annotation from that
shared state. Presentation scripts consume this result; they do not own scientific formulas.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from terra.artifacts import load_retarget_analysis
from terra.constants import SMPLH_TO_MYOFULLBODY
from terra.evaluation.registry import METRICS
from terra.evaluation.settings import BENCHMARK_THRESHOLDS, QUALITY_THRESHOLDS
from terra.evaluation.terrain import (
    TERRAIN_CONTACT_SOURCE_JOINTS,
    Measurement,
    measure,
    signed_distance_to_terrain_boxes,
    source_shape_path,
    to_json,
    traj_paths,
)
from terra.evaluation.timeline import load_source_motion, load_trajectory_timeline
from terra.terrain.stance import DEFAULT_CONTACT_JOINTS


@dataclass(frozen=True)
class Thresholds:
    """Continuous benchmark thresholds; phase thresholds live in ``Tolerances``."""

    penetration_m: float = BENCHMARK_THRESHOLDS["penetration_m"]
    support_penetration_m: float = BENCHMARK_THRESHOLDS["support_penetration_m"]
    foot_contact_height_m: float = BENCHMARK_THRESHOLDS["foot_contact_height_m"]
    source_contact_speed_m_s: float = BENCHMARK_THRESHOLDS["source_contact_speed_m_s"]
    skating_speed_m_s: float = BENCHMARK_THRESHOLDS["skating_speed_m_s"]
    terrain_contact_m: float = BENCHMARK_THRESHOLDS["terrain_contact_m"]
    joint_limit: float = BENCHMARK_THRESHOLDS["joint_limit"]
    joint_limit_linear_m: float = BENCHMARK_THRESHOLDS["joint_limit_linear_m"]
    tendon_jump: float = BENCHMARK_THRESHOLDS["tendon_jump"]
    self_collision_m: float = BENCHMARK_THRESHOLDS["self_collision_m"]

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        for name in (
            "foot_contact_height_m",
            "source_contact_speed_m_s",
            "skating_speed_m_s",
            "terrain_contact_m",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


JOINT_LIMIT_SENSITIVITY_TOLERANCES_RAD = (1e-5, 1e-4, 1e-3, 1e-2)


def joint_limit_sensitivity_key(tolerance_rad: float) -> str:
    return f"joint_limit_duration_pct_at_{tolerance_rad:.0e}_rad".replace("-", "m")


IDENTITY_FIELDS = (
    "method",
    "motion_class",
    "motion",
    "frames",
    "fps",
    "common_start_s",
    "common_end_s",
    "common_duration_s",
    "source_contact_s",
    "source_support_s",
    "desired_contact_point_s",
)
DIAGNOSTIC_FIELDS = (
    "retargeting_per_landmark_rmse_mm_json",
    "coupler_worst",
    "n_couplers",
    "joint_step_worst",
    "joint_step_frame",
    "solver_solved_frames",
    "timing_source",
    "timing_context_json",
    "penetration_frame_depth_sum_mm",
    "penetration_frame_depth_sq_sum_mm2",
    "penetration_frame_depth_n",
    "skating_frame_velocity_sum_m_s",
    "skating_frame_velocity_sq_sum_m2_s2",
    "skating_frame_velocity_n",
    "source_contact_transitions",
    "source_support_transitions",
    "desired_contact_transitions",
    *(joint_limit_sensitivity_key(tol) for tol in JOINT_LIMIT_SENSITIVITY_TOLERANCES_RAD),
)
PER_MOTION_FIELDS = IDENTITY_FIELDS + tuple(spec.key for spec in METRICS) + DIAGNOSTIC_FIELDS + ("error",)

# The order matches ``terra.terrain.DEFAULT_CONTACT_JOINTS`` exactly. OmniRetarget
# evaluates toe-link motion during desired toe sticking. TERRA's benchmark uses the
# project-wide four-probe contact convention, so each source probe is compared with its
# directly mapped robot body instead of first collapsing toe and ankle evidence by side.
SKATING_ROBOT_POINTS = tuple(("body", SMPLH_TO_MYOFULLBODY[joint]) for joint in DEFAULT_CONTACT_JOINTS)


@dataclass
class UnifiedMeasurement:
    """All downstream views derived from one method-motion measurement."""

    row: dict[str, Any]
    quality: dict[str, Any]
    annotations: dict[str, np.ndarray]
    detail: dict[str, Any]


_MODEL = None


def model():
    """Return the exact finger-disabled model used to produce the trajectories.

    Constructing the environment is intentional: ``MyoFullBody`` owns the complete
    finger-disabling transform, including removal of finger muscles and tendons.
    Disabling only finger joints would compile a 424-tendon model rather than the
    362-tendon retargeting and policy model.
    """
    global _MODEL
    if _MODEL is None:
        from musclemimic.environments.humanoids.myofullbody import MyoFullBody

        _MODEL = MyoFullBody(disable_fingers=True).model
    return _MODEL


def cache_file(cache_root: Path, subdir: str, motion: str, suffix: str = ".npz") -> Path:
    return cache_root / "MyoFullBody" / subdir / f"{motion}{suffix}"


def _scalar(value):
    value = np.asarray(value)
    return value.item() if value.shape == () else value


def load_timeline(
    cache_root: Path,
    subdir: str,
    motion: str,
    *,
    source_root: Path | None = None,
):
    """Read a complete timeline or reconstruct one from partial analysis metadata."""
    trajectory_path = cache_file(cache_root, subdir, motion)
    analysis_path = cache_file(cache_root, subdir, motion, "_analysis.npz")
    return load_trajectory_timeline(
        trajectory_path,
        analysis_path,
        motion,
        source_root=source_root,
    )


def load_site_calibration_state(cache_root: Path, subdir: str, motion: str) -> bool | None:
    """Return the source-landmark convention recorded by a retargeting run."""
    analysis_path = cache_file(cache_root, subdir, motion, "_analysis.npz")
    if not analysis_path.exists():
        return None
    with np.load(analysis_path, allow_pickle=True) as analysis:
        if "resolved_config_json" not in analysis.files:
            return None
        config = json.loads(str(_scalar(analysis["resolved_config_json"])))
    value = config.get("calibrate_sites")
    return None if value is None else bool(value)


def _joint_limit_excess(metric_model, qpos: np.ndarray, joint_type) -> np.ndarray:
    """Maximum SI-unit range excess for one scalar MuJoCo joint type."""
    excess = np.zeros(len(qpos), dtype=float)
    for joint_id in range(metric_model.njnt):
        if metric_model.joint(joint_id).type != joint_type:
            continue
        if not bool(metric_model.jnt_limited[joint_id]):
            continue
        low, high = metric_model.jnt_range[joint_id]
        values = qpos[:, metric_model.jnt_qposadr[joint_id]]
        excess = np.maximum(excess, np.maximum(low - values, values - high))
    return np.maximum(excess, 0.0)


def joint_limit_excess_rad(metric_model, qpos: np.ndarray) -> np.ndarray:
    """Maximum angular range excess over limited hinge joints at every frame."""
    import mujoco

    return _joint_limit_excess(metric_model, qpos, mujoco.mjtJoint.mjJNT_HINGE)


def joint_limit_excess_m(metric_model, qpos: np.ndarray) -> np.ndarray:
    """Maximum linear range excess over limited slide joints at every frame."""
    import mujoco

    return _joint_limit_excess(metric_model, qpos, mujoco.mjtJoint.mjJNT_SLIDE)


def joint_limit_frame_mask(
    metric_model,
    qpos: np.ndarray,
    tol: float,
    linear_tol_m: float | None = None,
) -> np.ndarray:
    """Frames exceeding either angular or linear scalar-joint limits."""
    linear_tol_m = tol if linear_tol_m is None else linear_tol_m
    return (joint_limit_excess_rad(metric_model, qpos) > tol) | (
        joint_limit_excess_m(metric_model, qpos) > linear_tol_m
    )


def source_contact_mask(measurement: Measurement) -> np.ndarray:
    mask = np.zeros((measurement.n_frames, 2), dtype=bool)
    side_index = {"left": 0, "right": 1}
    for stance in measurement.stances:
        mask[stance.start : stance.end, side_index[stance.side]] = True
    return mask


def _boolean_transition_count(mask: np.ndarray) -> int:
    """Count Boolean state changes across time, summed over independent columns."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 1:
        return int(np.count_nonzero(mask[1:] != mask[:-1]))
    if mask.ndim == 2:
        return int(np.count_nonzero(mask[1:] != mask[:-1]))
    raise ValueError("Boolean transition masks must have shape (T,) or (T, points)")


def foot_contact_metrics(
    clearance_m: np.ndarray,
    foot_position_m: np.ndarray,
    source_contact: np.ndarray,
    fps: float,
    thresholds: Thresholds,
    times: np.ndarray | None = None,
    common: tuple[float, float] | None = None,
    source_support: np.ndarray | None = None,
    *,
    return_annotations: bool = False,
):
    """Source-conditioned skating, floating, and support penetration in one calculation."""
    clearance_m = np.asarray(clearance_m, dtype=float)
    foot_position_m = np.asarray(foot_position_m, dtype=float)
    source_contact = np.asarray(source_contact, dtype=bool)
    if clearance_m.ndim != 2 or source_contact.ndim != 2 or clearance_m.shape[0] != source_contact.shape[0]:
        raise ValueError("clearance and source_contact must have matching frame counts and two dimensions")
    if source_support is None:
        if source_contact.shape != clearance_m.shape:
            raise ValueError("source_support is required when skating probes differ from support feet")
        source_support = source_contact
    else:
        source_support = np.asarray(source_support, dtype=bool)
    if source_support.shape != clearance_m.shape:
        raise ValueError("source_support and clearance must have matching (T, feet) shapes")
    if foot_position_m.shape != (*source_contact.shape, 3):
        raise ValueError("foot_position must have shape (T, skating probes, 3)")
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError(f"fps must be positive and finite, got {fps}")

    from musclemimic.utils.retarget.benchmark_timeline import sample_durations

    if times is None:
        times = np.arange(len(clearance_m), dtype=float) / fps
    times = np.asarray(times, dtype=float)
    if len(times) != len(clearance_m):
        raise ValueError("times and foot arrays must have matching lengths")
    common = common or (float(times[0]), float(times[-1] + 1.0 / fps))
    weights = sample_durations(times, *common)
    if float(weights.sum()) <= 0:
        raise ValueError("common interval contains no output samples")
    delta_t = np.diff(times)
    if np.any(delta_t <= 0):
        raise ValueError("times must be strictly increasing")
    velocity = np.zeros(source_contact.shape, dtype=float)
    velocity[1:] = np.linalg.norm(np.diff(foot_position_m[:, :, :2], axis=0), axis=2) / delta_t[:, None]
    active = weights > 0
    velocity_valid = np.zeros(len(clearance_m), dtype=bool)
    velocity_valid[1:] = active[1:] & active[:-1]
    skating_per_foot = source_contact & velocity_valid[:, None] & (velocity > thresholds.skating_speed_m_s)
    skating = np.any(skating_per_foot, axis=1)
    desired_sticking = np.any(source_contact, axis=1)
    desired_sticking_s = float(np.dot(desired_sticking, weights))
    desired_support_s = float(np.sum(weights[:, None] * source_support))
    floating = source_support & (clearance_m > thresholds.foot_contact_height_m)
    penetrating = source_support & (clearance_m < -thresholds.support_penetration_m)
    invalid = floating | penetrating
    support_clearance = clearance_m[source_support & active[:, None]]
    skating_velocity_per_frame = np.max(np.where(skating_per_foot, velocity, 0.0), axis=1)
    skating_velocity = skating_velocity_per_frame[skating_velocity_per_frame > 0.0]
    values = {
        "source_contact_s": desired_sticking_s,
        "source_support_s": desired_support_s,
        "source_contact_transitions": _boolean_transition_count(desired_sticking),
        "source_support_transitions": _boolean_transition_count(source_support),
        "skating_duration_pct": (
            100.0 * float(np.dot(skating, weights)) / desired_sticking_s if desired_sticking_s > 0 else float("nan")
        ),
        "skating_max_velocity_m_s": (
            float(skating_velocity.mean())
            if skating_velocity.size
            else (0.0 if desired_sticking_s > 0 else float("nan"))
        ),
        "skating_frame_velocity_sum_m_s": float(skating_velocity.sum()),
        "skating_frame_velocity_sq_sum_m2_s2": float(np.dot(skating_velocity, skating_velocity)),
        "skating_frame_velocity_n": int(skating_velocity.size),
        "floating_duration_pct": (
            100.0 * float(np.sum(weights[:, None] * floating)) / desired_support_s
            if desired_support_s > 0
            else float("nan")
        ),
        "floating_max_height_mm": (
            max(0.0, float(support_clearance.max())) * 1000.0 if support_clearance.size else float("nan")
        ),
        "support_penetration_duration_pct": (
            100.0 * float(np.sum(weights[:, None] * penetrating)) / desired_support_s
            if desired_support_s > 0
            else float("nan")
        ),
        "support_penetration_max_depth_mm": (
            max(0.0, -float(support_clearance.min())) * 1000.0 if support_clearance.size else float("nan")
        ),
        "invalid_support_duration_pct": (
            100.0 * float(np.sum(weights[:, None] * invalid)) / desired_support_s
            if desired_support_s > 0
            else float("nan")
        ),
    }
    if not return_annotations:
        return values
    annotations = {
        "skating": skating,
        "skating_per_foot": skating_per_foot,
        "support_floating": floating,
        "support_penetrating": penetrating,
        "invalid_support": invalid,
        "foot_velocity_m_s": velocity,
    }
    return values, annotations


def benchmark_self_penetration(measurement: Measurement) -> np.ndarray:
    """Per-frame penetration for the calibrated contralateral-leg collision scope."""
    return np.asarray(measurement.per_frame["interleg_selfpen"], dtype=float)


def source_probe_contact_on_output(
    joints: np.ndarray,
    source_fps: float,
    output_times: np.ndarray,
    speed_m_s: float = 0.300,
) -> np.ndarray:
    """Four separate kinematic probe-contact intervals mapped to output time.

    Columns follow ``DEFAULT_CONTACT_JOINTS``. Keeping the probes separate is
    essential for skating: a planted ankle must not turn a moving toe into a desired
    toe-sticking sample (or vice versa).
    """
    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS

    from musclemimic.utils.retarget.benchmark_timeline import map_boolean_intervals
    from terra.evaluation.terrain import _not_while_seated
    from terra.terrain import detect_seat_rests, detect_stance_events

    if source_fps <= 0 or not np.isfinite(source_fps):
        raise ValueError(f"source_fps must be positive and finite, got {source_fps}")
    if speed_m_s <= 0 or not np.isfinite(speed_m_s):
        raise ValueError(f"speed_m_s must be positive and finite, got {speed_m_s}")
    names = list(SMPLH_DEMO_JOINTS)
    events = detect_stance_events(
        joints,
        names,
        source_fps,
        contact_joints=DEFAULT_CONTACT_JOINTS,
        speed_ms=speed_m_s,
    )
    rests = detect_seat_rests(joints, names, source_fps)
    mask = np.zeros((len(joints), len(DEFAULT_CONTACT_JOINTS)), dtype=bool)
    for probe_index, joint in enumerate(DEFAULT_CONTACT_JOINTS):
        active = np.zeros(len(joints), dtype=bool)
        for event in events:
            if event.joint == joint:
                active[max(0, event.start) : min(len(joints), event.end)] = True
        padded = np.pad(active.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        source_stances = [
            (int(start), int(end))
            for start, end in zip(
                np.flatnonzero(changes == 1),
                np.flatnonzero(changes == -1),
                strict=True,
            )
            if (end - start) / source_fps >= QUALITY_THRESHOLDS["min_stance_s"]
        ]
        source_stances, _dropped = _not_while_seated(source_stances, rests)
        for start, end in source_stances:
            mask[start:end, probe_index] = True
    return np.column_stack(
        [map_boolean_intervals(mask[:, probe], source_fps, output_times) for probe in range(mask.shape[1])]
    )


def source_contact_on_output(
    joints: np.ndarray,
    source_fps: float,
    output_times: np.ndarray,
    speed_m_s: float = 0.300,
) -> np.ndarray:
    """Four-joint kinematic contact collapsed to left/right support intervals."""
    probes = source_probe_contact_on_output(joints, source_fps, output_times, speed_m_s=speed_m_s)
    return np.column_stack((probes[:, 0] | probes[:, 2], probes[:, 1] | probes[:, 3]))


def source_floor_contact_on_output(
    joints: np.ndarray,
    source_fps: float,
    output_times: np.ndarray,
) -> np.ndarray:
    """Four-joint kinematic support used for flat-floor contact preservation."""
    return source_contact_on_output(joints, source_fps, output_times)


def robot_point_positions(metric_model, qpos: np.ndarray, points: tuple[tuple[str, str], ...]) -> np.ndarray:
    """World positions of named robot body origins or sites at every frame."""
    import mujoco

    object_types = {"body": mujoco.mjtObj.mjOBJ_BODY, "site": mujoco.mjtObj.mjOBJ_SITE}
    resolved = []
    for kind, name in points:
        if kind not in object_types:
            raise ValueError(f"unsupported metric point kind {kind!r}")
        point_id = mujoco.mj_name2id(metric_model, object_types[kind], name)
        if point_id < 0:
            raise ValueError(f"metric robot {kind} {name!r} is absent from the model")
        resolved.append((kind, point_id))
    data = mujoco.MjData(metric_model)
    positions = np.empty((len(qpos), len(resolved), 3), dtype=float)
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(metric_model, data)
        positions[frame] = [
            data.xpos[point_id] if kind == "body" else data.site_xpos[point_id] for kind, point_id in resolved
        ]
    return positions


def robot_body_positions(metric_model, qpos: np.ndarray, body_names: tuple[str, ...]) -> np.ndarray:
    return robot_point_positions(metric_model, qpos, tuple(("body", name) for name in body_names))


def floor_contact_preservation(
    clearance_m: np.ndarray,
    source_contact: np.ndarray,
    thresholds: Thresholds,
    times: np.ndarray,
    common: tuple[float, float],
) -> dict[str, float]:
    """Preserved source foot-floor contact over desired foot-time on flat scenes."""
    clearance_m = np.asarray(clearance_m, dtype=float)
    source_contact = np.asarray(source_contact, dtype=bool)
    times = np.asarray(times, dtype=float)
    if clearance_m.shape != source_contact.shape or clearance_m.ndim != 2:
        raise ValueError("clearance and source_contact must have matching (T, feet) shapes")
    if len(times) != len(clearance_m):
        raise ValueError("times and foot arrays must have matching lengths")
    from musclemimic.utils.retarget.benchmark_timeline import sample_durations

    weights = sample_durations(times, *common)
    actual = (clearance_m >= -thresholds.support_penetration_m) & (clearance_m <= thresholds.foot_contact_height_m)
    desired_s = float(np.sum(weights[:, None] * source_contact))
    preserved_s = float(np.sum(weights[:, None] * source_contact * actual))
    return {
        "desired_contact_point_s": desired_s,
        "desired_contact_transitions": _boolean_transition_count(source_contact),
        "contact_preservation_pct": 100.0 * preserved_s / desired_s if desired_s > 0 else float("nan"),
    }


def terrain_contact_preservation(
    source_points_m: np.ndarray,
    robot_distance_m: np.ndarray,
    source_terrain,
    contact_threshold_m: float,
    times: np.ndarray,
    common: tuple[float, float],
) -> dict[str, float]:
    """OmniRetarget terrain-contact preservation over desired-contact time."""
    source_points_m = np.asarray(source_points_m, dtype=float)
    robot_distance_m = np.asarray(robot_distance_m, dtype=float)
    if source_points_m.ndim != 3 or robot_distance_m.shape != source_points_m.shape[:-1]:
        raise ValueError("source points and robot distances must have matching (T, points) shapes")
    if contact_threshold_m <= 0 or not np.isfinite(contact_threshold_m):
        raise ValueError("contact_threshold_m must be positive and finite")
    from musclemimic.utils.retarget.benchmark_timeline import sample_durations

    weights = sample_durations(np.asarray(times, dtype=float), *common)
    source_distance = signed_distance_to_terrain_boxes(source_points_m, source_terrain)
    desired = source_distance <= contact_threshold_m
    actual = robot_distance_m <= contact_threshold_m
    desired_frame = np.any(desired, axis=1)
    preserved_frame = desired_frame & np.all(~desired | actual, axis=1)
    desired_s = float(np.dot(weights, desired_frame))
    preserved_s = float(np.dot(weights, preserved_frame))
    return {
        "desired_contact_point_s": desired_s,
        "desired_contact_transitions": _boolean_transition_count(desired_frame),
        "contact_preservation_pct": 100.0 * preserved_s / desired_s if desired_s > 0 else 100.0,
    }


def common_retargeting_rmse(metric_model, qpos, output_times, source_joints, source_fps, active):
    """Fixed-timestamp pelvis-world and pelvis-relative landmark RMSEs."""
    import mujoco
    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS

    from musclemimic.utils.retarget.benchmark_timeline import interpolate_samples
    from terra.constants import SMPLH_TO_MYOFULLBODY

    names = list(SMPLH_DEMO_JOINTS)
    source = source_joints[:, [names.index(name) for name in SMPLH_TO_MYOFULLBODY]]
    targets = interpolate_samples(source, source_fps, output_times)
    ids = [mujoco.mj_name2id(metric_model, mujoco.mjtObj.mjOBJ_BODY, body) for body in SMPLH_TO_MYOFULLBODY.values()]
    data = mujoco.MjData(metric_model)
    deltas = []
    for frame in np.flatnonzero(active):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(metric_model, data)
        deltas.append(data.xpos[ids].copy() - targets[frame])
    if not deltas:
        raise ValueError("no RMSE samples in the common interval")
    delta = np.asarray(deltas)
    squared = np.sum(delta * delta, axis=2)
    per_landmark = np.sqrt(np.mean(squared, axis=0)) * 1000.0
    pelvis_world = float(np.sqrt(np.mean(squared[:, 0])) * 1000.0)
    relative = delta - delta[:, :1]
    pelvis_relative = float(np.sqrt(np.mean(np.sum(relative[:, 1:] * relative[:, 1:], axis=2))) * 1000.0)
    return pelvis_world, pelvis_relative, dict(zip(SMPLH_TO_MYOFULLBODY, per_landmark.tolist(), strict=True))


def phase_metrics(
    measurement: Measurement,
    active: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Derive phase-denominated slip and swing metrics from the shared terrain measurement."""
    tol = measurement.tol
    active = np.ones(measurement.n_frames, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if active.shape != (measurement.n_frames,):
        raise ValueError("phase active mask must match measurement frames")

    def contained(phase) -> bool:
        return phase.end > phase.start and bool(np.all(active[phase.start : phase.end]))

    stances = [stance for stance in measurement.stances if contained(stance)]
    swings = [swing for swing in measurement.swings if contained(swing)]
    measurable_swings = [swing for swing in swings if swing.measurable()]
    ratios = [swing.peak / swing.src_peak for swing in measurable_swings if np.isfinite(swing.peak)]
    dragging = [swing for swing in measurable_swings if swing.dragging(tol)]
    scraping = [swing for swing in swings if swing.scraping(tol)]
    slipping = [stance for stance in stances if stance.excess > tol.slip_tol]
    values = {
        "stance_slip_failure_pct": 100.0 * len(slipping) / len(stances) if stances else float("nan"),
        "swing_clearance_failure_pct": (
            100.0 * len(dragging) / len(measurable_swings) if measurable_swings else float("nan")
        ),
        "swing_scrape_failure_pct": (100.0 * len(scraping) / len(swings) if swings else float("nan")),
        "swing_clearance_ratio_median": float(np.median(ratios)) if ratios else float("nan"),
    }
    scraping_frames = np.zeros(measurement.n_frames, dtype=bool)
    dragging_frames = np.zeros(measurement.n_frames, dtype=bool)
    slipping_frames = np.zeros(measurement.n_frames, dtype=bool)
    forefoot_up_frames = np.zeros(measurement.n_frames, dtype=bool)
    hindfoot_up_frames = np.zeros(measurement.n_frames, dtype=bool)
    for swing in scraping:
        scraping_frames[swing.start : swing.end] = True
    for swing in dragging:
        dragging_frames[swing.start : swing.end] = True
    for stance in slipping:
        slipping_frames[stance.start : stance.slip_end or stance.end] = True
    for stance in stances:
        if stance.forefoot_up(tol):
            forefoot_up_frames[stance.start : stance.end] = True
        if stance.hindfoot_up(tol):
            hindfoot_up_frames[stance.start : stance.end] = True
    return values, {
        "forefoot_up": forefoot_up_frames,
        "hindfoot_up": hindfoot_up_frames,
        "stance_slip_failure": slipping_frames,
        "swing_clearance_failure": dragging_frames,
        "swing_scraping": scraping_frames,
    }


def coupler_metrics(metric_model, qpos: np.ndarray, active: np.ndarray) -> dict[str, Any]:
    """Joint-equality validity using the selected polynomial residual formula."""
    from terra._musclemimic import joint_couplers

    couplers = joint_couplers(metric_model)
    frames = np.asarray(qpos, dtype=float)[np.asarray(active, dtype=bool)]
    if not couplers:
        return {
            "coupler_mean_residual_deg": 0.0,
            "coupler_max_residual_deg": 0.0,
            "coupler_worst": None,
            "n_couplers": 0,
        }
    if not len(frames):
        raise ValueError("no active frames for joint-coupler validity")
    residuals = []
    maxima = []
    for dependent, independent, polynomial in couplers:
        x = frames[:, independent]
        y = frames[:, dependent]
        predicted = sum(float(coefficient) * x**degree for degree, coefficient in enumerate(polynomial))
        residual = np.degrees(np.abs(y - predicted))
        residuals.append(residual)
        maxima.append((float(residual.max()), int(dependent)))
    values = np.stack(residuals)
    worst_value, worst_address = max(maxima)
    worst_name = next(
        (
            metric_model.joint(joint_id).name
            for joint_id in range(metric_model.njnt)
            if int(metric_model.jnt_qposadr[joint_id]) == worst_address
        ),
        None,
    )
    return {
        "coupler_mean_residual_deg": float(values.mean()),
        "coupler_max_residual_deg": worst_value,
        "coupler_worst": worst_name,
        "n_couplers": len(couplers),
    }


def frame_step_metrics(
    metric_model,
    qpos: np.ndarray,
    transition_active: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Joint and free-root frame-step discontinuity on the common active transitions."""
    import mujoco

    from musclemimic.utils.retarget.msk_metrics import root_step_series

    qpos = np.asarray(qpos, dtype=float)
    transition_active = np.asarray(transition_active, dtype=bool)
    if transition_active.shape != (len(qpos),):
        raise ValueError("transition_active must match qpos frames")
    indices = [
        int(metric_model.jnt_qposadr[joint_id])
        for joint_id in range(metric_model.njnt)
        if metric_model.joint(joint_id).type == mujoco.mjtJoint.mjJNT_HINGE
    ]
    names = [
        metric_model.joint(joint_id).name
        for joint_id in range(metric_model.njnt)
        if metric_model.joint(joint_id).type == mujoco.mjtJoint.mjJNT_HINGE
    ]
    per_frame_joint = np.zeros(len(qpos), dtype=float)
    if len(qpos) > 1 and indices:
        steps = np.degrees(np.abs(np.diff(qpos[:, indices], axis=0)))
        per_frame_joint[1:] = steps.max(axis=1)
        selected = steps[transition_active[1:]]
        if selected.size:
            flat = int(np.argmax(selected))
            _row, joint_column = np.unravel_index(flat, selected.shape)
            joint_max = float(selected.ravel()[flat])
            joint_p999 = float(np.percentile(selected, 99.9))
            joint_worst = names[joint_column]
            active_frames = np.flatnonzero(transition_active)
            selected_rows = np.max(steps, axis=1)[transition_active[1:]]
            joint_frame = int(active_frames[int(np.argmax(selected_rows))])
        else:
            joint_max = joint_p999 = 0.0
            joint_worst = None
            joint_frame = -1
    else:
        joint_max = joint_p999 = 0.0
        joint_worst = None
        joint_frame = -1
    root_translation, root_rotation = root_step_series(metric_model, qpos)
    active_root_translation = root_translation[transition_active]
    active_root_rotation = root_rotation[transition_active]
    values = {
        "joint_step_max_deg": joint_max,
        "joint_step_p999_deg": joint_p999,
        "joint_step_worst": joint_worst,
        "joint_step_frame": joint_frame,
        "root_translation_step_max_mm": float(active_root_translation.max(initial=0.0)),
        "root_translation_step_p999_mm": (
            float(np.percentile(active_root_translation, 99.9)) if active_root_translation.size else 0.0
        ),
        "root_rotation_step_max_deg": float(active_root_rotation.max(initial=0.0)),
        "root_rotation_step_p999_deg": (
            float(np.percentile(active_root_rotation, 99.9)) if active_root_rotation.size else 0.0
        ),
    }
    annotations = {
        "joint_step_deg": per_frame_joint,
        "root_translation_step_mm": root_translation,
        "root_rotation_step_deg": root_rotation,
    }
    return values, annotations


def tendon_discontinuity_series(
    metric_model,
    qpos: np.ndarray,
    active: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive relative tendon events and physical steps in one MuJoCo traversal.

    This is the exact ``msk_metrics.tendon_stats_and_events`` update rule, exposed as
    per-frame series so the common timeline can be applied before maxima and durations.
    """
    import mujoco

    qpos = np.asarray(qpos, dtype=float)
    active = np.ones(len(qpos), dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if active.shape != (len(qpos),):
        raise ValueError("active must match qpos frames")
    relative_event = np.zeros(len(qpos), dtype=float)
    physical_step_mm = np.zeros(len(qpos), dtype=float)
    data = mujoco.MjData(metric_model)
    previous = None
    ema = np.zeros(metric_model.ntendon, dtype=float)
    ema_initialized = False
    rest = np.asarray(metric_model.tendon_length0, dtype=float)
    for frame, position in enumerate(qpos):
        if not active[frame]:
            previous = None
            ema.fill(0.0)
            ema_initialized = False
            continue
        data.qpos[:] = position
        mujoco.mj_forward(metric_model, data)
        current = data.ten_length.copy()
        if previous is not None and metric_model.ntendon:
            absolute = np.abs(current - previous)
            relative = absolute / np.maximum(rest, 1e-6)
            physical_step_mm[frame] = float(absolute.max(initial=0.0)) * 1000.0
            if not ema_initialized:
                ema[:] = relative
                ema_initialized = True
            else:
                ema[:] = 0.99 * ema + 0.01 * relative
            mask = relative > np.maximum(10.0 * ema, 1e-3)
            if np.any(mask):
                relative_event[frame] = float(np.max(relative[mask]))
        previous = current
    return relative_event, physical_step_mm


def solver_throughput(analysis_path: Path, common_duration_s: float) -> dict[str, Any]:
    """Read producer timing, preferring the uniform benchmark timer when available.

    Existing final artifacts predate the uniform timer but record ``retarget_fps``.
    Keeping that fallback makes them re-evaluable without rerunning a retargeter.
    """
    analysis = load_retarget_analysis(analysis_path)
    context = analysis.get("benchmark_timing_context")
    timing_context_json = json.dumps(context, sort_keys=True) if isinstance(context, dict) else ""
    has_elapsed = "benchmark_retarget_elapsed_s" in analysis
    has_benchmark_frames = "benchmark_solved_frames" in analysis
    if has_elapsed != has_benchmark_frames:
        raise ValueError("benchmark timing metadata is incomplete")
    if has_elapsed:
        elapsed_s = float(_scalar(analysis["benchmark_retarget_elapsed_s"]))
        solved_frames = int(_scalar(analysis["benchmark_solved_frames"]))
        timing_source = "uniform_retarget_call"
        if not np.isfinite(elapsed_s) or elapsed_s <= 0:
            raise ValueError(f"benchmark_retarget_elapsed_s must be positive and finite, got {elapsed_s}")
        if solved_frames < 1:
            raise ValueError(f"benchmark_solved_frames must be positive, got {solved_frames}")
        retarget_fps = solved_frames / elapsed_s
    elif "retarget_fps" in analysis:
        retarget_fps = float(_scalar(analysis["retarget_fps"]))
        solved_frames = int(np.asarray(analysis["pos_error"]).shape[0]) if "pos_error" in analysis else None
        timing_source = "producer_retarget_fps"
        timing_context_json = ""
        if not np.isfinite(retarget_fps) or retarget_fps <= 0:
            raise ValueError(f"retarget_fps must be positive and finite, got {retarget_fps}")
        if solved_frames is not None and solved_frames < 1:
            raise ValueError(f"pos_error must contain at least one solved frame, got {solved_frames}")
    else:
        return {
            "solver_native_fps": None,
            "solver_seconds_per_motion_second": None,
            "t_frame_s": None,
            "solver_solved_frames": None,
            "timing_source": "missing",
            "timing_context_json": timing_context_json,
        }
    elapsed_s = solved_frames / retarget_fps if solved_frames is not None else None
    cost = elapsed_s / common_duration_s if elapsed_s is not None and common_duration_s > 0 else None
    return {
        "solver_native_fps": retarget_fps,
        "solver_seconds_per_motion_second": cost,
        "t_frame_s": 1.0 / retarget_fps,
        "solver_solved_frames": solved_frames,
        "timing_source": timing_source,
        "timing_context_json": timing_context_json,
    }


def evaluate_method_motion(
    *,
    method_label: str,
    method_subdir: str,
    motion_class: str,
    motion: str,
    terrain_method: str | None,
    force_flat: bool,
    thresholds: Thresholds,
    cache_root: Path,
    source_root: Path | None,
    all_method_subdirs: tuple[str, ...],
    common_interval_override: tuple[float, float] | None = None,
) -> UnifiedMeasurement:
    """Measure one method-motion pair once and derive every downstream representation."""
    thresholds.validate()
    row = dict.fromkeys(PER_MOTION_FIELDS, "")
    row.update(method=method_label, motion_class=motion_class, motion=motion)
    quality: dict[str, Any] = {"method": method_subdir, "motion": motion}
    annotations: dict[str, np.ndarray] = {}
    detail: dict[str, Any] = {}
    try:
        from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS

        from loco_mujoco.core.terrain import TerrainSpec
        from musclemimic.utils.retarget.benchmark_timeline import (
            common_interval,
            interpolate_samples,
            sample_durations,
            time_weighted_fraction,
        )
        from terra.source import motion_world_joints
        from terra.terrain.metadata import TerrainMetadata

        root = cache_root.expanduser().resolve()
        timeline = load_timeline(root, method_subdir, motion, source_root=source_root)
        available_timelines = []
        for subdir in all_method_subdirs:
            try:
                available_timelines.append(load_timeline(root, subdir, motion, source_root=source_root))
            except FileNotFoundError:
                continue
        common_start, common_end = common_interval(
            timeline.source_start_s,
            timeline.source_end_s,
            available_timelines,
        )
        if common_interval_override is not None:
            requested_start, requested_end = common_interval_override
            tolerance = 1.0e-9
            if common_start > requested_start + tolerance or common_end < requested_end - tolerance:
                raise ValueError(
                    "candidate timelines do not cover requested common interval "
                    f"{requested_start:.9g}..{requested_end:.9g}s; available interval is "
                    f"{common_start:.9g}..{common_end:.9g}s"
                )
            common_start, common_end = requested_start, requested_end
        common_duration = common_end - common_start
        output_times = timeline.output_times()

        # The expensive terrain/phase measurement occurs exactly once. Every quality view
        # and every headline formula below consumes this same object.
        terrain_measurement = measure(
            motion,
            method=method_subdir,
            terrain_method=terrain_method,
            force_flat=force_flat,
            keep_per_frame=True,
            cache_root=root,
            source_root=source_root,
            terrain_contact_threshold_m=thresholds.terrain_contact_m,
        )
        detail = to_json(terrain_measurement)
        quality.update(terrain_measurement.summary())
        annotations.update({key: np.asarray(value) for key, value in terrain_measurement.per_frame.items()})

        trajectory_path = Path(traj_paths(motion, method=method_subdir, cache_root=root)[0])
        analysis_path = trajectory_path.with_name(trajectory_path.stem + "_analysis.npz")
        with np.load(trajectory_path, allow_pickle=True) as trajectory:
            qpos = np.asarray(trajectory["qpos"], dtype=float)
        raw_qpos = qpos.copy()
        if len(qpos) != terrain_measurement.n_frames:
            raise ValueError(
                f"trajectory has {len(qpos)} frames but terrain measurement has {terrain_measurement.n_frames}"
            )
        metric_model = model()
        if qpos.shape[1] != metric_model.nq:
            raise ValueError(f"trajectory nq={qpos.shape[1]}, metric model nq={metric_model.nq}")

        calibrate_sites = load_site_calibration_state(root, method_subdir, motion)
        source_motion = load_source_motion(motion, analysis_path, source_root=source_root)
        source_joints, source_fps = motion_world_joints(
            motion,
            use_fitted_shape=True,
            motion_data=source_motion,
            calibrate_sites=calibrate_sites,
            fitted_shape_path=source_shape_path("MyoFullBody", root),
        )
        source_probe_sticking = source_probe_contact_on_output(
            source_joints,
            source_fps,
            output_times,
            speed_m_s=thresholds.source_contact_speed_m_s,
        )
        source_sticking = np.column_stack(
            (
                source_probe_sticking[:, 0] | source_probe_sticking[:, 2],
                source_probe_sticking[:, 1] | source_probe_sticking[:, 3],
            )
        )
        skating_probe_positions = robot_point_positions(metric_model, qpos, SKATING_ROBOT_POINTS)
        sole_clearance = np.column_stack(
            [terrain_measurement.per_frame["support_left"], terrain_measurement.per_frame["support_right"]]
        )
        foot_values, foot_annotations = foot_contact_metrics(
            sole_clearance,
            skating_probe_positions,
            source_probe_sticking,
            terrain_measurement.fps,
            thresholds,
            times=output_times,
            common=(common_start, common_end),
            source_support=source_sticking,
            return_annotations=True,
        )
        annotations.update(foot_annotations)
        annotations["source_stance_probes"] = source_probe_sticking

        if force_flat:
            contact_values = floor_contact_preservation(
                sole_clearance,
                source_sticking,
                thresholds,
                output_times,
                (common_start, common_end),
            )
        else:
            source_names = list(SMPLH_DEMO_JOINTS)
            source_contact_points = interpolate_samples(
                source_joints[:, [source_names.index(name) for name in TERRAIN_CONTACT_SOURCE_JOINTS]],
                source_fps,
                output_times,
            )
            metadata_method = terrain_method if terrain_method is not None else method_subdir
            terrain_path = Path(traj_paths(motion, method=metadata_method, cache_root=root)[1])
            metadata = (
                TerrainMetadata.from_terrain(TerrainSpec())
                if not terrain_path.exists()
                else TerrainMetadata.load(terrain_path)
            )
            desired_terrain_contact = (
                signed_distance_to_terrain_boxes(source_contact_points, metadata.terrain)
                <= thresholds.terrain_contact_m
            )
            measured_desired_contact = np.asarray(
                terrain_measurement.per_frame["terrain_contact_desired"],
                dtype=bool,
            )
            if not np.array_equal(desired_terrain_contact, measured_desired_contact):
                raise ValueError("source terrain-contact masks disagree between measurement passes")
            contact_values = terrain_contact_preservation(
                source_contact_points,
                np.asarray(terrain_measurement.per_frame["terrain_contact_distance"], dtype=float),
                metadata.terrain,
                thresholds.terrain_contact_m,
                output_times,
                (common_start, common_end),
            )

        weights = sample_durations(output_times, common_start, common_end)
        active = weights > 0
        transition_active = np.zeros_like(active)
        transition_active[1:] = active[1:] & active[:-1]
        body_penetration = np.asarray(terrain_measurement.per_frame["body_pen"], dtype=float)
        penetration_frame_depth_mm = body_penetration[active & (body_penetration > thresholds.penetration_m)] * 1000.0
        self_penetration = benchmark_self_penetration(terrain_measurement)
        joint_excess = joint_limit_excess_rad(metric_model, qpos)
        joint_linear_excess = joint_limit_excess_m(metric_model, qpos)
        joint_bad = (joint_excess > thresholds.joint_limit) | (joint_linear_excess > thresholds.joint_limit_linear_m)
        relative_tendon_event, physical_tendon_step = tendon_discontinuity_series(metric_model, qpos, active)
        tendon_bad = (relative_tendon_event > thresholds.tendon_jump) & transition_active
        frame_values, frame_annotations = frame_step_metrics(metric_model, qpos, transition_active)
        coupler_values = coupler_metrics(metric_model, qpos, active)
        phase_values, phase_annotations = phase_metrics(terrain_measurement, active)
        pelvis_world_rmse, pelvis_relative_rmse, per_landmark = common_retargeting_rmse(
            metric_model,
            qpos,
            output_times,
            source_joints,
            source_fps,
            active,
        )
        throughput = solver_throughput(analysis_path, common_duration)
        annotations.update(frame_annotations)
        annotations.update(phase_annotations)
        annotations.update(
            {
                "contact_gap": np.asarray(terrain_measurement.per_frame["contact_gap_left"], dtype=bool)
                | np.asarray(terrain_measurement.per_frame["contact_gap_right"], dtype=bool),
                "environment_penetrating": active & (body_penetration > thresholds.penetration_m),
                "joint_limit_bad": joint_bad,
                "joint_limit_excess_rad": joint_excess,
                "joint_limit_excess_m": joint_linear_excess,
                "self_collision_bad": active & (self_penetration > thresholds.self_collision_m),
                "sole_vertical_clearance_m": np.column_stack(
                    [
                        terrain_measurement.per_frame["vert_left"],
                        terrain_measurement.per_frame["vert_right"],
                    ]
                ),
                "source_stance": source_sticking,
                "tendon_jump_bad": tendon_bad,
                "tendon_relative_event": relative_tendon_event,
                "tendon_step_mm": physical_tendon_step,
                "common_active": active,
            }
        )

        row.update(
            {
                "frames": len(qpos),
                "fps": terrain_measurement.fps,
                "common_start_s": common_start,
                "common_end_s": common_end,
                "common_duration_s": common_duration,
                "penetration_duration_pct": 100.0
                * time_weighted_fraction(
                    body_penetration > thresholds.penetration_m,
                    output_times,
                    common_start,
                    common_end,
                ),
                "penetration_max_depth_mm": (
                    float(penetration_frame_depth_mm.mean()) if penetration_frame_depth_mm.size else 0.0
                ),
                "penetration_frame_depth_sum_mm": float(penetration_frame_depth_mm.sum()),
                "penetration_frame_depth_sq_sum_mm2": float(
                    np.dot(penetration_frame_depth_mm, penetration_frame_depth_mm)
                ),
                "penetration_frame_depth_n": int(penetration_frame_depth_mm.size),
                "joint_limit_duration_pct": 100.0
                * time_weighted_fraction(joint_bad, output_times, common_start, common_end),
                "joint_limit_max_excess_deg": float(np.degrees(joint_excess[active].max(initial=0.0))),
                "tendon_jump_duration_pct": 100.0
                * time_weighted_fraction(tendon_bad, output_times, common_start, common_end),
                "tendon_max_jump": float(relative_tendon_event[tendon_bad].max(initial=0.0)),
                "tendon_max_step_mm": float(physical_tendon_step[transition_active].max(initial=0.0)),
                "self_collision_duration_pct": 100.0
                * time_weighted_fraction(
                    self_penetration > thresholds.self_collision_m,
                    output_times,
                    common_start,
                    common_end,
                ),
                "self_collision_max_depth_mm": float(self_penetration[active].max(initial=0.0)) * 1000.0,
                "pelvis_world_rmse_mm": pelvis_world_rmse,
                "pelvis_relative_landmark_rmse_mm": pelvis_relative_rmse,
                "retargeting_per_landmark_rmse_mm_json": json.dumps(per_landmark, sort_keys=True),
            }
        )
        row.update(foot_values)
        row.update(contact_values)
        row.update(phase_values)
        row.update(coupler_values)
        row.update(frame_values)
        row.update(throughput)
        row.update(
            {
                joint_limit_sensitivity_key(tolerance): 100.0
                * time_weighted_fraction(
                    joint_excess > tolerance,
                    output_times,
                    common_start,
                    common_end,
                )
                for tolerance in JOINT_LIMIT_SENSITIVITY_TOLERANCES_RAD
            }
        )
        if not np.array_equal(qpos, raw_qpos, equal_nan=True):
            raise RuntimeError("evaluation modified the raw qpos trajectory")
        quality.update(
            {
                key: row[key]
                for key in (
                    "stance_slip_failure_pct",
                    "swing_clearance_failure_pct",
                    "swing_scrape_failure_pct",
                    "swing_clearance_ratio_median",
                )
            }
        )
        return UnifiedMeasurement(row=row, quality=quality, annotations=annotations, detail=detail)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500]
        row["error"] = error
        quality["error"] = error
        return UnifiedMeasurement(row=row, quality=quality, annotations=annotations, detail=detail)
