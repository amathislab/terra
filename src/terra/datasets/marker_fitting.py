"""Shared, schema-aware marker fitting for external datasets.

Vielemeyer recordings are already C3D marker trajectories.  Darmstadt recordings are
MATLAB structs, so this script first exports a portable marker archive using the same
canonical marker labels accepted by :mod:`musclemimic.web_viewer.c3d_to_smpl`.  Published
touchdowns bound one complete stair traversal before fitting; this avoids fitting the long
wait and opposite-direction traversal that follow it in the original recording.

Every output has a ``.fit.json`` metadata file. A clip is eligible for retargeting only when its
marker error satisfies all three configurable mean/p95/max thresholds.  Runs are resumable;
the dataset commands expose ``--redo`` when refitting outputs is intentional.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# MoSh++/SMPL-H labels paired with the Darmstadt MATLAB field. ``Sac`` is a
# single sacral marker and Darmstadt has no separate left/right PSIS observations. It is
# therefore supplied to both symmetric posterior-waist attachment labels. This preserves
# the observation's central location and weighting but does not invent a pelvis-orientation
# measurement; the remaining trunk markers provide the orientation constraints.
DARMSTADT_MARKERS = (
    ("C7", "C7"),
    ("LSHO", "Sho_l"),
    ("RSHO", "Sho_r"),
    ("LELB", "Elb_lat_l"),
    ("LELBIN", "Elb_med_l"),
    ("RELB", "Elb_lat_r"),
    ("RELBIN", "Elb_med_r"),
    ("LIWR", "Wri_ant_l"),
    ("LOWR", "Wri_pos_l"),
    ("RIWR", "Wri_ant_r"),
    ("ROWR", "Wri_pos_r"),
    ("LFWT", "Sia_l"),
    ("RFWT", "Sia_r"),
    ("LBWT", "Sac"),
    ("RBWT", "Sac"),
    ("LTHI", "Trc_l"),
    ("RTHI", "Trc_r"),
    ("LKNE", "Kne_lat_l"),
    ("LKNI", "Kne_med_l"),
    ("RKNE", "Kne_lat_r"),
    ("RKNI", "Kne_med_r"),
    ("LANK", "Ank_lat_l"),
    ("LHEEI", "Ank_med_l"),
    ("RANK", "Ank_lat_r"),
    ("RHEEI", "Ank_med_r"),
    ("LTOE", "Mt1_l"),
    ("LMT5", "Mt5_l"),
    ("RTOE", "Mt1_r"),
    ("RMT5", "Mt5_r"),
)

# Full SMPL-H pose-vector components for knee motion outside the physiological hinge
# axis: left knee y/z and right knee y/z.  These are the exported equivalents of
# ``_KNEE_NON_HINGE_POSE_AA`` in the fitter and form an explicit conversion-quality gate.
KNEE_NON_HINGE_POSE_INDICES = (13, 14, 16, 17)


@dataclass
class Clip:
    dataset: str
    subject: str
    condition: str
    role: str
    motion: str
    source_path: str
    marker_path: str
    output_path: str
    expected_family: str
    terrain_class: str = ""
    biomechanics_path: str = ""
    expected_slope_deg: float | None = None
    expected_riser_m: float | None = None
    calibration_motion: str = ""
    frames: int = 0
    fps: float = 0.0
    marker_error_mean_mm: float | None = None
    marker_error_p95_mm: float | None = None
    marker_error_max_mm: float | None = None
    knee_nonhinge_max_deg: float | None = None
    fit_passed: bool = False
    status: str = "pending"
    elapsed_seconds: float = 0.0
    error: str = ""


def _interpolate(values: np.ndarray) -> np.ndarray:
    """Linearly fill missing marker coordinates without changing valid observations."""

    values = np.asarray(values, dtype=np.float32).copy()
    for marker in range(values.shape[1]):
        for coordinate in range(3):
            series = values[:, marker, coordinate]
            valid = np.isfinite(series)
            if not valid.any():
                raise ValueError(f"marker {marker} coordinate {coordinate} has no finite samples")
            if not valid.all():
                frames = np.arange(len(series))
                series[~valid] = np.interp(frames[~valid], frames[valid], series[valid])
    return values


def _touchdown_crop(annotation, n_frames: int, fps: float, padding_s: float) -> tuple[int, int]:
    values = np.concatenate(
        [
            np.atleast_1d(np.asarray(annotation.tdL, dtype=float)),
            np.atleast_1d(np.asarray(annotation.tdR, dtype=float)),
        ]
    )
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("touchdown annotation contains no finite frames")
    values -= 1.0  # MATLAB's one-based indices -> Python's zero-based indices.
    padding = round(float(padding_s) * fps)
    start = max(0, int(np.floor(values.min())) - padding)
    end = min(n_frames, int(np.ceil(values.max())) + padding + 1)
    if end <= start:
        raise ValueError(f"invalid touchdown crop ({start}, {end})")
    return start, end


def _validate_smplh(
    path: Path,
    *,
    enforce_knee_hinge: bool = False,
) -> tuple[int, float]:
    with np.load(path, allow_pickle=False) as data:
        required = {"poses", "trans", "betas", "gender", "mocap_framerate"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        poses = np.asarray(data["poses"])
        trans = np.asarray(data["trans"])
        betas = np.asarray(data["betas"])
        fps = float(np.asarray(data["mocap_framerate"]).reshape(()))
    if poses.ndim != 2 or poses.shape[1] != 156 or trans.shape != (len(poses), 3):
        raise ValueError(f"invalid SMPL-H shapes poses={poses.shape}, trans={trans.shape}")
    if len(poses) < 3 or not np.isfinite(poses).all() or not np.isfinite(trans).all() or not np.isfinite(betas).all():
        raise ValueError("SMPL-H trajectory is short or contains non-finite values")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid SMPL-H frame rate {fps}")
    knee_nonhinge = np.asarray(poses[:, KNEE_NON_HINGE_POSE_INDICES], dtype=float)
    maximum = float(np.degrees(np.max(np.abs(knee_nonhinge))))
    if enforce_knee_hinge and maximum > 1e-4:
        raise ValueError(f"SMPL-H knee-hinge gate failed: non-flexion component reaches {maximum:.6f} deg")
    return len(poses), fps


def _knee_nonhinge_max_deg(path: Path) -> float:
    """Return the largest exported non-hinge knee component in degrees."""

    with np.load(path, allow_pickle=False) as data:
        poses = np.asarray(data["poses"], dtype=float)
    return float(np.degrees(np.max(np.abs(poses[:, KNEE_NON_HINGE_POSE_INDICES]))))


def _quality_passed(error: dict, args: argparse.Namespace) -> bool:
    values = (error.get("mean_mm"), error.get("p95_mm"), error.get("max_mm"))
    return all(value is not None and np.isfinite(value) for value in values) and (
        float(values[0]) <= args.max_mean_mm
        and float(values[1]) <= args.max_p95_mm
        and float(values[2]) <= args.max_error_mm
    )


def _quality_gate_ratio(error: dict, args: argparse.Namespace) -> float:
    """Return the worst normalized marker-fit residual, or infinity if incomplete."""
    values = (error.get("mean_mm"), error.get("p95_mm"), error.get("max_mm"))
    limits = (args.max_mean_mm, args.max_p95_mm, args.max_error_mm)
    if not all(value is not None and np.isfinite(value) for value in values):
        return float("inf")
    return max(float(value) / float(limit) for value, limit in zip(values, limits, strict=True))


def _should_retry_fit(error: dict, args: argparse.Namespace, *, retry_failed_fit: bool) -> bool:
    """Return whether Stage II should receive the longer convergence retry."""

    return not _quality_passed(error, args) and (retry_failed_fit or _quality_gate_ratio(error, args) <= 1.10)


def _write_manifest(path: Path, clips: list[Clip]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(clips[0])) if clips else [field.name for field in Clip.__dataclass_fields__.values()]
    temporary = path.with_name(f".{path.name}.tmp")
    rows = []
    for clip in clips:
        row = asdict(clip)
        for field_name in ("marker_path", "output_path", "biomechanics_path"):
            if not row[field_name]:
                continue
            value = Path(row[field_name]).resolve()
            try:
                row[field_name] = value.relative_to(path.parent.resolve()).as_posix()
            except ValueError:
                row[field_name] = str(value)
        rows.append(row)
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _stage1_state_path(clip: Clip, args: argparse.Namespace) -> Path:
    """Return a subject state path compatible with the clip's marker layout.

    Most external recordings use one marker layout per subject and therefore share the
    usual subject-level Stage-I calibration. A small number of Darmstadt trials omit an
    entire physical marker. Those archives cannot reuse a state whose latent marker array
    was fitted with the full 29-label layout, so isolate each exceptional layout while
    retaining subject-level reuse for every clip with the same observations.
    """

    state_root = getattr(args, "stage1_state_root", None)
    if state_root is None:
        state_root = Path(args.output_root) / ".stage1"
    state_name = clip.subject
    if clip.dataset == "darmstadt" and Path(clip.marker_path).suffix.casefold() == ".npz":
        with np.load(clip.marker_path, allow_pickle=False) as marker_data:
            labels = tuple(str(label) for label in np.asarray(marker_data["labels"]).reshape(-1))
        canonical = tuple(label for label, _field in DARMSTADT_MARKERS)
        if labels != canonical:
            digest = hashlib.sha256("\0".join(labels).encode()).hexdigest()[:12]
            state_name = f"{clip.subject}-markers-{len(labels)}-{digest}"
    return Path(state_root) / clip.dataset / f"{state_name}.npz"


def _fit_clip(
    clip: Clip,
    args: argparse.Namespace,
    *,
    force_refit: bool = False,
    retry_failed_fit: bool = False,
) -> Clip:
    from musclemimic.web_viewer.c3d_to_smpl import fit_smpl_to_c3d, save_motion_data_as_amass_smplh_npz

    output_path = Path(clip.output_path)
    metadata_path = output_path.with_suffix(".fit.json")
    state_path = _stage1_state_path(clip, args)

    reuse_existing = output_path.exists() and metadata_path.exists() and not args.redo and not force_refit
    metadata = json.loads(metadata_path.read_text()) if reuse_existing else {}
    if reuse_existing and Path(metadata.get("stage1_state", "")).resolve() != state_path.resolve():
        reuse_existing = False
    if reuse_existing and retry_failed_fit and not _quality_passed(metadata.get("marker_error", {}), args):
        reuse_existing = False
    # A file fitted before the hinge constraint existed is not compatible with a run that
    # requests it, even if its marker-residual metadata happens to pass. Refit it instead
    # of silently blessing stale SMPL-H rotations.
    if reuse_existing and bool(metadata.get("enforce_knee_hinge", False)) != bool(args.enforce_knee_hinge):
        reuse_existing = False
    vielemeyer_foot_marker_weight = float(getattr(args, "vielemeyer_foot_marker_weight", 1.0))
    raw_heel_weight = getattr(args, "vielemeyer_heel_marker_weight", None)
    vielemeyer_heel_marker_weight = vielemeyer_foot_marker_weight if raw_heel_weight is None else float(raw_heel_weight)
    if not np.isfinite(vielemeyer_foot_marker_weight) or vielemeyer_foot_marker_weight <= 0:
        raise ValueError("Vielemeyer foot-marker weight must be finite and positive")
    if not np.isfinite(vielemeyer_heel_marker_weight) or vielemeyer_heel_marker_weight <= 0:
        raise ValueError("Vielemeyer heel-marker weight must be finite and positive")
    if reuse_existing and float(metadata.get("vielemeyer_foot_marker_weight", 1.0)) != (
        vielemeyer_foot_marker_weight if clip.dataset == "vielemeyer" else 1.0
    ):
        reuse_existing = False
    if reuse_existing and float(
        metadata.get(
            "vielemeyer_heel_marker_weight",
            metadata.get("vielemeyer_foot_marker_weight", 1.0),
        )
    ) != (vielemeyer_heel_marker_weight if clip.dataset == "vielemeyer" else 1.0):
        reuse_existing = False
    if reuse_existing:
        frames, fps = _validate_smplh(
            output_path,
            enforce_knee_hinge=args.enforce_knee_hinge,
        )
        error = metadata.get("marker_error", {})
        clip.frames = frames
        clip.fps = fps
        clip.marker_error_mean_mm = error.get("mean_mm")
        clip.marker_error_p95_mm = error.get("p95_mm")
        clip.marker_error_max_mm = error.get("max_mm")
        clip.knee_nonhinge_max_deg = _knee_nonhinge_max_deg(output_path)
        clip.fit_passed = _quality_passed(error, args)
        clip.status = "existing"
        clip.elapsed_seconds = float(metadata.get("elapsed_seconds", 0.0))
        return clip

    if args.redo_stage1 and state_path.exists():
        state_path.unlink()
    started = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:

        def fit(stage2_iters: int):
            foot_weights = None
            if clip.dataset == "vielemeyer" and (
                vielemeyer_foot_marker_weight != 1.0 or vielemeyer_heel_marker_weight != 1.0
            ):
                foot_weights = dict.fromkeys(
                    ("LANK", "LHEE", "LTOE", "RANK", "RHEE", "RTOE"), vielemeyer_foot_marker_weight
                )
                foot_weights.update(dict.fromkeys(("LHEE", "RHEE"), vielemeyer_heel_marker_weight))
            return fit_smpl_to_c3d(
                clip.marker_path,
                args.smpl_model_path,
                surface_model_type="smplh",
                gender="neutral",
                target_fps=args.target_fps,
                stage1_iters=args.stage1_iters,
                stage2_iters=stage2_iters,
                n_ref_frames=args.n_ref_frames,
                stage1_shape_solver=args.stage1_shape_solver,
                device=args.device,
                stage2_solver="batched_lbfgs",
                enforce_knee_hinge=args.enforce_knee_hinge,
                stage2_marker_weight_overrides=foot_weights,
                strict_frame_picking=False,
                least_avail_markers=args.least_avail_markers,
                optimize_toes=True,
                stage1_state_path=str(state_path),
            )

        actual_stage2_iters = int(args.stage2_iters)
        motion_data = fit(actual_stage2_iters)
        initial_error = motion_data.get("debug", {}).get("marker_error", {})
        retry_error = None
        # A narrowly missed residual gate can be optimizer convergence, not bad mocap.
        # Ordinary clips retry only near misses. Dataset calibration selection may request
        # the longer solve for any miss before rejecting a subject-wide Stage-I state.
        if _should_retry_fit(initial_error, args, retry_failed_fit=retry_failed_fit):
            retry_iters = max(160, 2 * actual_stage2_iters)
            retry_data = fit(retry_iters)
            retry_error = retry_data.get("debug", {}).get("marker_error", {})
            if _quality_gate_ratio(retry_error, args) < _quality_gate_ratio(initial_error, args):
                motion_data = retry_data
                actual_stage2_iters = retry_iters
        save_motion_data_as_amass_smplh_npz(motion_data, output_path)
        clip.frames, clip.fps = _validate_smplh(
            output_path,
            enforce_knee_hinge=args.enforce_knee_hinge,
        )
        clip.knee_nonhinge_max_deg = _knee_nonhinge_max_deg(output_path)
        error = motion_data.get("debug", {}).get("marker_error", {})
        clip.marker_error_mean_mm = error.get("mean_mm")
        clip.marker_error_p95_mm = error.get("p95_mm")
        clip.marker_error_max_mm = error.get("max_mm")
        clip.fit_passed = _quality_passed(error, args)
        clip.status = "generated"
        clip.elapsed_seconds = time.time() - started
        metadata = {
            "motion": clip.motion,
            "dataset": clip.dataset,
            "subject": clip.subject,
            "condition": clip.condition,
            "role": clip.role,
            "source_path": str(Path(clip.source_path).resolve()),
            "marker_path": str(Path(clip.marker_path).resolve()),
            "stage1_state": str(state_path.resolve()),
            "stage1_reused": bool(motion_data.get("debug", {}).get("stagei", {}).get("reused")),
            "smpl_model_path": str(Path(args.smpl_model_path).resolve()),
            "surface_model_type": "smplh",
            "gender": "neutral",
            "target_fps": args.target_fps,
            "stage1_iters": args.stage1_iters,
            "stage2_iters": actual_stage2_iters,
            "initial_stage2_iters": args.stage2_iters,
            "initial_marker_error": initial_error,
            "retry_marker_error": retry_error,
            "n_ref_frames": args.n_ref_frames,
            "stage1_shape_solver": args.stage1_shape_solver,
            "stage2_solver": "batched_lbfgs",
            "enforce_knee_hinge": args.enforce_knee_hinge,
            "vielemeyer_foot_marker_weight": (vielemeyer_foot_marker_weight if clip.dataset == "vielemeyer" else 1.0),
            "vielemeyer_heel_marker_weight": (vielemeyer_heel_marker_weight if clip.dataset == "vielemeyer" else 1.0),
            "knee_nonhinge_max_deg": clip.knee_nonhinge_max_deg,
            "device": args.device,
            "marker_error": error,
            "quality_thresholds_mm": {
                "mean": args.max_mean_mm,
                "p95": args.max_p95_mm,
                "max": args.max_error_mm,
            },
            "fit_passed": clip.fit_passed,
            "elapsed_seconds": clip.elapsed_seconds,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    except Exception as exc:
        clip.status = "failed"
        clip.elapsed_seconds = time.time() - started
        clip.error = f"{type(exc).__name__}: {exc}"
    return clip
