"""Config-driven SMPL-H to TERRA dataset execution.

Raw dataset parsing belongs in dataset-specific converters.  This module begins at the
common boundary: an AMASS-compatible SMPL-H archive plus a manifest row.  Dataset
differences are declarative (calibration policy, contact adapter, terrain options, and
retargeting overrides), allowing one runner to process every dataset without branching
on dataset names.
"""

from __future__ import annotations

import csv
import functools
import json
import logging
import os
import platform
import re
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from terra.api import retarget
from terra.artifacts import (
    load_retarget_analysis,
    retarget_cache_paths,
    save_retarget_result,
    validate_retarget_artifacts,
)
from terra.paths import StorageRoots
from terra.reconstruction import (
    CalibrationEvidence,
    ReconstructionRequest,
    ValidationPolicy,
    add_posed_seat_support,
    reconstruct_terrain,
    smplh_terrain_fit_options,
)
from terra.runtime import (
    ensure_environment_registered,
    resolve_model_path,
    shape_cache_path,
)
from terra.runtime import (
    ensure_robot_shape as ensure_runtime_robot_shape,
)
from terra.smplh import load_smplh_motion
from terra.terrain.metadata import TerrainMetadata
from terra.terrain.stance import DEFAULT_CONTACT_JOINTS

CONFIG_SCHEMA_VERSION = 1
CONFIG_TABLES = {"schema_version", "dataset", "input", "terrain", "retarget"}
CALIBRATION_MODES = {"self", "manifest", "none"}
TERRAIN_MODES = {"fit", "flat", "precomputed"}
_FOOT_CALIBRATION_CACHE: dict[tuple[Path, Path, Path, str, bool], tuple[dict[str, float], dict[str, float]]] = {}


def benchmark_timing_context() -> dict[str, object]:
    """Describe the compute allocation that produced solver wall-clock timing."""

    cpu_model = platform.processor().strip()
    if not cpu_model:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.casefold().startswith("model name"):
                    cpu_model = line.partition(":")[2].strip()
                    break
        except OSError:
            pass

    def positive_integer(name: str, fallback: int) -> int:
        raw = os.environ.get(name, str(fallback))
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
        if value < 1:
            raise ValueError(f"{name} must be a positive integer, got {raw!r}")
        return value

    visible_cpus = os.cpu_count() or 1
    return {
        "cpu_model": cpu_model,
        "cpu_request": positive_integer("TERRA_RUNAI_CPU_REQUEST", visible_cpus),
        "worker_processes": positive_integer("TERRA_RUN_WORKERS", 1),
        "threads_per_worker": positive_integer("TERRA_RUNAI_THREADS_PER_WORKER", 1),
        "omp_num_threads": positive_integer("OMP_NUM_THREADS", 1),
        "mkl_num_threads": positive_integer("MKL_NUM_THREADS", 1),
        "openblas_num_threads": positive_integer("OPENBLAS_NUM_THREADS", 1),
        "xla_flags": os.environ.get("XLA_FLAGS", "").strip(),
    }


def _repo_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    # Installed package data normally has no project marker above it. External
    # storage roots still make the data/runs/smpl aliases unambiguous; cwd is the
    # least surprising base for any remaining custom relative path.
    return Path.cwd().resolve()


def _table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return value


@dataclass(frozen=True)
class DatasetConfig:
    """Resolved paths and processing policy for one motion collection.

    Load this from a versioned TOML file with :func:`load_dataset_config`.
    ``input_root`` contains source SMPL-H archives; ``cache_root`` holds
    published retargeting artifacts, while ``run_root`` holds runner reports.
    Terrain and calibration fields describe how those archives become a
    selected retargeting cohort.
    """

    path: Path
    repo_root: Path
    storage_roots: StorageRoots
    name: str
    description: str
    input_root: Path
    manifest_path: Path | None
    input_glob: str
    calibration_mode: str
    calibration_template: str | None
    contact_joints: tuple[str, ...]
    calibrate_sites: bool
    terrain_mode: str
    terrain_fit: dict[str, Any]
    terrain_source_dir: Path | None
    terrain_source_method: str | None
    method: str
    env_name: str
    smpl_model_path: Path
    cache_root: Path
    reference_cache_root: Path
    run_root: Path
    retarget_overrides: dict[str, Any]
    method_overrides: dict[str, Any]
    posed_seat_frame: str | None = None


@dataclass(frozen=True)
class MotionRecord:
    """One source motion selected from a converter manifest or AMASS tree.

    ``motion`` is the portable ID relative to ``input_root``, without
    ``.npz``. ``source_path`` names the actual SMPL-H archive. A failed marker
    fit remains represented by ``fit_passed=False`` so cohort accounting does
    not silently exclude it.
    """

    motion: str
    dataset: str
    source_path: Path
    subject: str = ""
    condition: str = ""
    terrain_class: str = ""
    expected_family: str = ""
    calibration_path: Path | None = None
    fit_passed: bool = True
    metadata: dict[str, str] = field(default_factory=dict)


def load_dataset_config(
    path: str | Path,
    *,
    storage_roots: StorageRoots | None = None,
    environment: Mapping[str, str] | None = None,
) -> DatasetConfig:
    """Resolve one dataset's TOML policy into absolute input and output paths.

    With no explicit ``storage_roots``, ``TERRA_DATA_ROOT``,
    ``TERRA_ARTIFACT_ROOT``, and ``TERRA_MODEL_ROOT`` provide the aliases in
    bundled configs. ``environment`` can supply those variables without
    mutating the process environment. The config selects its source manifest
    or archive glob, terrain mode, calibration, model root, and output caches.
    This validates the policy and path syntax; it does not download data or
    assert that every selected source archive already exists.
    """

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    if data.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"{config_path} must declare schema_version = {CONFIG_SCHEMA_VERSION}")
    unknown_tables = sorted(set(data) - CONFIG_TABLES)
    if unknown_tables:
        raise ValueError(
            "dataset configs contain processing policy only; unsupported top-level "
            f"table(s): {', '.join(unknown_tables)}"
        )

    root = storage_roots.repository_root if storage_roots is not None else _repo_root(config_path)
    roots = storage_roots or StorageRoots.from_environment(root, environment=environment)
    dataset = _table(data, "dataset")
    inputs = _table(data, "input")
    terrain = _table(data, "terrain")
    retarget = _table(data, "retarget")
    name = str(dataset.get("name", "")).strip().casefold()
    if not name:
        raise ValueError("[dataset].name is required")

    input_root = roots.resolve_input(inputs.get("root"), environment=environment)
    model_path = roots.resolve_model(retarget.get("smpl_model_path"), environment=environment)
    cache_root = roots.resolve_artifact(retarget.get("cache_root"), environment=environment)
    reference_cache_root = roots.resolve_artifact(
        retarget.get("reference_cache_root", retarget.get("cache_root")),
        environment=environment,
    )
    run_root = roots.resolve_artifact(retarget.get("run_root"), environment=environment)
    if (
        input_root is None
        or model_path is None
        or cache_root is None
        or reference_cache_root is None
        or run_root is None
    ):
        raise ValueError("[input].root and [retarget] smpl_model_path/cache_root/run_root are required")

    calibration_mode = str(terrain.get("calibration", "self")).casefold()
    contact_source = str(terrain.get("contact_source", "kinematic")).casefold()
    terrain_mode = str(terrain.get("mode", "fit")).casefold()
    if calibration_mode not in CALIBRATION_MODES:
        raise ValueError(f"[terrain].calibration must be one of {sorted(CALIBRATION_MODES)}")
    if contact_source != "kinematic":
        raise ValueError("[terrain].contact_source must be 'kinematic'")
    if terrain_mode not in TERRAIN_MODES:
        raise ValueError(f"[terrain].mode must be one of {sorted(TERRAIN_MODES)}")
    posed_seat_frame = str(terrain.get("posed_seat_frame", "apparatus")).casefold()
    if posed_seat_frame not in {"normalized", "apparatus"}:
        raise ValueError("[terrain].posed_seat_frame must be 'normalized' or 'apparatus'")
    terrain_source_dir = roots.resolve_input(terrain.get("source_dir"), environment=environment)
    raw_source_method = terrain.get("source_method")
    terrain_source_method = None if raw_source_method is None else str(raw_source_method).strip().casefold()
    if terrain_mode == "precomputed":
        if terrain_source_dir is None:
            raise ValueError("[terrain].source_dir is required when mode='precomputed'")
        if not terrain_source_method:
            raise ValueError("[terrain].source_method is required when mode='precomputed'")
    elif terrain_source_dir is not None or terrain_source_method is not None:
        raise ValueError("[terrain].source_dir/source_method require mode='precomputed'")
    calibration_template = terrain.get("calibration_template")
    if calibration_mode == "manifest" and not calibration_template and not inputs.get("manifest"):
        raise ValueError("manifest calibration requires [input].manifest or a calibration_template")

    contact_joints = tuple(str(value) for value in terrain.get("contact_joints", DEFAULT_CONTACT_JOINTS))
    if contact_joints != DEFAULT_CONTACT_JOINTS:
        raise ValueError(f"[terrain].contact_joints must be {list(DEFAULT_CONTACT_JOINTS)!r}")

    return DatasetConfig(
        path=config_path,
        repo_root=root,
        storage_roots=roots,
        name=name,
        description=str(dataset.get("description", "")).strip(),
        input_root=input_root,
        manifest_path=roots.resolve_input(inputs.get("manifest"), environment=environment),
        input_glob=str(inputs.get("glob", "**/*.npz")),
        calibration_mode=calibration_mode,
        calibration_template=(str(calibration_template) if calibration_template is not None else None),
        contact_joints=contact_joints,
        calibrate_sites=bool(terrain.get("calibrate_sites", True)),
        terrain_mode=terrain_mode,
        posed_seat_frame=posed_seat_frame,
        terrain_fit=dict(_table(terrain, "fit")),
        terrain_source_dir=terrain_source_dir,
        terrain_source_method=terrain_source_method,
        method=str(retarget.get("method", "terra")),
        env_name=str(retarget.get("env_name", "MyoFullBody")),
        smpl_model_path=model_path,
        cache_root=cache_root,
        reference_cache_root=reference_cache_root,
        run_root=run_root,
        retarget_overrides=dict(_table(retarget, "overrides")),
        method_overrides=dict(_table(retarget, "method_overrides")),
    )


def _bool(value: object, *, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    return str(value).strip().casefold() in {"1", "true", "yes", "pass", "passed"}


def _record_path(
    value: str | None,
    root: Path,
) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    return (root / path).resolve()


def _calibration_path(row: dict[str, str], config: DatasetConfig, source_path: Path) -> Path | None:
    if config.calibration_mode == "none":
        return None
    if config.calibration_mode == "self":
        return source_path
    explicit = _record_path(
        row.get("calibration_path"),
        config.input_root,
    )
    if explicit is not None:
        return explicit
    motion = row.get("calibration_motion", "").strip()
    if not motion and config.calibration_template:
        context = dict(row)
        subject_match = re.search(r"(\d+)", row.get("subject", ""))
        if subject_match is not None:
            subject_number = int(subject_match.group(1))
            context.update(
                subject_number=subject_number,
                subject_padded=f"{subject_number:03d}",
            )
        try:
            motion = config.calibration_template.format_map(context)
        except KeyError as exc:
            raise ValueError(f"calibration template for {config.name} requires missing manifest field {exc}") from exc
    if not motion:
        raise ValueError(f"motion {row.get('motion', source_path.stem)} has no calibration motion")
    return (config.input_root / f"{motion.removesuffix('.npz')}.npz").resolve()


def _row_source(row: dict[str, str], config: DatasetConfig) -> Path:
    for field_name in ("smplh_path", "output_path"):
        path = _record_path(
            row.get(field_name),
            config.input_root,
        )
        if path is not None:
            return path
    motion = row.get("motion", "").strip()
    if not motion:
        raise ValueError("dataset manifest row has neither a motion nor an output path")
    return (config.input_root / f"{motion.removesuffix('.npz')}.npz").resolve()


def load_motion_records(
    config: DatasetConfig,
    *,
    include_non_retarget: bool = False,
    selected_motions: Sequence[str] | None = None,
) -> list[MotionRecord]:
    """Read converted-motion rows or discover AMASS-compatible archives.

    When the config names a converter manifest, the default includes only
    retargeting rows. ``include_non_retarget=True`` also retains calibration
    and other roles without changing their source files or provenance. Failed
    fits remain as records with ``fit_passed=False``. For a manifest-free AMASS
    config, ``selected_motions`` limits discovery to IDs relative to
    ``input_root`` without the ``.npz`` suffix. The caller applies any later
    selection order or subset to manifest-backed records.
    """

    records: list[MotionRecord] = []
    if config.manifest_path is not None:
        if not config.manifest_path.is_file():
            raise FileNotFoundError(
                f"converted manifest not found: {config.manifest_path}; run the dataset's "
                "marker-to-SMPL-H converter first"
            )
        with config.manifest_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if not include_non_retarget and row.get("role", "retarget").strip().casefold() not in {"", "retarget"}:
                continue
            source_path = _row_source(row, config)
            motion = row.get("motion", "").strip()
            if not motion:
                motion = source_path.relative_to(config.input_root).with_suffix("").as_posix()
            status = row.get("status", "").strip().casefold()
            passed_default = status not in {"failed", "missing", "error"}
            passed = _bool(row.get("fit_passed"), default=passed_default)
            records.append(
                MotionRecord(
                    motion=motion,
                    dataset=config.name,
                    source_path=source_path,
                    subject=row.get("subject", ""),
                    condition=row.get("condition", row.get("movement", "")),
                    terrain_class=row.get("terrain_class", ""),
                    expected_family=row.get("expected_family", ""),
                    calibration_path=_calibration_path(row, config, source_path),
                    fit_passed=passed,
                    metadata=dict(row),
                )
            )
    else:
        source_paths = (
            [(config.input_root / f"{motion.removesuffix('.npz')}.npz").resolve() for motion in selected_motions]
            if selected_motions is not None
            else sorted(config.input_root.glob(config.input_glob))
        )
        for source_path in source_paths:
            if (selected_motions is None and not source_path.is_file()) or source_path.name.endswith("_analysis.npz"):
                continue
            motion = source_path.relative_to(config.input_root).with_suffix("").as_posix()
            row = {"motion": motion}
            records.append(
                MotionRecord(
                    motion=motion,
                    dataset=config.name,
                    source_path=source_path.resolve(),
                    calibration_path=_calibration_path(row, config, source_path.resolve()),
                )
            )
    motions = [record.motion for record in records]
    if len(motions) != len(set(motions)):
        duplicate = next(motion for motion in motions if motions.count(motion) > 1)
        raise ValueError(f"dataset manifest contains duplicate motion {duplicate!r}")
    return records


def ensure_robot_shape(config: DatasetConfig, logger: logging.Logger | None = None) -> Path:
    """Fit or reuse the MyoFullBody shape shared by this dataset's motions.

    The neutral SMPL-H model must exist at ``config.smpl_model_path``. The
    fitted shape is written under ``config.cache_root``; precomputed-terrain
    configs use ``config.reference_cache_root`` so reconstruction and
    retargeting refer to the same calibration. Call this once before parallel
    cohort terrain fitting. Returns the absolute ``shape_optimized.pkl`` path.
    """

    from terra._musclemimic import load_robot_conf_file

    ensure_environment_registered(config.env_name)
    model_path = resolve_model_path(config.smpl_model_path)
    shape_root = config.reference_cache_root if config.terrain_mode == "precomputed" else config.cache_root
    shape_path = shape_cache_path(config.env_name, shape_root)
    robot_conf = load_robot_conf_file(config.env_name)
    ensure_runtime_robot_shape(
        config.env_name,
        robot_conf,
        model_path,
        shape_path,
        logger or logging.getLogger("terra.dataset"),
    )
    return shape_path


def _world_joints(
    config: DatasetConfig,
    record: MotionRecord,
    path: Path,
    *,
    return_normalization: bool = False,
):
    from terra.source import motion_world_joints

    if not path.is_file():
        raise FileNotFoundError(path)
    data = load_smplh_motion(path)
    return motion_world_joints(
        record.motion,
        env_name=config.env_name,
        use_fitted_shape=True,
        motion_data=data,
        calibrate_sites=config.calibrate_sites,
        return_normalization=return_normalization,
        smpl_model_path=config.smpl_model_path,
        fitted_shape_path=shape_cache_path(
            config.env_name,
            config.reference_cache_root if config.terrain_mode == "precomputed" else config.cache_root,
        ),
    )


def _record_calibration(
    config: DatasetConfig,
    record: MotionRecord,
    joints: np.ndarray,
    fps: float,
) -> tuple[dict[str, float] | None, dict[str, float]]:
    """Return the production foot-pitch and sole-offset calibration for one record."""

    if record.calibration_path is None:
        return None, {}
    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS
    from terra.terrain import (
        calibrate_neutral_foot_pitch,
        detect_stance_events,
        joint_surface_offsets,
        paired_sole_offsets,
    )

    calibration_key = (
        record.calibration_path,
        config.smpl_model_path,
        shape_cache_path(
            config.env_name,
            config.reference_cache_root if config.terrain_mode == "precomputed" else config.cache_root,
        ),
        config.env_name,
        config.calibrate_sites,
    )
    cached_calibration = _FOOT_CALIBRATION_CACHE.get(calibration_key)
    if cached_calibration is not None:
        neutral, offsets = map(dict, cached_calibration)
        return neutral, offsets
    if record.calibration_path == record.source_path:
        calibration_joints, calibration_fps = joints, fps
    else:
        calibration_joints, calibration_fps = _world_joints(config, record, record.calibration_path)
    neutral = calibrate_neutral_foot_pitch(calibration_joints, list(SMPLH_DEMO_JOINTS), calibration_fps)
    missing_neutral = {"L", "R"} - set(neutral)
    if missing_neutral:
        raise ValueError(f"foot-pitch calibration is missing side(s) {sorted(missing_neutral)}")
    events = detect_stance_events(calibration_joints, list(SMPLH_DEMO_JOINTS), calibration_fps)
    offsets = paired_sole_offsets(joint_surface_offsets(events))
    required_offsets = set(DEFAULT_CONTACT_JOINTS)
    missing_offsets = required_offsets - set(offsets)
    if missing_offsets:
        raise ValueError(f"sole calibration is missing probe(s) {sorted(missing_offsets)}")
    _FOOT_CALIBRATION_CACHE[calibration_key] = (dict(neutral), dict(offsets))
    return neutral, offsets


def fit_record_terrain(
    config: DatasetConfig,
    record: MotionRecord,
    *,
    reconstruction_profile: str | None = None,
    fit_options_override: Mapping[str, Any] | None = None,
) -> tuple[object | None, dict[str, Any], dict[str, Any], dict[str, float]]:
    """Fit and validate support geometry for one normalized SMPL-H record.

    For fit mode, ``ensure_robot_shape`` must have populated the shared
    shape cache first. The function evaluates world-frame joints and applies
    the config's foot
    calibration and terrain options, then returns ``(terrain, fit_report,
    validation, sole_offsets_m)``. A flat config returns ``terrain=None``.
    ``reconstruction_profile=None`` uses the production fitting path; explicit
    profiles are for the reconstruction benchmark and add their settings to
    the fit report.
    """

    from terra.terrain import (
        TERRA_FULL_PROFILE,
        resolve_terra_reconstruction_profile,
    )

    profile = resolve_terra_reconstruction_profile(
        TERRA_FULL_PROFILE if reconstruction_profile is None else reconstruction_profile
    )
    record_profile = reconstruction_profile is not None

    if config.terrain_mode == "flat":
        result = reconstruct_terrain(
            ReconstructionRequest(
                mode="flat",
                joints=None,
                joint_names=(),
                fps=None,
                profile=profile,
                record_profile=record_profile,
                report_fields={"dataset_config": config.name},
            )
        )
        return result.terrain, result.report, result.validation, {}

    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS

    joints, fps, normalization = _world_joints(
        config,
        record,
        record.source_path,
        return_normalization=True,
    )
    neutral, offsets = _record_calibration(config, record, joints, fps)

    fit_options = dict(config.terrain_fit)
    fit_options.update(dict(fit_options_override or {}))
    if neutral is None:
        fit_options = smplh_terrain_fit_options(
            fit_options, use_fitted_shape=True, calibrate_sites=config.calibrate_sites
        )
        neutral = fit_options.get("neutral_foot_pitch")
    else:
        fit_options["neutral_foot_pitch_source"] = "provided_flat_reference"
    seat_frame = getattr(config, "posed_seat_frame", None) or "apparatus"
    ground_correction = (
        0.0 if seat_frame == "normalized" else -float(normalization["source_to_normalized_translation_m"][2])
    )
    if (
        profile.use_posed_seat_surface
        and fit_options.get("seat", "auto") == "auto"
        and "seat_support_heights" not in fit_options
    ):
        fit_options = add_posed_seat_support(
            fit_options,
            joints,
            fps,
            lambda: load_smplh_motion(record.source_path),
            config.smpl_model_path,
            shape_cache_path(config.env_name, config.cache_root),
            use_fitted_shape=True,
            calibrate_sites=config.calibrate_sites,
            ground_datum_correction_m=ground_correction,
        )
    input_metadata = None
    if record_profile:
        input_metadata = {
            "input_stage": "pre_retarget_myofullbody_fitted_site_calibrated",
            "adapter": "terra.source.motion_world_joints",
            "env_name": config.env_name,
            "use_fitted_shape": True,
            "calibrate_sites": config.calibrate_sites,
            "joint_order": list(SMPLH_DEMO_JOINTS),
            "fps": float(fps),
            "joints": {
                "shape": list(joints.shape),
                "dtype": joints.dtype.str,
                "units": "m",
            },
            "normalization": normalization,
        }
    result = reconstruct_terrain(
        ReconstructionRequest(
            mode="fit",
            joints=joints,
            joint_names=tuple(SMPLH_DEMO_JOINTS),
            fps=fps,
            fit_options=fit_options,
            calibration=CalibrationEvidence(
                neutral_foot_pitch=neutral,
                joint_offsets=offsets or None,
                source=getattr(config, "calibration_mode", "manifest" if record.calibration_path else "none"),
                path=None if record.calibration_path is None else str(record.calibration_path),
            ),
            profile=profile,
            record_profile=record_profile,
            report_fields={
                "dataset_config": config.name,
                "calibration_path": (str(record.calibration_path) if record.calibration_path is not None else None),
                "contact_source": "kinematic",
            },
            input_metadata=input_metadata,
            seat_ground_datum_correction_m=ground_correction,
            validation=ValidationPolicy(compensate_sloped_offsets=not bool(offsets)),
        )
    )
    return result.terrain, result.report, result.validation, offsets


def _json_default(value: object) -> object:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _selected_terrain_family(report: dict[str, Any], terrain: object | None) -> str:
    """Derive the reconstructed family from fitted output, never manifest metadata."""

    if report.get("model") == "ramp":
        return "ramp"
    if terrain is not None and getattr(terrain, "boxes", ()):
        return "steps"
    return "flat"


@functools.cache
def _precomputed_run_identity(source_dir: Path, expected_method: str) -> dict[str, str]:
    """Validate one reconstruction run and return its stable scientific identity."""

    from terra.benchmarking.reconstruction.provenance import validate_run_provenance

    run_path = source_dir / "run.json"
    try:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid reconstruction run JSON: {run_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("method") != expected_method:
        raise ValueError(f"reconstruction run method does not match {expected_method!r}: {run_path}")
    provenance = validate_run_provenance(payload, run_path)
    return {
        "method": provenance.method,
        "method_identity_sha256": provenance.method_identity_sha256,
        "scientific_identity_sha256": provenance.scientific_identity_sha256,
    }


def _precomputed_terrain(
    config: DatasetConfig,
    record: MotionRecord,
    *,
    calibrate: bool = True,
) -> tuple[object, dict[str, Any], dict[str, Any], dict[str, float], dict[str, str]]:
    """Load one provenance-bound reconstruction record for retargeting."""

    from terra.benchmarking.reconstruction.provenance import RECORD_PROVENANCE_SCHEMA, file_sha256

    source_dir = config.terrain_source_dir
    expected_method = config.terrain_source_method
    if source_dir is None or expected_method is None:
        raise ValueError("precomputed terrain mode requires a source directory and method")
    run_identity = _precomputed_run_identity(source_dir, expected_method)
    record_path = source_dir / f"{record.motion.replace('/', '__')}.json"
    if record_path.is_symlink():
        raise ValueError(f"precomputed terrain records must not be symlinks: {record_path}")
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid reconstruction record JSON: {record_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"reconstruction record must contain an object: {record_path}")
    if payload.get("method") != expected_method or payload.get("motion") != record.motion:
        raise ValueError(f"reconstruction record method/motion mismatch: {record_path}")
    provenance = payload.get("provenance")
    expected_provenance = {"schema": RECORD_PROVENANCE_SCHEMA} | {
        key: run_identity[key] for key in ("method_identity_sha256", "scientific_identity_sha256")
    }
    if provenance != expected_provenance:
        raise ValueError(f"reconstruction record provenance does not match its run: {record_path}")
    terrain_value = payload.get("terrain")
    if not isinstance(terrain_value, Mapping):
        raise ValueError(f"reconstruction record has no terrain object: {record_path}")
    terrain = TerrainMetadata.from_dict(terrain_value).terrain
    identity = run_identity | {"record_sha256": file_sha256(record_path)}
    offsets: dict[str, float] = {}
    if calibrate and record.calibration_path is not None:
        joints, fps = _world_joints(config, record, record.source_path)
        _neutral, offsets = _record_calibration(config, record, joints, fps)
    report = {
        "model": expected_method,
        "dataset_config": config.name,
        "terrain_source": "precomputed_reconstruction_record",
        "terrain_source_path": str(record_path),
        "terrain_reconstruction_source": identity,
    }
    source_validation = payload.get("validation")
    validation = {
        "passed": None if not isinstance(source_validation, Mapping) else source_validation.get("passed"),
        "source": "precomputed_reconstruction_record",
    }
    return terrain, report, validation, offsets, identity


def run_motion(
    config: DatasetConfig,
    record: MotionRecord,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Process one selected archive and report its published artifacts.

    TERRA fits terrain from the motion or consumes a declared precomputed
    reconstruction; comparison methods use the authoritative TERRA terrain.
    A successful run publishes the trajectory and analysis under
    ``config.cache_root`` and a detailed terrain report under
    ``config.run_root/terrain``. The returned status mapping includes motion
    identity, ``ok`` or ``cached``, artifact paths, frames, and timing.
    Failed source conversion is reported as ``conversion_failed`` without
    inventing a trajectory. Invalid inputs or processing errors may raise.
    """

    started = time.time()
    terrain_class = record.terrain_class or ("flat" if config.terrain_mode == "flat" else "")
    expected_family = record.expected_family or ("flat" if config.terrain_mode == "flat" else "")
    base: dict[str, Any] = {
        "motion": record.motion,
        "dataset": config.name,
        "subject": record.subject,
        "condition": record.condition,
        "benchmark_group": record.metadata.get("benchmark_group", ""),
        "terrain_class": terrain_class,
        "expected_family": expected_family,
        "fit_passed": record.fit_passed,
        "marker_fit_mean_mm": record.metadata.get("marker_error_mean_mm", ""),
        "marker_fit_p95_mm": record.metadata.get("marker_error_p95_mm", ""),
        "marker_fit_max_mm": record.metadata.get("marker_error_max_mm", ""),
        "status": "pending",
        "error": "",
    }
    if not record.fit_passed:
        base.update(status="conversion_failed", elapsed_seconds=0.0)
        return base
    if not record.source_path.is_file():
        base.update(
            status="conversion_failed",
            error=f"SMPL-H archive not found: {record.source_path}",
            elapsed_seconds=0.0,
        )
        return base

    paths = retarget_cache_paths(
        config.cache_root,
        record.motion,
        method=config.method,
        env_name=config.env_name,
    )
    if not overwrite and paths.trajectory_path.is_file() and paths.analysis_path.is_file():
        validated = validate_retarget_artifacts(
            config.cache_root,
            record.motion,
            method=config.method,
            env_name=config.env_name,
        )
        if config.terrain_mode == "precomputed":
            _terrain, _report, _validation, _offsets, expected_identity = _precomputed_terrain(
                config,
                record,
                calibrate=False,
            )
            analysis = load_retarget_analysis(validated.analysis_path)
            if analysis.get("terrain_reconstruction_source") != expected_identity:
                raise ValueError(
                    "cached retargeting artifact was produced from a different reconstructed terrain; "
                    f"use a fresh cache root or pass --overwrite: {validated.analysis_path}"
                )
        base.update(
            status="cached",
            trajectory_path=str(validated.trajectory_path),
            analysis_path=str(validated.analysis_path),
            terrain_path=str(validated.terrain_path or ""),
            frames=validated.num_frames,
            frequency=validated.frequency,
            elapsed_seconds=time.time() - started,
        )
        terrain_record_path = config.run_root / "terrain" / f"{record.motion.replace('/', '__')}.json"
        if terrain_record_path.is_file():
            terrain_record = json.loads(terrain_record_path.read_text())
            base.update(
                terrain_model=terrain_record.get("fit", {}).get("model", ""),
                terrain_family_selected=terrain_record.get("selected_family", ""),
                terrain_family_correct=terrain_record.get("family_correct", ""),
                terrain_class=(terrain_class or terrain_record.get("selected_family", "")),
                terrain_validation_passed=terrain_record.get("validation", {}).get("passed", ""),
            )
        return base

    terrain_identity: dict[str, str] | None = None
    if config.method == "terra":
        if config.terrain_mode == "precomputed":
            terrain, report, validation, offsets, terrain_identity = _precomputed_terrain(config, record)
        else:
            terrain, report, validation, offsets = fit_record_terrain(config, record)
        terrain_input = terrain
        overrides = dict(config.retarget_overrides)
        overrides["calibrate_sites"] = config.calibrate_sites
        if offsets:
            overrides["source_sole_offsets"] = offsets
    else:
        if config.terrain_mode == "precomputed":
            terrain, report, validation, _offsets, terrain_identity = _precomputed_terrain(
                config, record, calibrate=False
            )
            terrain_input = terrain
        else:
            # The comparison setup reconstructs the scene exactly once with TERRA. Every
            # baseline consumes that immutable terrain metadata, including SMPL (which ignores terrain
            # during optimization but still needs it for identical playback and scoring).
            terra_artifacts = validate_retarget_artifacts(
                config.reference_cache_root,
                record.motion,
                method="terra",
                env_name=config.env_name,
            )
            if terra_artifacts.terrain_path is None:
                if config.terrain_mode != "flat":
                    raise FileNotFoundError(
                        f"authoritative TERRA terrain is missing for non-flat motion {record.motion}"
                    )
                terrain = None
                terrain_input = None
                report = {
                    "model": "flat",
                    "dataset_config": config.name,
                    "terrain_source_path": None,
                    "terrain_source": "authoritative_terra_implicit_flat_scene",
                }
            else:
                terrain = TerrainMetadata.load(terra_artifacts.terrain_path).terrain
                terrain_input = terra_artifacts.terrain_path
                report = {
                    "model": "shared_terra_metadata",
                    "dataset_config": config.name,
                    "terrain_source_path": str(terra_artifacts.terrain_path),
                }
            validation = {"passed": None, "source": "authoritative_terra_metadata"}
        offsets = {}
        overrides = dict(config.method_overrides)
    retarget_started = time.perf_counter()
    result = retarget(
        record.source_path,
        method=config.method,
        terrain=terrain_input,
        env_name=config.env_name,
        config=overrides,
        smpl_model_path=config.smpl_model_path,
        cache_root=config.cache_root,
        fitted_shape_path=(
            shape_cache_path(config.env_name, config.reference_cache_root)
            if config.method == "terra" and config.terrain_mode == "precomputed"
            else None
        ),
    )
    retarget_elapsed_s = time.perf_counter() - retarget_started
    position_error = np.asarray(result.analysis.get("pos_error"))
    if position_error.ndim < 1 or position_error.shape[0] < 1:
        raise ValueError("retarget analysis must contain at least one producer-native pos_error frame")
    benchmark_solved_frames = int(position_error.shape[0])
    result = replace(
        result,
        analysis=dict(result.analysis)
        | {
            "benchmark_retarget_elapsed_s": retarget_elapsed_s,
            "benchmark_solved_frames": benchmark_solved_frames,
            "benchmark_timing_context": benchmark_timing_context(),
            "benchmark_timing_scope": (
                "complete terra.api.retarget call; excludes terrain reconstruction and artifact publication"
            ),
            **(
                {
                    "terrain_reconstruction_source": terrain_identity,
                    "terrain_reconstruction_source_path": report["terrain_source_path"],
                }
                if terrain_identity is not None
                else {}
            ),
        },
    )
    artifacts = save_retarget_result(
        result,
        config.cache_root,
        record.motion,
        env_name=config.env_name,
        overwrite=overwrite,
    )
    selected_family = _selected_terrain_family(report, terrain)
    terrain_record = {
        "motion": record.motion,
        "dataset": config.name,
        "expected_family": expected_family,
        "selected_family": selected_family,
        "family_correct": not expected_family or selected_family == expected_family,
        "calibrated_joint_offsets_m": offsets,
        "terrain": terrain.to_dict() if terrain is not None else None,
        "fit": report,
        "validation": validation,
    }
    terrain_path = config.run_root / "terrain" / f"{record.motion.replace('/', '__')}.json"
    terrain_path.parent.mkdir(parents=True, exist_ok=True)
    terrain_path.write_text(json.dumps(terrain_record, indent=2, default=_json_default) + "\n")
    base.update(
        status="ok",
        trajectory_path=str(artifacts.trajectory_path),
        analysis_path=str(artifacts.analysis_path),
        terrain_path=str(artifacts.terrain_path or ""),
        frames=len(result.trajectory.data.qpos),
        frequency=float(result.trajectory.info.frequency),
        terrain_model=report.get("model", "flat"),
        terrain_family_selected=selected_family,
        terrain_family_correct=terrain_record["family_correct"],
        terrain_class=terrain_class or selected_family,
        terrain_validation_passed=bool(validation.get("passed", False)),
        elapsed_seconds=time.time() - started,
    )
    return base


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DatasetConfig",
    "MotionRecord",
    "benchmark_timing_context",
    "ensure_robot_shape",
    "fit_record_terrain",
    "load_dataset_config",
    "load_motion_records",
    "run_motion",
]
