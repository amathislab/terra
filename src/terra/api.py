"""Stable orchestration API for retargeting external motions.

The solver-level functions in :mod:`terra.pipeline` accept already-loaded motion and
robot configuration objects. This module owns input dispatch, environment/model
resolution, method orchestration, stability retries, and run metadata. Persistent artifact I/O lives in
:mod:`terra.artifacts`; SMPL-H archive parsing lives in :mod:`terra.smplh`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from terra._c3d_options import resolve_c3d_options
from terra._methods import SUPPORTED_METHODS, RetargetingMethod, validate_method
from terra._musclemimic import (
    TerrainSpec,
    Trajectory,
    extend_motion,
    load_robot_conf_file,
    retarget_c3d_to_trajectory,
)
from terra._revision import write_git_commit
from terra.artifacts import (
    RETARGET_ARTIFACT_FORMAT_VERSION,
    RetargetArtifacts,
    RetargetPaths,
    ValidatedRetargetArtifacts,
    load_retarget_analysis,
    normalize_motion_name,
    retarget_cache_paths,
    save_retarget_result,
    validate_retarget_artifacts,
)
from terra.baselines import GMR_BASELINE, SMPL_BASELINE, fit_gmr_baseline, fit_smpl_baseline
from terra.baselines.gmr import runtime_paths as gmr_runtime_paths
from terra.contracts import RetargetResult
from terra.mat import MatSchemaInput, load_mat_schema, prepare_mat_marker_archive
from terra.pipeline import fit_terra_motion
from terra.profiles import resolve_method_profile
from terra.runtime import (
    ensure_environment_registered,
    ensure_robot_shape,
    resolve_cache_root,
    resolve_model_path,
    shape_cache_path,
)
from terra.smplh import load_smplh_motion
from terra.terrain.metadata import TerrainMetadata
from terra.trc import TrcUpAxis, prepare_trc_marker_archive

TerrainInput = TerrainSpec | Mapping[str, object] | str | Path | None


@dataclass(frozen=True)
class _RetargetRequest:
    """Validated inputs shared by the SMPL-H and C3D orchestration paths."""

    source: Path
    method: RetargetingMethod
    overrides: dict[str, object]
    logger: logging.Logger
    playback_terrain: TerrainSpec | None
    solver_terrain: TerrainSpec | Literal["auto"] | None
    terrain_metadata: dict[str, str]


def _resolve_source(path: str | Path, suffix: str) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"motion source not found: {source}")
    if source.suffix.casefold() != suffix:
        raise ValueError(f"expected a {suffix} motion, got {source.name!r}")
    return source


def _resolve_terrain_input(
    terrain: TerrainInput,
) -> tuple[TerrainSpec | None, TerrainSpec | Literal["auto"] | None, dict[str, str]]:
    """Return a playback terrain, solver input, and input metadata."""
    if terrain is None:
        return None, None, {}
    if isinstance(terrain, TerrainSpec):
        return terrain, terrain, {}
    if isinstance(terrain, Mapping):
        spec = TerrainSpec.from_dict(dict(terrain))
        return spec, spec, {}
    if isinstance(terrain, str) and terrain == "auto":
        return None, "auto", {}

    path = Path(terrain).expanduser().resolve()
    spec = TerrainMetadata.load(path).terrain
    return (
        spec,
        spec,
        {
            "terrain_source_path": str(path),
        },
    )


def _terrain_from_analysis(
    analysis: Mapping[str, object],
    fallback: TerrainSpec | None,
) -> TerrainSpec | None:
    value = analysis.get("terrain")
    if isinstance(value, Mapping):
        return TerrainMetadata.from_dict(dict(value)).terrain
    return fallback


def _environment_params(robot_conf) -> dict[str, object]:
    if isinstance(robot_conf, Mapping):
        return dict(robot_conf.get("env_params", {}))
    return dict(getattr(robot_conf, "env_params", {}))


def _gmr_config(
    overrides: Mapping[str, object],
    model_path: Path | None = None,
    cache_root: str | Path | None = None,
) -> dict[str, object]:
    resolved = GMR_BASELINE.resolved_config(overrides)
    resolved.pop("algorithm", None)
    resolved.pop("allow_cache_download", None)
    if model_path is not None:
        resolved["smpl_model_path"] = str(model_path)
    if cache_root is not None:
        resolved["cache_root"] = str(Path(cache_root).expanduser().resolve())
    return resolved


def _validate_solver_terrain(
    method: RetargetingMethod,
    terrain: TerrainSpec | Literal["auto"] | None,
) -> None:
    if terrain == "auto" and method not in {"terra", "omniretarget"}:
        raise ValueError(f"method={method!r} cannot reconstruct terrain itself; pass a TerrainSpec or JSON path")


def _validate_method_overrides(method: RetargetingMethod, overrides: Mapping[str, object]) -> None:
    if method == "smpl":
        unknown = sorted(set(overrides) - set(SMPL_BASELINE.config))
        if unknown:
            raise ValueError(
                "method='smpl' does not accept configuration overrides outside its "
                f"declared solve-rate controls; unknown field(s): {', '.join(unknown)}"
            )


def _prepare_request(
    source_path: str | Path,
    *,
    suffix: str,
    method: RetargetingMethod,
    terrain: TerrainInput,
    env_name: str,
    config: Mapping[str, object] | None,
    logger: logging.Logger | None,
) -> _RetargetRequest:
    """Validate and normalize the inputs common to every source format."""
    selected_method = validate_method(method)
    source = _resolve_source(source_path, suffix)
    overrides = dict(config or {})
    _validate_method_overrides(selected_method, overrides)
    ensure_environment_registered(env_name)
    playback_terrain, solver_terrain, terrain_metadata = _resolve_terrain_input(terrain)
    _validate_solver_terrain(selected_method, solver_terrain)
    return _RetargetRequest(
        source=source,
        method=selected_method,
        overrides=overrides,
        logger=logger or logging.getLogger("terra.retarget"),
        playback_terrain=playback_terrain,
        solver_terrain=solver_terrain,
        terrain_metadata=terrain_metadata,
    )


def _finalize_result(
    request: _RetargetRequest,
    trajectory: Trajectory,
    analysis: Mapping[str, object],
    metadata: Mapping[str, object],
) -> RetargetResult:
    """Attach common metadata and recover the fitted playback terrain."""
    recorded_analysis = {key: value for key, value in analysis.items() if not key.casefold().endswith("_sha256")}
    recorded_analysis.update(request.terrain_metadata)
    recorded_analysis.update(metadata)
    playback_terrain = _terrain_from_analysis(recorded_analysis, request.playback_terrain)
    return RetargetResult(
        trajectory,
        recorded_analysis,
        playback_terrain,
        request.method,
        request.source,
    )


def _apply_public_stability(
    result: RetargetResult,
    *,
    method: RetargetingMethod,
    stability_policy: Literal["retry", "off"],
    config: Mapping[str, object] | None,
    retry: Callable[[dict[str, object]], RetargetResult],
) -> RetargetResult:
    if stability_policy not in {"retry", "off"}:
        raise ValueError("stability_policy must be 'retry' or 'off'")
    if method != "terra" or stability_policy == "off":
        return result
    from terra.stability import apply_stability_policy

    return apply_stability_policy(result, config=config, retry=retry)


def _c3d_method_configs(
    method: RetargetingMethod,
    overrides: Mapping[str, object],
    terrain: TerrainSpec | Literal["auto"] | None,
    smpl_model_path: str | Path | None,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if method == "gmr":
        gmr_config = _gmr_config(overrides, resolve_model_path(smpl_model_path))
        if isinstance(terrain, TerrainSpec):
            gmr_config["terrain"] = terrain.to_dict()
        return gmr_config, None

    if method in {"terra", "omniretarget"}:
        terra_config = dict(overrides)
        if terrain is not None:
            terra_config["terrain"] = terrain if terrain == "auto" else terrain.to_dict()
        return None, terra_config

    return None, None


def retarget_smplh(
    source_path: str | Path,
    *,
    method: RetargetingMethod = "terra",
    terrain: TerrainInput = "auto",
    env_name: str = "MyoFullBody",
    config: Mapping[str, object] | None = None,
    smpl_model_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    fitted_shape_path: str | Path | None = None,
    logger: logging.Logger | None = None,
    stability_policy: Literal["retry", "off"] = "retry",
) -> RetargetResult:
    """Retarget an arbitrary AMASS-compatible SMPL-H archive.

    Args:
        source_path: One ``.npz`` archive containing AMASS ``poses``, ``trans``,
            ``betas``, ``gender``, and a supported frame-rate field. Canonical
            ``pose_aa``/``fps`` fields are also accepted.
        method: ``terra``, ``omniretarget``, ``gmr``, or ``smpl``.
        terrain: ``"auto"``, a :class:`TerrainSpec`, a compatible mapping or JSON
            path, or ``None`` for flat ground. Only TERRA and OmniRetarget can infer
            terrain from the motion.
        env_name: Registered MuscleMimic target environment.
        config: Method-specific overrides. MM-SMPL accepts only its declared
            solve-rate controls for already fitted SMPL-H input.
        smpl_model_path: SMPL-H model root. When omitted, use ``TERRA_MODEL_ROOT``.
        cache_root: Root for the fitted robot-shape cache. When omitted, use the
            direct cache below ``TERRA_ARTIFACT_ROOT``.
        fitted_shape_path: Existing fitted MyoFullBody shape to reuse. This keeps
            experimental output caches separate without changing body calibration.
        logger: Optional destination for fitting progress.

    Returns:
        A materialized trajectory, analysis metadata, and playback terrain.

    Raises:
        FileNotFoundError: If the source or resolved SMPL-H model root is absent.
        ValueError: If the archive, method, terrain, environment, or configuration is
            invalid.

    GMR and SMPL require explicit terrain when paired non-flat playback geometry is
    needed; SMPL intentionally does not use that geometry during optimization.
    """

    request = _prepare_request(
        source_path,
        suffix=".npz",
        method=method,
        terrain=terrain,
        env_name=env_name,
        config=config,
        logger=logger,
    )
    model_path = resolve_model_path(smpl_model_path)
    resolved_cache_root = resolve_cache_root(cache_root)
    write_git_commit(resolved_cache_root)
    robot_conf = load_robot_conf_file(env_name)
    shape_path: Path | None = None

    if request.method == "gmr":
        gmr_config = _gmr_config(request.overrides, model_path, resolved_cache_root)
        if request.solver_terrain is not None:
            gmr_config["terrain"] = request.solver_terrain.to_dict()
        trajectory, analysis = fit_gmr_baseline(
            env_name,
            robot_conf,
            str(request.source),
            request.logger,
            gmr_config,
        )
    else:
        motion_data = load_smplh_motion(request.source)
        if fitted_shape_path is None:
            shape_path = shape_cache_path(env_name, resolved_cache_root)
            ensure_robot_shape(env_name, robot_conf, model_path, shape_path, request.logger)
        else:
            shape_path = Path(fitted_shape_path).expanduser().resolve()
            if not shape_path.is_file():
                raise FileNotFoundError(f"fitted robot shape not found: {shape_path}")
        if request.method == "smpl":
            trajectory, analysis = fit_smpl_baseline(
                env_name,
                robot_conf,
                str(model_path),
                motion_data,
                str(shape_path),
                request.logger,
                request.overrides,
            )
        else:
            terra_config = resolve_method_profile(request.method, request.overrides)
            if request.solver_terrain is not None:
                terra_config["terrain"] = (
                    request.solver_terrain if request.solver_terrain == "auto" else request.solver_terrain.to_dict()
                )
            trajectory, analysis = fit_terra_motion(
                env_name,
                robot_conf,
                motion_data,
                request.logger,
                terra_config,
                smpl_model_path=model_path,
                fitted_shape_path=shape_path,
            )

    trajectory = extend_motion(env_name, _environment_params(robot_conf), trajectory, request.logger)
    metadata: dict[str, object] = {
        "source_path": str(request.source),
        "source_format": "smplh",
        "smpl_model_path": str(model_path),
        "retargeting_method": request.method,
    }
    if shape_path is not None:
        metadata.update(
            fitted_shape_path=str(shape_path),
        )
    result = _finalize_result(request, trajectory, analysis, metadata)
    return _apply_public_stability(
        result,
        method=request.method,
        stability_policy=stability_policy,
        config=config,
        retry=lambda overrides: retarget_smplh(
            source_path, method=method, terrain=terrain, env_name=env_name,
            config=overrides, smpl_model_path=smpl_model_path, cache_root=cache_root,
            fitted_shape_path=fitted_shape_path, logger=logger, stability_policy="off",
        ),
    )


def _retarget_marker_trajectory(
    source_path: str | Path,
    *,
    source_suffix: str,
    source_format: Literal["c3d", "mat", "trc"],
    fit_source_path: str | Path | None = None,
    trc_up_axis: TrcUpAxis = "y",
    mat_schema: MatSchemaInput | None = None,
    mat_selectors: Mapping[str, int] | None = None,
    method: RetargetingMethod = "terra",
    terrain: TerrainInput = "auto",
    env_name: str = "MyoFullBody",
    config: Mapping[str, object] | None = None,
    c3d_options: Mapping[str, object] | None = None,
    c3d_model_path: str | Path | None = None,
    smpl_model_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> RetargetResult:
    request = _prepare_request(
        source_path,
        suffix=source_suffix,
        method=method,
        terrain=terrain,
        env_name=env_name,
        config=config,
        logger=logger,
    )
    if request.method == "smpl" and request.overrides:
        raise ValueError("MM-SMPL solve-rate controls require an already fitted SMPL-H input")
    # Validate user-controlled options before resolving filesystem defaults so
    # malformed configuration fails at the public boundary.
    options = resolve_c3d_options(
        c3d_options,
        c3d_model_path=c3d_model_path,
        smpl_model_path=smpl_model_path,
        cache_root=cache_root,
    )
    if c3d_model_path is None:
        raise ValueError(
            "c3d_model_path is required for marker input; pass the downloaded SMPL-X/SMPL-H marker-fitting root"
        )
    model_path = resolve_model_path(smpl_model_path)
    marker_model_path = Path(c3d_model_path).expanduser().resolve()
    if not marker_model_path.exists():
        raise FileNotFoundError(
            f"Marker-fitting model root not found: {marker_model_path}. "
            "Set TERRA_MODEL_ROOT or pass c3d_model_path explicitly."
        )
    resolved_cache_root = resolve_cache_root(cache_root)
    write_git_commit(resolved_cache_root)
    if source_format == "trc":
        fit_source_path = prepare_trc_marker_archive(request.source, resolved_cache_root, up_axis=trc_up_axis)
    elif source_format == "mat":
        if mat_schema is None:
            raise ValueError("mat_schema is required for .mat marker input")
        resolved_mat_schema = load_mat_schema(mat_schema)
        fit_source_path = prepare_mat_marker_archive(
            request.source,
            resolved_mat_schema,
            resolved_cache_root,
            selectors=mat_selectors,
        )
    options.update(
        c3d_fit_model_path=str(marker_model_path),
        retarget_smpl_model_path=str(model_path),
        cache_root=str(resolved_cache_root),
    )
    gmr_config, terra_config = _c3d_method_configs(
        request.method,
        request.overrides,
        request.solver_terrain,
        model_path,
    )
    if gmr_config is not None:
        gmr_config["cache_root"] = str(resolved_cache_root)

    with gmr_runtime_paths(
        resolved_cache_root if request.method == "gmr" else None,
        model_path if request.method == "gmr" else None,
    ):
        trajectory, analysis = retarget_c3d_to_trajectory(
            str(Path(fit_source_path).expanduser().resolve() if fit_source_path is not None else request.source),
            env_name,
            retargeting_method=request.method,
            gmr_config=gmr_config,
            terra_config=terra_config,
            logger=logger,
            **options,
        )
    metadata: dict[str, object] = {
        "source": source_format,
        "source_path": str(request.source),
        "source_format": source_format,
        "retargeting_method": request.method,
        "c3d_model_path": str(marker_model_path),
        "smpl_model_path": str(model_path),
    }
    if fit_source_path is not None:
        metadata["source_marker_archive_path"] = str(Path(fit_source_path).expanduser().resolve())
    if source_format == "trc":
        metadata["trc_up_axis"] = trc_up_axis
    elif source_format == "mat":
        metadata["mat_schema"] = resolved_mat_schema
        metadata["mat_selectors"] = dict(mat_selectors or {})
    return _finalize_result(request, trajectory, analysis, metadata)


def retarget_c3d(
    source_path: str | Path,
    *,
    method: RetargetingMethod = "terra",
    terrain: TerrainInput = "auto",
    env_name: str = "MyoFullBody",
    config: Mapping[str, object] | None = None,
    c3d_options: Mapping[str, object] | None = None,
    c3d_model_path: str | Path | None = None,
    smpl_model_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    logger: logging.Logger | None = None,
    stability_policy: Literal["retry", "off"] = "retry",
) -> RetargetResult:
    """Fit C3D markers and return a MyoFullBody trajectory in memory.

    ``c3d_model_path`` is required for the marker surface fit. By default that
    fit uses a neutral SMPL-X model; pass
    ``c3d_options={"surface_model_type": "smplh"}`` to fit against SMPL-H.
    ``smpl_model_path`` separately points to the neutral SMPL-H model used
    for robot retargeting. The two roots can be equal for a SMPL-H surface fit.
    ``cache_root`` stores fitting intermediates and the robot shape.

    ``terrain="auto"`` reconstructs terrain for TERRA or OmniRetarget.
    The result is in memory; use :func:`save_retarget_result` to publish it.
    Fitting requires a marker set recognized by the selected surface model.
    """

    result = _retarget_marker_trajectory(
        source_path,
        source_suffix=".c3d",
        source_format="c3d",
        method=method,
        terrain=terrain,
        env_name=env_name,
        config=config,
        c3d_options=c3d_options,
        c3d_model_path=c3d_model_path,
        smpl_model_path=smpl_model_path,
        cache_root=cache_root,
        logger=logger,
    )

    return _apply_public_stability(
        result,
        method=method,
        stability_policy=stability_policy,
        config=config,
        retry=lambda overrides: retarget_c3d(
            source_path, method=method, terrain=terrain, env_name=env_name, config=overrides,
            c3d_options=c3d_options, c3d_model_path=c3d_model_path, smpl_model_path=smpl_model_path,
            cache_root=cache_root, logger=logger, stability_policy="off"
        ),
    )

def retarget_trc(
    source_path: str | Path,
    *,
    method: RetargetingMethod = "terra",
    terrain: TerrainInput = "auto",
    env_name: str = "MyoFullBody",
    config: Mapping[str, object] | None = None,
    c3d_options: Mapping[str, object] | None = None,
    c3d_model_path: str | Path | None = None,
    smpl_model_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    trc_up_axis: TrcUpAxis = "y",
    logger: logging.Logger | None = None,
    stability_policy: Literal["retry", "off"] = "retry",
) -> RetargetResult:
    """Normalize, fit, and retarget one TRC marker trajectory.

    The TRC is converted to metres and Z-up, then cached in a portable marker
    archive before fitting. ``trc_up_axis="y"`` applies the Gait120
    ``[X, -Z, Y]`` transform; ``"z"`` preserves recorded XYZ coordinates.
    Choose the axis that matches the source file, not the target terrain.
    As for :func:`retarget_c3d`, ``c3d_model_path`` identifies the marker
    surface model and ``smpl_model_path`` the retargeting SMPL-H model.
    The returned result is in memory until explicitly published.
    """

    if trc_up_axis not in {"y", "z"}:
        raise ValueError("trc_up_axis must be 'y' or 'z'")
    result = _retarget_marker_trajectory(
        source_path,
        source_suffix=".trc",
        source_format="trc",
        trc_up_axis=trc_up_axis,
        method=method,
        terrain=terrain,
        env_name=env_name,
        config=config,
        c3d_options=c3d_options,
        c3d_model_path=c3d_model_path,
        smpl_model_path=smpl_model_path,
        cache_root=cache_root,
        logger=logger,
    )

    return _apply_public_stability(
        result,
        method=method,
        stability_policy=stability_policy,
        config=config,
        retry=lambda overrides: retarget_trc(
            source_path, method=method, terrain=terrain, env_name=env_name, config=overrides,
            c3d_options=c3d_options, c3d_model_path=c3d_model_path, smpl_model_path=smpl_model_path,
            cache_root=cache_root, trc_up_axis=trc_up_axis, logger=logger, stability_policy="off"
        ),
    )

def retarget_mat(
    source_path: str | Path,
    *,
    mat_schema: MatSchemaInput,
    mat_selectors: Mapping[str, int] | None = None,
    method: RetargetingMethod = "terra",
    terrain: TerrainInput = "auto",
    env_name: str = "MyoFullBody",
    config: Mapping[str, object] | None = None,
    c3d_options: Mapping[str, object] | None = None,
    c3d_model_path: str | Path | None = None,
    smpl_model_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    logger: logging.Logger | None = None,
    stability_policy: Literal["retry", "off"] = "retry",
) -> RetargetResult:
    """Extract, fit, and retarget one marker trajectory from a MAT file.

    MAT containers have no universal marker layout. ``mat_schema`` is a
    mapping or JSON path that names a position tensor or per-marker fields,
    labels, frame rate, units, and coordinate transform. A nested record can
    be selected with zero-based ``mat_selectors`` indices. Extraction first
    converts coordinates to metres and Z-up. As for :func:`retarget_c3d`,
    ``c3d_model_path`` identifies the marker surface model and
    ``smpl_model_path`` the retargeting SMPL-H model. The returned result is
    in memory until explicitly published.
    """

    result = _retarget_marker_trajectory(
        source_path,
        source_suffix=".mat",
        source_format="mat",
        mat_schema=mat_schema,
        mat_selectors=mat_selectors,
        method=method,
        terrain=terrain,
        env_name=env_name,
        config=config,
        c3d_options=c3d_options,
        c3d_model_path=c3d_model_path,
        smpl_model_path=smpl_model_path,
        cache_root=cache_root,
        logger=logger,
    )

    return _apply_public_stability(
        result,
        method=method,
        stability_policy=stability_policy,
        config=config,
        retry=lambda overrides: retarget_mat(
            source_path, mat_schema=mat_schema, mat_selectors=mat_selectors, method=method,
            terrain=terrain, env_name=env_name, config=overrides, c3d_options=c3d_options,
            c3d_model_path=c3d_model_path, smpl_model_path=smpl_model_path, cache_root=cache_root,
            logger=logger, stability_policy="off"
        ),
    )

def retarget(
    source_path: str | Path,
    *,
    method: RetargetingMethod = "terra",
    terrain: TerrainInput = "auto",
    env_name: str = "MyoFullBody",
    config: Mapping[str, object] | None = None,
    smpl_model_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    fitted_shape_path: str | Path | None = None,
    c3d_options: Mapping[str, object] | None = None,
    c3d_model_path: str | Path | None = None,
    trc_up_axis: TrcUpAxis | None = None,
    mat_schema: MatSchemaInput | None = None,
    mat_selectors: Mapping[str, int] | None = None,
    logger: logging.Logger | None = None,
    stability_policy: Literal["retry", "off"] = "retry",
) -> RetargetResult:
    """Retarget one source motion and return its trajectory in memory.

    The source extension dispatches to SMPL-H ``.npz`` or marker ``.c3d``,
    ``.trc``, or ``.mat`` handling. TERRA's default ``terrain="auto"``
    reconstructs support geometry; pass ``None`` for flat ground or a
    compatible terrain metadata JSON path for known geometry. Marker inputs
    need ``c3d_model_path``; MAT also needs ``mat_schema``. The neutral
    SMPL-H model is resolved from ``smpl_model_path`` or ``TERRA_MODEL_ROOT``.

    The returned :class:`terra.contracts.RetargetResult` has ``trajectory``,
    ``analysis``, and optional ``terrain``. Publish it with
    :func:`save_retarget_result` before materializing a training cohort.
    For example::

        result = retarget(source, smpl_model_path=model_root,
                          cache_root=cache_root)
        save_retarget_result(result, cache_root, "Study/Trial")

    ``cache_root`` stores the fitted robot shape; passing the same root to
    publication keeps the cache self-contained. TERRA uses a final-trajectory
    stability gate and retries once with a conservative step size by default.

    Raises:
        ValueError: If the source format or format-specific options are invalid.
        FileNotFoundError: If the motion or a required model file is absent.
    """

    if stability_policy not in {"retry", "off"}:
        raise ValueError("stability_policy must be 'retry' or 'off'")
    suffix = Path(source_path).suffix.casefold()
    if suffix == ".npz":
        if c3d_options is not None or c3d_model_path is not None:
            raise ValueError("c3d_options and c3d_model_path require a marker source (.c3d, .trc, or .mat)")
        if trc_up_axis is not None:
            raise ValueError("trc_up_axis requires a .trc source")
        if mat_schema is not None or mat_selectors is not None:
            raise ValueError("mat_schema and mat_selectors require a .mat source")

        def run(overrides: Mapping[str, object] | None) -> RetargetResult:
            return retarget_smplh(
                source_path,
                method=method,
                terrain=terrain,
                env_name=env_name,
                config=overrides,
                smpl_model_path=smpl_model_path,
                cache_root=cache_root,
                fitted_shape_path=fitted_shape_path,
                logger=logger,
                stability_policy="off",
            )

    elif suffix == ".c3d":
        if trc_up_axis is not None:
            raise ValueError("trc_up_axis requires a .trc source")
        if mat_schema is not None or mat_selectors is not None:
            raise ValueError("mat_schema and mat_selectors require a .mat source")
        if fitted_shape_path is not None:
            raise ValueError("fitted_shape_path is supported only for an already fitted .npz source")

        def run(overrides: Mapping[str, object] | None) -> RetargetResult:
            return retarget_c3d(
                source_path,
                method=method,
                terrain=terrain,
                env_name=env_name,
                config=overrides,
                c3d_options=c3d_options,
                c3d_model_path=c3d_model_path,
                smpl_model_path=smpl_model_path,
                cache_root=cache_root,
                logger=logger,
                stability_policy="off",
            )

    elif suffix == ".trc":
        if mat_schema is not None or mat_selectors is not None:
            raise ValueError("mat_schema and mat_selectors require a .mat source")
        if fitted_shape_path is not None:
            raise ValueError("fitted_shape_path is supported only for an already fitted .npz source")

        def run(overrides: Mapping[str, object] | None) -> RetargetResult:
            return retarget_trc(
                source_path,
                method=method,
                terrain=terrain,
                env_name=env_name,
                config=overrides,
                c3d_options=c3d_options,
                c3d_model_path=c3d_model_path,
                smpl_model_path=smpl_model_path,
                cache_root=cache_root,
                trc_up_axis=trc_up_axis or "y",
                logger=logger,
                stability_policy="off",
            )

    elif suffix == ".mat":
        if fitted_shape_path is not None:
            raise ValueError("fitted_shape_path is supported only for an already fitted .npz source")
        if trc_up_axis is not None:
            raise ValueError("trc_up_axis requires a .trc source")
        if mat_schema is None:
            raise ValueError("mat_schema is required for .mat marker input")

        def run(overrides: Mapping[str, object] | None) -> RetargetResult:
            return retarget_mat(
                source_path,
                mat_schema=mat_schema,
                mat_selectors=mat_selectors,
                method=method,
                terrain=terrain,
                env_name=env_name,
                config=overrides,
                c3d_options=c3d_options,
                c3d_model_path=c3d_model_path,
                smpl_model_path=smpl_model_path,
                cache_root=cache_root,
                logger=logger,
                stability_policy="off",
            )

    else:
        raise ValueError(f"cannot infer motion format from extension {suffix!r}; expected .npz, .c3d, .trc, or .mat")

    result = run(config)
    if method != "terra" or stability_policy == "off":
        return result
    from terra.stability import apply_stability_policy

    return apply_stability_policy(result, config=config, retry=lambda overrides: run(overrides))


__all__ = [
    "RETARGET_ARTIFACT_FORMAT_VERSION",
    "SUPPORTED_METHODS",
    "RetargetArtifacts",
    "RetargetPaths",
    "RetargetResult",
    "RetargetingMethod",
    "TerrainInput",
    "ValidatedRetargetArtifacts",
    "load_retarget_analysis",
    "load_smplh_motion",
    "normalize_motion_name",
    "retarget",
    "retarget_c3d",
    "retarget_cache_paths",
    "retarget_mat",
    "retarget_smplh",
    "retarget_trc",
    "save_retarget_result",
    "validate_retarget_artifacts",
]
