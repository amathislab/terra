"""Publication and validation of retargeted motion artifacts.

This module owns the persistent wire format. It deliberately avoids importing the
retargeting solvers so callers can inspect completed artifacts in lightweight tools
and training launchers without initializing JAX, Torch, or MuJoCo.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import numpy as np

from terra._files import commit_staged_files, staged_write
from terra._methods import RetargetingMethod, validate_method
from terra._revision import write_git_commit

if TYPE_CHECKING:
    from terra.contracts import RetargetResult
    from terra.terrain.metadata import TerrainMetadata

_ANALYSIS_JSON_PREFIX = "__terra_json__:"
# Spaces occur in a small number of source AMASS clip names (notably EyesJapan).
# They are portable filename characters and Path handles them without shell parsing, so
# rejecting them loses otherwise valid benchmark motions.  Keep every other restriction:
# components must still be non-empty, relative, and free of platform punctuation.
_MOTION_NAME_PART = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._ -]*")
_TRAJECTORY_TIME_FIELDS = frozenset(
    {
        "qpos",
        "qvel",
        "xpos",
        "xquat",
        "cvel",
        "subtree_com",
        "cvel_parent",
        "subtree_com_root",
        "site_xpos",
        "site_xmat",
        "observations",
        "next_observations",
        "absorbings",
        "dones",
        "actions",
        "rewards",
    }
)


@dataclass(frozen=True)
class RetargetArtifacts:
    """Canonical paths returned after publication.

    ``terrain_path`` is ``None`` only when the result has no terrain file.
    """

    motion_name: str
    trajectory_path: Path
    analysis_path: Path
    terrain_path: Path | None


@dataclass(frozen=True)
class RetargetPaths:
    """Expected cache paths for a motion, whether or not the files exist."""

    motion_name: str
    trajectory_path: Path
    analysis_path: Path
    terrain_path: Path


@dataclass(frozen=True, slots=True)
class RetargetSegmentRequest:
    """One derived temporal artifact requested from a shared source motion."""

    motion_name: str
    start_frame: int
    end_frame_exclusive: int
    segment_index: int
    segment_count: int
    segment_policy: str


@dataclass(frozen=True)
class ValidatedRetargetArtifacts:
    """Validated metadata for one published retargeting artifact set.

    The trajectory arrays are checked without pickle deserialization but are not
    retained in memory. ``terrain_path`` is ``None`` for a flat result without a
    terrain file.
    """

    motion_name: str
    method: RetargetingMethod
    trajectory_path: Path
    analysis_path: Path
    terrain_path: Path | None
    num_frames: int
    frequency: float
    qpos_dimension: int
    qvel_dimension: int
    nonflat_terrain: bool


def normalize_motion_name(name: str | Path) -> Path:
    """Return a validated portable identifier for use inside trajectory caches.

    The identifier may contain nested path components, but never absolute paths,
    ``.``/``..`` components, empty components, or platform-specific punctuation.
    A trailing ``.npz``, ``.c3d``, ``.trc``, or ``.mat`` extension is removed.
    """

    raw = str(name).replace("\\", "/").strip()
    if not raw:
        raise ValueError("motion name cannot be empty")
    raw_parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"motion name must be a safe relative path, got {name!r}")
    relative = PurePosixPath(raw)
    if relative.is_absolute():
        raise ValueError(f"motion name must be a safe relative path, got {name!r}")
    if relative.suffix.casefold() in {".npz", ".c3d", ".trc", ".mat"}:
        relative = relative.with_suffix("")
    if not relative.parts or any(
        part != part.strip() or _MOTION_NAME_PART.fullmatch(part) is None for part in relative.parts
    ):
        raise ValueError(f"motion name must be a safe relative path, got {name!r}")
    return Path(*relative.parts)


def retarget_cache_paths(
    cache_root: str | Path,
    motion_name: str | Path,
    *,
    method: RetargetingMethod = "terra",
    env_name: str = "MyoFullBody",
) -> RetargetPaths:
    """Return expected paths for one portable motion ID.

    For the default environment and method, ``Study/Trial`` maps below
    ``cache_root/MyoFullBody/terra/Study/`` to ``Trial.npz``,
    ``Trial_analysis.npz``, and ``Trial_terrain.json``. The terrain path is
    returned even when the motion is flat and no terrain file exists.
    This function only constructs paths; use :func:`validate_retarget_artifacts`
    to check a published set.
    """

    from terra._musclemimic import retargeting_cache_dir

    selected_method = validate_method(method)
    relative = normalize_motion_name(motion_name)
    cache_dir = retargeting_cache_dir(
        Path(cache_root).expanduser().resolve(),
        env_name,
        selected_method,
    )
    base = cache_dir / relative
    trajectory_path = base.parent / f"{base.name}.npz"
    return RetargetPaths(
        motion_name=relative.as_posix(),
        trajectory_path=trajectory_path,
        analysis_path=base.parent / f"{base.name}_analysis.npz",
        terrain_path=base.parent / f"{base.name}_terrain.json",
    )


def load_retarget_analysis(path: str | Path) -> dict[str, object]:
    """Load a published analysis archive without enabling NumPy pickle.

    Structured values written by :func:`save_retarget_result` are decoded from the
    package's JSON envelope. Numeric arrays remain NumPy arrays.
    """

    analysis_path = Path(path).expanduser().resolve()
    with np.load(analysis_path, allow_pickle=False) as archive:
        return {name: _decode_analysis_value(archive[name]) for name in archive.files}


def validate_retarget_artifacts(
    cache_root: str | Path,
    motion_name: str | Path,
    *,
    method: RetargetingMethod = "terra",
    env_name: str = "MyoFullBody",
    require_nonflat_terrain: bool = False,
) -> ValidatedRetargetArtifacts:
    """Validate a published motion before playback or policy training.

    ``cache_root`` is the directory passed to :func:`save_retarget_result`;
    ``motion_name`` is its portable relative identifier, not a file path.
    The read side safely validates trajectory arrays and loads any terrain metadata. A flat result
    may have no terrain file. Set ``require_nonflat_terrain=True`` only when
    the selected cohort must have reconstructed non-flat support.

    Returns:
        Paths, frame count, frequency, dimensions, and non-flat status.

    Raises:
        FileNotFoundError: If a required artifact is absent.
        ValueError: If trajectory arrays or terrain are invalid.
    """

    from terra.terrain.metadata import TerrainMetadata

    selected_method = validate_method(method)
    paths = retarget_cache_paths(cache_root, motion_name, method=selected_method, env_name=env_name)
    required = [paths.trajectory_path, paths.analysis_path]
    if require_nonflat_terrain:
        required.append(paths.terrain_path)
    missing = [path for path in required if not path.is_file()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"retargeted artifact(s) not found: {rendered}")

    load_retarget_analysis(paths.analysis_path)

    num_frames, frequency, qpos_dimension, qvel_dimension = _validate_trajectory_archive(paths.trajectory_path)

    terrain_path = None
    terrain = None
    if paths.terrain_path.is_file():
        terrain = TerrainMetadata.load(paths.terrain_path).terrain
        terrain_path = paths.terrain_path

    nonflat_terrain = terrain is not None and not terrain.is_flat
    if require_nonflat_terrain and not nonflat_terrain:
        raise ValueError(f"retargeted artifact requires non-flat terrain: {paths.terrain_path}")
    return ValidatedRetargetArtifacts(
        motion_name=paths.motion_name,
        method=selected_method,
        trajectory_path=paths.trajectory_path,
        analysis_path=paths.analysis_path,
        terrain_path=terrain_path,
        num_frames=num_frames,
        frequency=frequency,
        qpos_dimension=qpos_dimension,
        qvel_dimension=qvel_dimension,
        nonflat_terrain=nonflat_terrain,
    )


def save_retarget_result(
    result: RetargetResult,
    cache_root: str | Path,
    motion_name: str | Path,
    *,
    env_name: str = "MyoFullBody",
    overwrite: bool = False,
) -> RetargetArtifacts:
    """Publish a retargeted motion in the cache layout consumed by PPO.

    ``motion_name`` is a portable relative ID such as ``Study/Trial``.
    This writes ``MyoFullBody/<method>/<motion>.npz``, a paired
    ``_analysis.npz`` report, and optional ``_terrain.json`` under
    ``cache_root``. Flat results may omit the terrain file. Files are staged
    and validated before publication; the analysis is committed last, and an
    interrupted replacement restores the preceding set. Analysis archives
    never require pickle deserialization.

    Returns:
        The published trajectory, analysis, and optional terrain paths.

    Raises:
        FileExistsError: If a target exists and ``overwrite`` is false.
        ValueError: If the name or serializable analysis is invalid.
    """

    paths = retarget_cache_paths(
        cache_root,
        motion_name,
        method=result.method,
        env_name=env_name,
    )
    terrain_metadata = _result_terrain_metadata(result)
    expected_outputs = [paths.trajectory_path, paths.analysis_path, paths.terrain_path]
    collisions = [path for path in expected_outputs if path.exists()]
    if collisions and not overwrite:
        rendered = ", ".join(str(path) for path in collisions)
        raise FileExistsError(f"retargeted artifact(s) already exist: {rendered}")

    analysis = dict(result.analysis)
    for field in (
        "motion_name",
        "retargeting_method",
        "terrain",
        "terrain_file",
        "terrain_path",
        "trajectory_file",
        "trajectory_path",
    ):
        analysis.pop(field, None)
    _encode_analysis(analysis)

    with ExitStack() as staging:
        staged_trajectory = staging.enter_context(
            staged_write(paths.trajectory_path, lambda path: result.trajectory.save(str(path)))
        )
        _validate_trajectory_archive(staged_trajectory)

        terrain_path = None
        staged_terrain = None
        if terrain_metadata is not None:
            staged_terrain = staging.enter_context(staged_write(paths.terrain_path, terrain_metadata.save))
            terrain_path = paths.terrain_path

        encoded_analysis = _encode_analysis(analysis)
        staged_analysis = staging.enter_context(
            staged_write(paths.analysis_path, lambda path: np.savez(path, **encoded_analysis))
        )
        _validate_staged_artifacts(staged_analysis, staged_trajectory, staged_terrain)
        operations = [(paths.trajectory_path, staged_trajectory)]
        if staged_terrain is not None:
            operations.append((paths.terrain_path, staged_terrain))
        elif overwrite and paths.terrain_path.exists():
            operations.append((paths.terrain_path, None))
        operations.append((paths.analysis_path, staged_analysis))
        commit_staged_files(operations)
    write_git_commit(Path(cache_root))
    return RetargetArtifacts(
        motion_name=paths.motion_name,
        trajectory_path=paths.trajectory_path,
        analysis_path=paths.analysis_path,
        terrain_path=terrain_path,
    )


def _load_segment_source_archive(source_path: Path, source_num_frames: int) -> dict[str, np.ndarray]:
    """Load and validate the fields shared by every segment of one source."""

    # These are trusted, already-validated local retargeting caches. Some static
    # trajectory metadata is stored in NumPy object arrays by MuscleMimic.
    with np.load(source_path, allow_pickle=True) as archive:
        if "split_points" not in archive.files:
            raise ValueError(f"trajectory archive has no split_points: {source_path}")
        split_points = np.asarray(archive["split_points"])
        if split_points.shape != (2,) or not np.array_equal(split_points, [0, source_num_frames]):
            raise ValueError(f"source trajectory must contain exactly one motion: {source_path}")
        values = {name: np.asarray(archive[name]) for name in archive.files}
    for name in _TRAJECTORY_TIME_FIELDS & values.keys():
        value = values[name]
        if value.size and (value.ndim < 1 or value.shape[0] != source_num_frames):
            raise ValueError(
                f"trajectory time field {name!r} does not match its {source_num_frames} frames: {source_path}"
            )
    return values


def _write_trajectory_segment(
    source_values: Mapping[str, np.ndarray],
    destination_path: Path,
    *,
    start_frame: int,
    end_frame_exclusive: int,
) -> None:
    """Write one interval from a source archive already resident in memory."""

    values: dict[str, np.ndarray] = {}
    for name, value in source_values.items():
        if name == "split_points":
            values[name] = np.asarray(
                [0, end_frame_exclusive - start_frame],
                dtype=value.dtype,
            )
        elif name in _TRAJECTORY_TIME_FIELDS and value.size:
            values[name] = value[start_frame:end_frame_exclusive].copy()
        else:
            values[name] = value
    np.savez(destination_path, **values)


def _validate_segment_request(request: RetargetSegmentRequest, source_num_frames: int) -> None:
    if isinstance(request.start_frame, bool) or not isinstance(request.start_frame, int) or request.start_frame < 0:
        raise ValueError("segment start_frame must be a non-negative integer")
    if (
        isinstance(request.end_frame_exclusive, bool)
        or not isinstance(request.end_frame_exclusive, int)
        or request.end_frame_exclusive > source_num_frames
        or request.end_frame_exclusive - request.start_frame < 2
    ):
        raise ValueError("segment end_frame_exclusive must define at least two source frames")
    if request.segment_count < 2 or not 1 <= request.segment_index <= request.segment_count:
        raise ValueError("segment index/count must identify one of at least two segments")
    if not isinstance(request.segment_policy, str) or not request.segment_policy:
        raise ValueError("segment_policy must be non-empty")


def _save_loaded_retarget_segment(
    source: ValidatedRetargetArtifacts,
    source_values: Mapping[str, np.ndarray],
    destination_cache_root: str | Path,
    request: RetargetSegmentRequest,
    *,
    method: RetargetingMethod,
    env_name: str,
    overwrite: bool,
) -> RetargetArtifacts:
    paths = retarget_cache_paths(destination_cache_root, request.motion_name, method=method, env_name=env_name)
    if paths.motion_name == source.motion_name and paths.trajectory_path == source.trajectory_path:
        raise ValueError("segment destination must differ from its source artifact")
    expected_outputs = [paths.trajectory_path, paths.analysis_path, paths.terrain_path]
    collisions = [path for path in expected_outputs if path.exists()]
    if collisions and not overwrite:
        rendered = ", ".join(str(path) for path in collisions)
        raise FileExistsError(f"retargeted segment artifact(s) already exist: {rendered}")

    analysis: dict[str, object] = {
        "segment_start_frame": request.start_frame,
        "segment_end_frame_exclusive": request.end_frame_exclusive,
        "segment_index": request.segment_index,
        "segment_count": request.segment_count,
        "segment_frames": request.end_frame_exclusive - request.start_frame,
        "segment_duration_s": (request.end_frame_exclusive - request.start_frame) / source.frequency,
    }
    with ExitStack() as staging:
        staged_trajectory = staging.enter_context(
            staged_write(
                paths.trajectory_path,
                lambda path: _write_trajectory_segment(
                    source_values,
                    path,
                    start_frame=request.start_frame,
                    end_frame_exclusive=request.end_frame_exclusive,
                ),
            )
        )
        _validate_trajectory_archive(staged_trajectory)

        staged_terrain = None
        if source.terrain_path is not None:
            staged_terrain = staging.enter_context(
                staged_write(paths.terrain_path, lambda path: shutil.copy2(source.terrain_path, path))
            )

        encoded_analysis = _encode_analysis(analysis)
        staged_analysis = staging.enter_context(
            staged_write(paths.analysis_path, lambda path: np.savez(path, **encoded_analysis))
        )
        _validate_staged_artifacts(staged_analysis, staged_trajectory, staged_terrain)
        operations = [(paths.trajectory_path, staged_trajectory)]
        if staged_terrain is not None:
            operations.append((paths.terrain_path, staged_terrain))
        elif overwrite and paths.terrain_path.exists():
            operations.append((paths.terrain_path, None))
        operations.append((paths.analysis_path, staged_analysis))
        commit_staged_files(operations)
    return RetargetArtifacts(
        motion_name=paths.motion_name,
        trajectory_path=paths.trajectory_path,
        analysis_path=paths.analysis_path,
        terrain_path=paths.terrain_path if source.terrain_path is not None else None,
    )


def save_retarget_segments(
    source_cache_root: str | Path,
    source_motion_name: str | Path,
    destination_cache_root: str | Path,
    requests: Sequence[RetargetSegmentRequest],
    *,
    method: RetargetingMethod = "terra",
    env_name: str = "MyoFullBody",
    overwrite: bool = False,
) -> tuple[RetargetArtifacts, ...]:
    """Publish several canonical intervals while reading their source only once."""

    if not requests:
        raise ValueError("at least one retarget segment request is required")
    if len({request.motion_name for request in requests}) != len(requests):
        raise ValueError("retarget segment request motion names must be unique")
    selected_method = validate_method(method)
    source = validate_retarget_artifacts(
        source_cache_root,
        source_motion_name,
        method=selected_method,
        env_name=env_name,
    )
    for request in requests:
        _validate_segment_request(request, source.num_frames)
    source_values = _load_segment_source_archive(source.trajectory_path, source.num_frames)
    artifacts = tuple(
        _save_loaded_retarget_segment(
            source,
            source_values,
            destination_cache_root,
            request,
            method=selected_method,
            env_name=env_name,
            overwrite=overwrite,
        )
        for request in requests
    )
    write_git_commit(Path(destination_cache_root))
    return artifacts


def save_retarget_segment(
    source_cache_root: str | Path,
    source_motion_name: str | Path,
    destination_cache_root: str | Path,
    segment_motion_name: str | Path,
    *,
    start_frame: int,
    end_frame_exclusive: int,
    segment_index: int,
    segment_count: int,
    segment_policy: str,
    method: RetargetingMethod = "terra",
    env_name: str = "MyoFullBody",
    overwrite: bool = False,
) -> RetargetArtifacts:
    """Publish a canonical training view of one retargeted trajectory interval.

    The trajectory's time-indexed arrays are sliced without recomputing any
    kinematics. Static model data and the paired terrain remain byte-for-byte
    equivalent to the source. The derived analysis archive describes the selected
    frame interval.
    """

    return save_retarget_segments(
        source_cache_root,
        source_motion_name,
        destination_cache_root,
        (
            RetargetSegmentRequest(
                motion_name=str(segment_motion_name),
                start_frame=start_frame,
                end_frame_exclusive=end_frame_exclusive,
                segment_index=segment_index,
                segment_count=segment_count,
                segment_policy=segment_policy,
            ),
        ),
        method=method,
        env_name=env_name,
        overwrite=overwrite,
    )[0]


def _validate_staged_artifacts(
    analysis_path: Path,
    trajectory_path: Path,
    terrain_path: Path | None,
) -> None:
    """Read back staged files and verify them before publication."""

    load_retarget_analysis(analysis_path)
    _validate_trajectory_archive(trajectory_path)
    if terrain_path is not None:
        from terra.terrain.metadata import TerrainMetadata

        TerrainMetadata.load(terrain_path)


def _analysis_json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set | frozenset):
        return sorted(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(f"analysis value of type {type(value).__name__} is not serializable")


def _encode_analysis(analysis: Mapping[str, object]) -> dict[str, np.ndarray | object]:
    encoded: dict[str, np.ndarray | object] = {}
    for key, value in analysis.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"analysis keys must be non-empty strings, got {key!r}")
        try:
            array = np.asarray(value)
        except ValueError:
            array = np.asarray(value, dtype=object)
        if array.dtype != object:
            encoded[key] = array
            continue
        try:
            payload = json.dumps(
                value,
                default=_analysis_json_default,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"analysis value {key!r} is not safely serializable: {exc}") from exc
        encoded[key] = np.asarray(_ANALYSIS_JSON_PREFIX + payload)
    return encoded


def _decode_analysis_value(value: object) -> object:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, str) and value.startswith(_ANALYSIS_JSON_PREFIX):
        return json.loads(value.removeprefix(_ANALYSIS_JSON_PREFIX))
    return value


def _validate_trajectory_archive(path: Path) -> tuple[int, float, int, int]:
    """Validate the trajectory fields shared by playback and PPO."""

    with np.load(path, allow_pickle=False) as archive:
        missing = sorted({"frequency", "qpos", "qvel"} - set(archive.files))
        if missing:
            raise ValueError(f"trajectory archive is missing {', '.join(missing)}: {path}")
        qpos = np.asarray(archive["qpos"])
        qvel = np.asarray(archive["qvel"])
        frequency_value = np.asarray(archive["frequency"])

    if qpos.ndim != 2 or qpos.shape[0] < 2 or qpos.shape[1] < 7:
        raise ValueError(f"trajectory qpos must have shape (frames>=2, nq>=7), got {qpos.shape}: {path}")
    if qvel.ndim != 2 or qvel.shape[0] != qpos.shape[0] or qvel.shape[1] < 6:
        raise ValueError(f"trajectory qvel must have shape ({len(qpos)}, nv>=6), got {qvel.shape}: {path}")
    if not all(
        np.issubdtype(array.dtype, np.number) and not np.issubdtype(array.dtype, np.complexfloating)
        for array in (qpos, qvel)
    ):
        raise ValueError(f"trajectory qpos/qvel must be real numeric arrays: {path}")
    if not (np.isfinite(qpos).all() and np.isfinite(qvel).all()):
        raise ValueError(f"trajectory contains non-finite qpos or qvel values: {path}")
    if frequency_value.size != 1:
        raise ValueError(f"trajectory frequency must be scalar, got shape {frequency_value.shape}: {path}")
    try:
        frequency = float(frequency_value.reshape(()).item())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"trajectory frequency must be numeric: {path}") from exc
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError(f"trajectory frequency must be positive and finite, got {frequency!r}: {path}")
    return len(qpos), frequency, qpos.shape[1], qvel.shape[1]


def _result_terrain_metadata(result: RetargetResult) -> TerrainMetadata | None:
    from terra.terrain.metadata import TerrainMetadata

    serialized = result.analysis.get("terrain")
    if serialized is not None and not isinstance(serialized, Mapping):
        raise ValueError("result analysis terrain must be a mapping or None")
    metadata = None if serialized is None else TerrainMetadata.from_dict(dict(serialized))
    if metadata is None:
        return None if result.terrain is None else TerrainMetadata.from_terrain(result.terrain)
    if result.terrain is None:
        raise ValueError("result analysis contains terrain but result.terrain is None")
    if metadata.terrain != result.terrain:
        raise ValueError("result.terrain does not match the target terrain recorded in result.analysis")
    return metadata


__all__ = [
    "RetargetArtifacts",
    "RetargetPaths",
    "RetargetSegmentRequest",
    "ValidatedRetargetArtifacts",
    "load_retarget_analysis",
    "normalize_motion_name",
    "retarget_cache_paths",
    "save_retarget_result",
    "save_retarget_segment",
    "save_retarget_segments",
    "validate_retarget_artifacts",
]
