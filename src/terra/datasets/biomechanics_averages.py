"""Subject/motion-type EMG and GRF traces for biomechanical validation.

The motion-level :mod:`terra.datasets.biomechanics` sidecars preserve exact capture
clocks.  This module builds the complementary comparison product used after a policy
rollout: one phase-normalized, trial-averaged trace for every subject and motion type,
plus a CSV mapping every converted motion to its trace.

Dataset-provided processed products take precedence.  Gait120's 101-point EMG,
Darmstadt's ``FullyProcessed`` EMG, and Vielemeyer's calculated NPZ GRFs are copied
without reimplementing their signal processing.  A trace is derived from synchronized
trial-level signals only where the release has no corresponding processed product.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from terra.datasets.biomechanics import biomechanics_path
from terra.paths import StorageRoots

TRIAL_AVERAGE_SCHEMA_VERSION = 2
PHASE_SAMPLES = 101
GAIT120_COMPLETE_CONTACT_MIN_FRACTION = 0.30

_DARMSTADT_MUSCLES = (
    ("BF", "BicepsFemoris"),
    ("RF", "RectusFemoris"),
    ("VL", "VastusLateralis"),
    ("GAS", "GastrocnemiusLateralis"),
    ("SOL", "Soleus"),
    ("TA", "TibialisAnterior"),
)
_DARMSTADT_HEIGHTS = {"riser_0.10": ("low", 1), "riser_0.17": ("normal", 2), "riser_0.24": ("high", 3)}
_DARMSTADT_STRIDE_SIDES = {
    "ascent": ("right", "left", "right", "left", "right", "left", "right", "left", "right", "left", "right"),
    "descent": ("left", "right", "left", "right", "left", "right", "left", "right", "left", "right", "left"),
}
_VIELEMEYER_CONDITIONS = {
    "level_down": "level_down",
    "level_up": "level_up",
    "ramp75_down": "ramp_75_down",
    "ramp75_up": "ramp_75_up",
    "ramp_10_down": "ramp_10_down",
    "ramp_10_up": "ramp_10_up",
}


def _empty_emg(strides: int, phases: int) -> dict[str, Any]:
    shape = (strides, phases, 0)
    return {
        "emg_available": np.array(False),
        "emg_mean": np.empty(shape, dtype=np.float32),
        "emg_std": np.empty(shape, dtype=np.float32),
        "emg_valid": np.empty(shape, dtype=bool),
        "emg_channels": np.empty(0, dtype="U1"),
        "emg_muscles": np.empty(0, dtype="U1"),
        "emg_channel_sides": np.empty(0, dtype="U1"),
        "emg_units": np.array(""),
        "emg_processing": np.array("unavailable"),
        "emg_source_kind": np.array("unavailable"),
        "emg_source_paths": np.empty(0, dtype="U1"),
        "emg_n_source_traces": np.zeros(strides, dtype=np.int64),
    }


def _empty_grf(strides: int, phases: int) -> dict[str, Any]:
    shape = (strides, phases, 0, 3)
    return {
        "grf_available": np.array(False),
        "grf_force_mean": np.empty(shape, dtype=np.float32),
        "grf_force_std": np.empty(shape, dtype=np.float32),
        "grf_moment_available": np.array(False),
        "grf_moment_mean": np.empty(shape, dtype=np.float32),
        "grf_moment_std": np.empty(shape, dtype=np.float32),
        "grf_valid": np.empty(shape[:-1], dtype=bool),
        "grf_channels": np.empty(0, dtype="U1"),
        "grf_axes": np.asarray(("x", "y", "z")),
        "grf_force_units": np.array(""),
        "grf_moment_units": np.array(""),
        "grf_coordinate_frame": np.array("unavailable"),
        "grf_processing": np.array("unavailable"),
        "grf_source_kind": np.array("unavailable"),
        "grf_source_paths": np.empty(0, dtype="U1"),
        "grf_n_source_traces": np.zeros((strides, 0), dtype=np.int64),
    }


def _base_payload(
    *,
    dataset: str,
    subject: str,
    motion_type: str,
    condition: str,
    direction: str,
    phase_percent: np.ndarray,
    phase_definition: str,
    stride_positions: Sequence[str],
    stride_sides: Sequence[str],
    source_motions: Sequence[str],
) -> dict[str, Any]:
    phase = np.asarray(phase_percent, dtype=np.float32).reshape(-1)
    positions = np.asarray(stride_positions)
    sides = np.asarray(stride_sides)
    if positions.shape != sides.shape or not len(positions):
        raise ValueError("stride position and side labels must be non-empty and aligned")
    payload: dict[str, Any] = {
        "schema_version": np.array(TRIAL_AVERAGE_SCHEMA_VERSION, dtype=np.int64),
        "dataset": np.array(dataset),
        "subject": np.array(subject),
        "motion_type": np.array(motion_type),
        "condition": np.array(condition),
        "direction": np.array(direction),
        "phase_percent": phase,
        "phase_definition": np.array(phase_definition),
        "stride_positions": positions,
        "stride_sides": sides,
        "source_motion_count": np.array(len(source_motions), dtype=np.int64),
        "source_motions": np.asarray(source_motions),
    }
    payload.update(_empty_emg(len(positions), len(phase)))
    payload.update(_empty_grf(len(positions), len(phase)))
    return payload


def write_trial_average(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write and validate one subject/motion-type comparison trace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(target)
    validate_trial_average(target)


def validate_trial_average(path: str | Path) -> dict[str, Any]:
    """Validate the portable trial-average schema and array alignment."""

    required = {
        "schema_version",
        "dataset",
        "subject",
        "motion_type",
        "condition",
        "direction",
        "phase_percent",
        "phase_definition",
        "stride_positions",
        "stride_sides",
        "source_motion_count",
        "source_motions",
        "emg_available",
        "emg_mean",
        "emg_std",
        "emg_valid",
        "emg_channels",
        "emg_muscles",
        "emg_channel_sides",
        "emg_units",
        "emg_processing",
        "emg_source_kind",
        "emg_source_paths",
        "emg_n_source_traces",
        "grf_available",
        "grf_force_mean",
        "grf_force_std",
        "grf_moment_available",
        "grf_moment_mean",
        "grf_moment_std",
        "grf_valid",
        "grf_channels",
        "grf_axes",
        "grf_force_units",
        "grf_moment_units",
        "grf_coordinate_frame",
        "grf_processing",
        "grf_source_kind",
        "grf_source_paths",
        "grf_n_source_traces",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"missing trial-average fields: {', '.join(missing)}")
        version = int(np.asarray(data["schema_version"]).reshape(()))
        dataset = str(np.asarray(data["dataset"]).reshape(()))
        subject = str(np.asarray(data["subject"]).reshape(()))
        phase = np.asarray(data["phase_percent"], dtype=np.float64)
        positions = np.asarray(data["stride_positions"]).reshape(-1)
        sides = np.asarray(data["stride_sides"]).reshape(-1)
        source_count = int(np.asarray(data["source_motion_count"]).reshape(()))
        source_motions = np.asarray(data["source_motions"]).reshape(-1)
        emg_available = bool(np.asarray(data["emg_available"]).reshape(()))
        emg_mean = np.asarray(data["emg_mean"])
        emg_std = np.asarray(data["emg_std"])
        emg_valid = np.asarray(data["emg_valid"], dtype=bool)
        emg_channels = np.asarray(data["emg_channels"]).reshape(-1)
        emg_muscles = np.asarray(data["emg_muscles"]).reshape(-1)
        emg_sides = np.asarray(data["emg_channel_sides"]).reshape(-1)
        emg_counts = np.asarray(data["emg_n_source_traces"], dtype=np.int64)
        grf_available = bool(np.asarray(data["grf_available"]).reshape(()))
        moment_available = bool(np.asarray(data["grf_moment_available"]).reshape(()))
        force_mean = np.asarray(data["grf_force_mean"])
        force_std = np.asarray(data["grf_force_std"])
        moment_mean = np.asarray(data["grf_moment_mean"])
        moment_std = np.asarray(data["grf_moment_std"])
        grf_valid = np.asarray(data["grf_valid"], dtype=bool)
        grf_channels = np.asarray(data["grf_channels"]).reshape(-1)
        grf_axes = np.asarray(data["grf_axes"]).reshape(-1)
        grf_counts = np.asarray(data["grf_n_source_traces"], dtype=np.int64)
    if version != TRIAL_AVERAGE_SCHEMA_VERSION:
        raise ValueError(f"unsupported trial-average schema version {version}")
    if phase.ndim != 1 or len(phase) < 2 or not np.isfinite(phase).all() or np.any(np.diff(phase) <= 0):
        raise ValueError("phase_percent must be finite and strictly increasing")
    if not np.isclose(phase[0], 0.0) or not np.isclose(phase[-1], 100.0):
        raise ValueError("phase_percent must span 0 through 100")
    if not len(positions) or len(sides) != len(positions):
        raise ValueError("stride labels are empty or misaligned")
    if source_count != len(source_motions) or source_count < 1:
        raise ValueError("source_motion_count does not match source_motions")
    emg_shape = (len(positions), len(phase), len(emg_channels))
    if emg_mean.shape != emg_shape or emg_std.shape != emg_shape or emg_valid.shape != emg_shape:
        raise ValueError(f"invalid EMG trace shape {emg_mean.shape}; expected {emg_shape}")
    if len(emg_muscles) != len(emg_channels) or len(emg_sides) != len(emg_channels):
        raise ValueError("EMG channel metadata is misaligned")
    if emg_counts.shape != (len(positions),):
        raise ValueError("emg_n_source_traces must have one value per stride position")
    if emg_available:
        if not len(emg_channels) or not emg_valid.any():
            raise ValueError("available EMG has no valid samples")
        if not np.isfinite(emg_mean[emg_valid]).all() or not np.isfinite(emg_std[emg_valid]).all():
            raise ValueError("valid EMG samples are non-finite")
    elif len(emg_channels):
        raise ValueError("unavailable EMG must have no channels")
    grf_shape = (len(positions), len(phase), len(grf_channels), 3)
    if force_mean.shape != grf_shape or force_std.shape != grf_shape:
        raise ValueError(f"invalid GRF trace shape {force_mean.shape}; expected {grf_shape}")
    if moment_mean.shape != grf_shape or moment_std.shape != grf_shape:
        raise ValueError("GRF moment arrays do not match force arrays")
    if grf_valid.shape != grf_shape[:-1] or grf_counts.shape != (len(positions), len(grf_channels)):
        raise ValueError("GRF validity/count arrays are misaligned")
    if grf_axes.shape != (3,):
        raise ValueError("grf_axes must label exactly three components")
    if grf_available:
        active = np.broadcast_to(grf_valid[..., None], grf_shape)
        if not len(grf_channels) or not grf_valid.any() or not np.isfinite(force_mean[active]).all():
            raise ValueError("available GRF has no finite valid samples")
        if not np.isfinite(force_std[active]).all():
            raise ValueError("valid GRF standard deviations are non-finite")
        if moment_available and not np.isfinite(moment_mean[active]).all():
            raise ValueError("valid GRF moments are non-finite")
    elif len(grf_channels):
        raise ValueError("unavailable GRF must have no channels")
    return {
        "dataset": dataset,
        "subject": subject,
        "phase_samples": len(phase),
        "stride_positions": len(positions),
        "emg_available": emg_available,
        "grf_available": grf_available,
    }


def _mean_std(traces: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not traces:
        raise ValueError("cannot average an empty trace collection")
    values = np.stack([np.asarray(trace, dtype=np.float64) for trace in traces])
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    total = np.where(finite, values, 0.0).sum(axis=0)
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    square = np.where(finite, (values - mean) ** 2, 0.0).sum(axis=0)
    std = np.sqrt(np.divide(square, count, out=np.full_like(square, np.nan), where=count > 0))
    return mean.astype(np.float32), std.astype(np.float32)


def _phase_resample(values: np.ndarray, samples: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or len(array) < 2:
        raise ValueError(f"cannot phase-normalize shape {array.shape}")
    source = np.linspace(0.0, 1.0, len(array))
    target = np.linspace(0.0, 1.0, samples)
    flat = array.reshape(len(array), -1)
    result = np.stack([np.interp(target, source, flat[:, index]) for index in range(flat.shape[1])], axis=-1)
    return result.reshape((samples, *array.shape[1:])).astype(np.float32)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"converted manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _motion_path(dataset_root: Path, row: dict[str, str]) -> Path:
    declared = Path(row.get("output_path", ""))
    if declared.is_absolute() and declared.is_file():
        return declared
    relative = dataset_root / declared
    if relative.is_file():
        return relative
    canonical = dataset_root / f"{row['motion']}.npz"
    if not canonical.is_file():
        raise FileNotFoundError(f"converted motion not found for {row['motion']}: {canonical}")
    return canonical


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


@dataclass(frozen=True)
class Gait120CycleWindow:
    """One released Gait120 step window and its fully recorded foot contacts."""

    source_path: Path
    start_time_s: float
    end_time_s: float
    complete_grf_sides: tuple[str, ...]


def gait120_grf_channels(sidecar: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """Return the canonical force channels represented by a Gait120 sidecar.

    Walking trials use separate left/right plates, whose order varies between
    recordings. Chair transitions use one plate under both feet and therefore
    expose a single ``combined`` channel.
    """

    if not bool(np.asarray(sidecar["grf_available"]).reshape(())):
        return ()
    declared = tuple(map(str, np.asarray(sidecar["grf_channels"]).reshape(-1)))
    if len(declared) == 2 and set(declared) == {"left", "right"}:
        return ("left", "right")
    if declared == ("combined",):
        return declared
    raise ValueError(f"unsupported Gait120 aggregate GRF channels {declared}")


def _gait120_source_path(path: Path, data_root: Path | None) -> Path:
    """Resolve a sidecar source path after moving the dataset to another host."""

    if path.is_file() or data_root is None:
        return path
    parts = path.parts
    try:
        anchor = parts.index("Gait120-original")
    except ValueError:
        return path
    relocated = data_root.joinpath(*parts[anchor:])
    return relocated if relocated.is_file() else path


def gait120_cycle_windows(
    sidecar: Mapping[str, np.ndarray],
    *,
    data_root: str | Path | None = None,
) -> list[Gait120CycleWindow]:
    """Identify force contacts fully represented by each synchronized step.

    The two force-plate channels are stored as zero outside plate contact.  A
    zero can therefore mean either physical swing or that the other foot's
    stance was not captured in this step file.  Across Gait120 there is a wide,
    unambiguous coverage gap: boundary spill occupies at most 19.6% of a step,
    whereas a complete stance occupies at least 41.1%.  Only contacts above the
    frozen 30% threshold are valid inputs to a gait-averaged GRF trace.
    """

    from terra.datasets.gait120 import load_gait120_trc

    source_start = float(np.asarray(sidecar["source_start_time_s"]).reshape(()))
    root = Path(data_root).expanduser().resolve() if data_root is not None else None
    paths = [
        _gait120_source_path(Path(str(value)), root) for value in np.asarray(sidecar["source_trc_paths"]).reshape(-1)
    ]
    grf_available = bool(np.asarray(sidecar["grf_available"]).reshape(()))
    if grf_available:
        native_time = np.asarray(sidecar["grf_native_time_s"], dtype=np.float64)
        valid = np.asarray(sidecar["grf_valid_native"], dtype=bool)
        declared_channels = tuple(map(str, np.asarray(sidecar["grf_channels"]).reshape(-1)))
        channels = gait120_grf_channels(sidecar)
        valid = valid[:, [declared_channels.index(channel) for channel in channels]]
        if native_time.ndim != 1 or valid.shape != (len(native_time), len(channels)):
            raise ValueError("Gait120 native GRF validity is misaligned with its clock and channels")
    else:
        native_time = np.empty(0, dtype=np.float64)
        valid = np.empty((0, 0), dtype=bool)
        channels = ()
    windows = []
    for path in paths:
        trc = load_gait120_trc(path)
        start = float(trc.times[0] - source_start)
        end = float(trc.times[-1] - source_start)
        complete: tuple[str, ...] = ()
        if grf_available:
            if start < native_time[0] - 1e-6 or end > native_time[-1] + 0.001:
                raise ValueError(f"Gait120 GRF does not cover gait cycle {path}")
            sample_period = float(np.median(np.diff(native_time))) if len(native_time) > 1 else 0.0
            within = (native_time >= start - 0.51 * sample_period) & (native_time <= end + 0.51 * sample_period)
            if not within.any():
                raise ValueError(f"Gait120 gait cycle has no native GRF samples: {path}")
            fractions = np.mean(valid[within], axis=0)
            complete = tuple(
                side
                for side, fraction in zip(channels, fractions, strict=True)
                if fraction >= GAIT120_COMPLETE_CONTACT_MIN_FRACTION
            )
        windows.append(Gait120CycleWindow(path, start, end, complete))
    return windows


def _trace_path(output_root: Path, dataset: str, subject: str, motion_type: str) -> Path:
    return output_root / dataset / subject / f"{_safe_name(motion_type)}.npz"


def _index_rows(
    rows: Sequence[dict[str, str]],
    *,
    dataset: str,
    subject: str,
    motion_type: str,
    condition: str,
    direction: str,
    trace_path: Path,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    key = f"{dataset}/{subject}/{motion_type}"
    return [
        {
            "dataset": dataset,
            "subject": subject,
            "motion_type": motion_type,
            "condition": condition,
            "direction": direction,
            "motion": row["motion"],
            "motion_biomechanics_path": str(biomechanics_path(_motion_path(Path(row["_dataset_root"]), row)).resolve()),
            "trace_key": key,
            "trace_path": str(trace_path.resolve()),
            "phase_definition": str(np.asarray(payload["phase_definition"]).reshape(())),
            "phase_samples": len(payload["phase_percent"]),
            "stride_positions": len(payload["stride_positions"]),
            "emg_available": bool(np.asarray(payload["emg_available"]).reshape(())),
            "grf_available": bool(np.asarray(payload["grf_available"]).reshape(())),
            "emg_source_kind": str(np.asarray(payload["emg_source_kind"]).reshape(())),
            "grf_source_kind": str(np.asarray(payload["grf_source_kind"]).reshape(())),
        }
        for row in rows
    ]


def _gait120_grf_cycles(
    sidecar_path: Path, *, data_root: Path | None = None
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    with np.load(sidecar_path, allow_pickle=False) as data:
        if not bool(np.asarray(data["grf_available"]).reshape(())):
            return [], [], []
        windows = gait120_cycle_windows(data, data_root=data_root)
        declared_channels = tuple(map(str, np.asarray(data["grf_channels"])))
        channels = gait120_grf_channels(data)
        order = [declared_channels.index(channel) for channel in channels]
        native_time = np.asarray(data["grf_native_time_s"], dtype=np.float64)
        force = np.asarray(data["grf_force_native"], dtype=np.float64)[:, order]
        moment = np.asarray(data["grf_moment_native"], dtype=np.float64)[:, order]
        source_paths = [str(value) for value in np.asarray(data["grf_source_paths"]).reshape(-1)]
    forces = []
    moments = []
    for window in windows:
        if not window.complete_grf_sides:
            continue
        target = np.linspace(window.start_time_s, window.end_time_s, PHASE_SAMPLES)
        target = np.clip(target, native_time[0], native_time[-1])
        sampled_force = np.stack(
            [
                np.interp(target, native_time, force[:, channel, axis])
                for channel in range(len(channels))
                for axis in range(3)
            ],
            axis=-1,
        ).reshape(PHASE_SAMPLES, len(channels), 3)
        sampled_moment = np.stack(
            [
                np.interp(target, native_time, moment[:, channel, axis])
                for channel in range(len(channels))
                for axis in range(3)
            ],
            axis=-1,
        ).reshape(PHASE_SAMPLES, len(channels), 3)
        for channel, label in enumerate(channels):
            if label not in window.complete_grf_sides:
                sampled_force[:, channel] = np.nan
                sampled_moment[:, channel] = np.nan
        forces.append(sampled_force.astype(np.float32))
        moments.append(sampled_moment.astype(np.float32))
    return forces, moments, source_paths


def _build_gait120(
    *, data_root: Path, artifact_root: Path, output_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_root = artifact_root / "gait120" / "smplh"
    rows = _read_manifest(dataset_root / "manifest.csv")
    for row in rows:
        row["_dataset_root"] = str(dataset_root)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(f"S{int(row['subject']):03d}", row["movement"])].append(row)
    matches: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for (subject, movement), group in sorted(grouped.items()):
        emg_traces: list[np.ndarray] = []
        emg_sources: set[str] = set()
        channels: np.ndarray | None = None
        grf_forces: list[np.ndarray] = []
        grf_moments: list[np.ndarray] = []
        grf_sources: set[str] = set()
        grf_channels: tuple[str, ...] | None = None
        for row in group:
            motion_path = _motion_path(dataset_root, row)
            emg_path = motion_path.with_name("AllSteps_emg.npz")
            with np.load(emg_path, allow_pickle=False) as data:
                values = np.asarray(data["emg"], dtype=np.float32)
                current_channels = np.asarray(data["channels"])
                if values.ndim != 3 or values.shape[1] != PHASE_SAMPLES:
                    raise ValueError(f"invalid processed Gait120 EMG shape {values.shape} in {emg_path}")
                if channels is not None and not np.array_equal(channels, current_channels):
                    raise ValueError(f"Gait120 EMG channel order differs in {emg_path}")
                channels = current_channels
                emg_traces.extend(values)
                emg_sources.add(str(np.asarray(data["source_mat"]).reshape(())))
            sidecar_path = biomechanics_path(motion_path)
            with np.load(sidecar_path, allow_pickle=False) as sidecar:
                current_grf_channels = gait120_grf_channels(sidecar)
            if current_grf_channels:
                if grf_channels is not None and current_grf_channels != grf_channels:
                    raise ValueError(
                        f"Gait120 GRF channel layout differs in {sidecar_path}: "
                        f"{current_grf_channels!r} versus {grf_channels!r}"
                    )
                grf_channels = current_grf_channels
            forces, moments, sources = _gait120_grf_cycles(sidecar_path, data_root=data_root)
            grf_forces.extend(forces)
            grf_moments.extend(moments)
            grf_sources.update(sources)
        if channels is None:
            raise ValueError(f"no processed Gait120 EMG for {subject}/{movement}")
        source_motions = sorted(row["motion"] for row in group)
        payload = _base_payload(
            dataset="gait120",
            subject=subject,
            motion_type=movement,
            condition=movement,
            direction="ascent" if movement.endswith("Ascent") else "descent" if movement.endswith("Descent") else "",
            phase_percent=np.linspace(0.0, 100.0, PHASE_SAMPLES),
            phase_definition=(
                "published transition phase (0-100%)"
                if movement in {"SitToStand", "StandToSit"}
                else "published gait cycle (0-100%)"
            ),
            stride_positions=("cycle",),
            stride_sides=("right",),
            source_motions=source_motions,
        )
        emg_mean, emg_std = _mean_std(emg_traces)
        payload.update(
            {
                "emg_available": np.array(True),
                "emg_mean": emg_mean[None],
                "emg_std": emg_std[None],
                "emg_valid": np.isfinite(emg_mean[None]),
                "emg_channels": channels,
                "emg_muscles": np.asarray([str(value).replace("Gastrocnemuis", "Gastrocnemius") for value in channels]),
                "emg_channel_sides": np.asarray(["right"] * len(channels)),
                "emg_units": np.array("dimensionless MVC-normalized envelope"),
                "emg_processing": np.array(
                    "release EMGs_interpolated; mean and population standard deviation across trials/steps"
                ),
                "emg_source_kind": np.array("release_processed_101_point"),
                "emg_source_paths": np.asarray(sorted(emg_sources)),
                "emg_n_source_traces": np.array([len(emg_traces)], dtype=np.int64),
            }
        )
        if grf_forces:
            force_mean, force_std = _mean_std(grf_forces)
            moment_mean, moment_std = _mean_std(grf_moments)
            grf_counts = np.sum(
                [np.isfinite(trace).all(axis=(0, 2)) for trace in grf_forces],
                axis=0,
                dtype=np.int64,
            )
            payload.update(
                {
                    "grf_available": np.array(True),
                    "grf_force_mean": force_mean[None],
                    "grf_force_std": force_std[None],
                    "grf_moment_available": np.array(True),
                    "grf_moment_mean": moment_mean[None],
                    "grf_moment_std": moment_std[None],
                    "grf_valid": np.isfinite(force_mean[None]).all(axis=-1),
                    "grf_channels": np.asarray(grf_channels),
                    "grf_axes": np.asarray(("terra_x", "terra_y", "terra_z")),
                    "grf_force_units": np.array("N"),
                    "grf_moment_units": np.array("Nm"),
                    "grf_coordinate_frame": np.array("TERRA Z-up right-handed laboratory frame"),
                    "grf_processing": np.array(
                        "trial-level MOT force/moment; complete force-plate contacts (>=30% step coverage) "
                        "selected before gait averaging; synchronized to each TRC gait cycle and linearly normalized "
                        "to 101 points"
                    ),
                    "grf_source_kind": np.array("derived_from_synchronized_trials"),
                    "grf_source_paths": np.asarray(sorted(grf_sources)),
                    "grf_n_source_traces": grf_counts[None],
                }
            )
        path = _trace_path(output_root, "gait120", subject, movement)
        write_trial_average(path, payload)
        matches.extend(
            _index_rows(
                group,
                dataset="gait120",
                subject=subject,
                motion_type=movement,
                condition=movement,
                direction=str(np.asarray(payload["direction"]).reshape(())),
                trace_path=path,
                payload=payload,
            )
        )
        aggregates.append(_aggregate_row(path, payload))
    return matches, aggregates


def _mat_trials(container: Any, configuration: int) -> np.ndarray:
    configurations = np.asarray(container, dtype=object).reshape(-1)
    return np.asarray(configurations[configuration - 1], dtype=object).reshape(-1)


def _mat_field(value: Any, field: str) -> np.ndarray:
    if hasattr(value, field):
        result = getattr(value, field)
    elif isinstance(value, np.void) and value.dtype.names and field in value.dtype.names:
        result = value[field]
    else:
        raise KeyError(f"MATLAB trial has no {field!r}")
    return np.asarray(result, dtype=np.float64)


def _darmstadt_stride_map(direction: str, base_configuration: int) -> list[tuple[int, str, int]]:
    if direction == "ascent":
        specifications = (
            (0, "r", 0),
            (0, "l", 0),
            (0, "r", 1),
            (0, "l", 1),
            (0, "r", 2),
            (0, "l", 2),
            (0, "r", 3),
            (3, "l", 1),
            (3, "r", 2),
            (3, "l", 2),
            (3, "r", 3),
        )
    else:
        specifications = (
            (3, "l", 0),
            (3, "r", 0),
            (3, "l", 1),
            (3, "r", 1),
            (0, "l", 0),
            (0, "r", 0),
            (0, "l", 1),
            (0, "r", 1),
            (0, "l", 2),
            (0, "r", 2),
            (0, "l", 3),
        )
    return [(base_configuration + offset, side, touchdown) for offset, side, touchdown in specifications]


def _darmstadt_grf(
    forces: Any,
    touchdowns: Any,
    *,
    base_configuration: int,
    direction: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    force_means = []
    force_stds = []
    moment_means = []
    moment_stds = []
    counts = []
    for configuration, side, touchdown_index in _darmstadt_stride_map(direction, base_configuration):
        force_trials = _mat_trials(forces, configuration)
        td_trials = _mat_trials(touchdowns, configuration)
        traces_force = []
        traces_moment = []
        for force_trial, td_trial in zip(force_trials, td_trials, strict=False):
            values = np.squeeze(_mat_field(force_trial, f"All_{side}"))
            touchdown = _mat_field(td_trial, f"td{side.upper()}").reshape(-1)
            if values.ndim != 2 or values.shape[1] != 6 or touchdown_index + 1 >= len(touchdown):
                continue
            start = round(float(touchdown[touchdown_index])) - 1
            end = round(float(touchdown[touchdown_index + 1])) - 1
            if start < 0 or end <= start or end >= len(values) or (end - start) / 200.0 > 2.0:
                continue
            segment = np.nan_to_num(values[start : end + 1], nan=0.0, posinf=0.0, neginf=0.0)
            sign = 1.0 if direction == "ascent" else -1.0
            transform = np.array(((0.0, sign, 0.0), (-sign, 0.0, 0.0), (0.0, 0.0, 1.0)))
            traces_force.append(_phase_resample(segment[:, :3] @ transform.T, 100)[:, None])
            traces_moment.append(_phase_resample(segment[:, 3:] @ transform.T, 100)[:, None])
        if not traces_force:
            raise ValueError(
                f"no usable Darmstadt GRF traces for configuration {configuration}, {direction}, side {side}"
            )
        force_mean, force_std = _mean_std(traces_force)
        moment_mean, moment_std = _mean_std(traces_moment)
        force_means.append(force_mean)
        force_stds.append(force_std)
        moment_means.append(moment_mean)
        moment_stds.append(moment_std)
        counts.append(len(traces_force))
    return (
        np.stack((np.stack(force_means), np.stack(force_stds))),
        np.stack((np.stack(moment_means), np.stack(moment_stds))),
        np.asarray(counts, dtype=np.int64)[:, None],
    )


def _build_darmstadt(
    *, data_root: Path, artifact_root: Path, output_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from scipy.io import loadmat

    dataset_root = artifact_root / "darmstadt" / "smplh"
    input_root = data_root / "Darmstadt-Stair-Ambulation"
    processed_path = input_root / "darmstadt-processed" / "FullyProcessed" / "Data_fully_processed.mat"
    processed = loadmat(processed_path, struct_as_record=False, squeeze_me=True)["data"]
    rows = _read_manifest(dataset_root / "manifest.csv")
    for row in rows:
        row["_dataset_root"] = str(dataset_root)
        parts = Path(row["motion"]).parts
        row["configuration"] = re.search(r"(\d+)$", parts[-3]).group(1)  # type: ignore[union-attr]
        row["direction"] = parts[-1].removesuffix("_stageii")
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["subject"], row["condition"], row["direction"])].append(row)
    matches: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    source_cache: dict[tuple[str, str], tuple[Any, Any, Path, Path]] = {}
    for (subject, condition, direction), group in sorted(grouped.items()):
        subject_number = int(subject.removeprefix("D"))
        cache_key = (subject, direction)
        if cache_key not in source_cache:
            force_path = input_root / "Preprocessed" / "Forces" / f"Forces{subject_number}.mat"
            touchdown_path = input_root / "touchdowns" / "Processed" / "Touchdowns" / f"Touchdowns{subject_number}.mat"
            force_data = loadmat(force_path, struct_as_record=False, squeeze_me=True)["Forces"]
            touchdown_data = loadmat(touchdown_path, struct_as_record=False, squeeze_me=True)[f"TD_{direction}"]
            source_cache[cache_key] = force_data, touchdown_data, force_path, touchdown_path
        force_data, touchdown_data, force_path, touchdown_path = source_cache[cache_key]
        height_name, base_configuration = _DARMSTADT_HEIGHTS[condition]
        direction_data = getattr(getattr(processed, height_name), direction)
        emg_mean = np.empty((11, 100, len(_DARMSTADT_MUSCLES)), dtype=np.float32)
        emg_std = np.empty_like(emg_mean)
        for stride in range(11):
            stride_data = getattr(direction_data, f"stride{stride + 1}")
            for channel, (field, _muscle) in enumerate(_DARMSTADT_MUSCLES):
                parameter = getattr(stride_data, field)
                emg_mean[stride, :, channel] = np.asarray(parameter.subj_mean, dtype=np.float32)[subject_number - 1]
                emg_std[stride, :, channel] = np.asarray(parameter.subj_std, dtype=np.float32)[subject_number - 1]
        grf_stats, moment_stats, grf_counts = _darmstadt_grf(
            force_data,
            touchdown_data,
            base_configuration=base_configuration,
            direction=direction,
        )
        motion_type = f"{condition}_{direction}"
        stride_prefix = "A" if direction == "ascent" else "D"
        source_motions = sorted(row["motion"] for row in group)
        payload = _base_payload(
            dataset="darmstadt",
            subject=subject,
            motion_type=motion_type,
            condition=condition,
            direction=direction,
            phase_percent=np.linspace(0.0, 100.0, 100),
            phase_definition="gait cycle from ipsilateral touchdown to next ipsilateral touchdown (0-100%)",
            stride_positions=tuple(f"{stride_prefix}{index}" for index in range(1, 12)),
            stride_sides=_DARMSTADT_STRIDE_SIDES[direction],
            source_motions=source_motions,
        )
        valid = np.isfinite(emg_mean) & np.isfinite(emg_std)
        payload.update(
            {
                "emg_available": np.array(bool(valid.any())),
                "emg_mean": emg_mean,
                "emg_std": emg_std,
                "emg_valid": valid,
                "emg_channels": np.asarray([field for field, _muscle in _DARMSTADT_MUSCLES]),
                "emg_muscles": np.asarray([muscle for _field, muscle in _DARMSTADT_MUSCLES]),
                "emg_channel_sides": np.asarray(["ipsilateral"] * len(_DARMSTADT_MUSCLES)),
                "emg_units": np.array("percent of subject maximum activity"),
                "emg_processing": np.array(
                    "release FullyProcessed subject mean/std; signal-error and outlier exclusions retained"
                ),
                "emg_source_kind": np.array("release_fully_processed_subject_mean"),
                "emg_source_paths": np.asarray([str(processed_path.resolve())]),
                # The release does not publish retained trial counts after per-muscle exclusions.
                "emg_n_source_traces": np.full(11, -1, dtype=np.int64),
                "grf_available": np.array(True),
                "grf_force_mean": grf_stats[0],
                "grf_force_std": grf_stats[1],
                "grf_moment_available": np.array(True),
                "grf_moment_mean": moment_stats[0],
                "grf_moment_std": moment_stats[1],
                "grf_valid": np.isfinite(grf_stats[0]).all(axis=-1),
                "grf_channels": np.asarray(("ipsilateral",)),
                "grf_axes": np.asarray(("forward", "left", "up")),
                "grf_force_units": np.array("N"),
                "grf_moment_units": np.array("Nm"),
                "grf_coordinate_frame": np.array("right-handed locomotion frame: forward, left, up"),
                "grf_processing": np.array(
                    "release All_l/All_r and touchdown stride map; NaN non-contact samples set to zero; usable strides <=2 s normalized to 100 points"
                ),
                "grf_source_kind": np.array("derived_from_synchronized_trials"),
                "grf_source_paths": np.asarray([str(force_path.resolve()), str(touchdown_path.resolve())]),
                "grf_n_source_traces": grf_counts,
            }
        )
        path = _trace_path(output_root, "darmstadt", subject, motion_type)
        write_trial_average(path, payload)
        matches.extend(
            _index_rows(
                group,
                dataset="darmstadt",
                subject=subject,
                motion_type=motion_type,
                condition=condition,
                direction=direction,
                trace_path=path,
                payload=payload,
            )
        )
        aggregates.append(_aggregate_row(path, payload))
    return matches, aggregates


def _vielemeyer_processed_traces(archive: Path) -> dict[str, tuple[list[np.ndarray], list[str]]]:
    grouped: dict[str, tuple[list[np.ndarray], list[str]]] = {}
    with zipfile.ZipFile(archive) as source:
        for member in sorted(name for name in source.namelist() if name.endswith(".npz")):
            source_condition = Path(member).parent.name
            if source_condition not in _VIELEMEYER_CONDITIONS:
                continue
            condition = _VIELEMEYER_CONDITIONS[source_condition]
            with np.load(io.BytesIO(source.read(member)), allow_pickle=False) as data:
                trace = np.stack(
                    [np.stack([data[f"GRF{axis}{contact}"] for axis in "xyz"], axis=-1) for contact in (1, 2)],
                    axis=1,
                ).astype(np.float32)
            if trace.shape != (PHASE_SAMPLES, 2, 3) or not np.isfinite(trace).all():
                raise ValueError(f"invalid calculated Vielemeyer GRF in {archive}!{member}: {trace.shape}")
            traces, paths = grouped.setdefault(condition, ([], []))
            traces.append(trace)
            paths.append(f"{archive.resolve()}!{member}")
    return grouped


def _vielemeyer_incomplete_trace(sidecar_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    with np.load(sidecar_path, allow_pickle=False) as data:
        force = np.asarray(data["grf_platform_force_native"], dtype=np.float64)
        moment = np.asarray(data["grf_platform_moment_native"], dtype=np.float64)
        valid = np.asarray(data["grf_platform_valid_native"], dtype=bool)
    contacts = []
    for platform in range(valid.shape[1]):
        indices = np.flatnonzero(valid[:, platform])
        if len(indices) > 1:
            contacts.append((int(indices[0]), int(indices[-1]), platform))
    contacts.sort()
    if len(contacts) < 2:
        return None
    force_traces = []
    moment_traces = []
    for start, end, platform in contacts[:2]:
        force_traces.append(_phase_resample(force[start : end + 1, platform], PHASE_SAMPLES))
        moment_traces.append(_phase_resample(moment[start : end + 1, platform], PHASE_SAMPLES))
    return np.stack(force_traces, axis=1), np.stack(moment_traces, axis=1)


def _build_vielemeyer(
    *, data_root: Path, artifact_root: Path, output_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_root = artifact_root / "vielemeyer" / "smplh"
    input_root = data_root / "Vielemeyer-Ramp-Walking"
    rows = _read_manifest(dataset_root / "manifest.csv")
    for row in rows:
        row["_dataset_root"] = str(dataset_root)
        row["direction"] = "up" if row["condition"].endswith("up") else "down"
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["subject"], row["condition"])].append(row)
    processed: dict[str, dict[str, tuple[list[np.ndarray], list[str]]]] = {}
    for archive in sorted(input_root.glob("Ref[0-9][0-9].zip")):
        processed[archive.stem] = _vielemeyer_processed_traces(archive)
    matches: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for (subject, condition), group in sorted(grouped.items()):
        moment_available = False
        if subject in processed and condition in processed[subject]:
            force_traces, source_paths = processed[subject][condition]
            moment_traces: list[np.ndarray] = []
            source_kind = "release_calculated_npz"
            force_units = "body weight"
            moment_units = ""
            axes = ("x", "y", "z")
            coordinate_frame = "Vielemeyer calculated-data release x/y/z frame"
            processing = "release GRF1/GRF2 normalized to contact time and body weight; mean and population standard deviation across trials"
        else:
            force_traces = []
            moment_traces = []
            source_paths = []
            for row in group:
                motion_path = _motion_path(dataset_root, row)
                trace = _vielemeyer_incomplete_trace(biomechanics_path(motion_path))
                if trace is not None:
                    force_trace, moment_trace = trace
                    force_traces.append(force_trace)
                    moment_traces.append(moment_trace)
                    source_paths.append(str(biomechanics_path(motion_path).resolve()))
            if not force_traces:
                raise ValueError(f"no usable Vielemeyer GRF traces for {subject}/{condition}")
            moment_available = True
            source_kind = "derived_incomplete_subject_c3d"
            force_units = "N"
            moment_units = "Nm"
            axes = ("terra_x", "terra_y", "terra_z")
            coordinate_frame = "TERRA Z-up right-handed laboratory frame"
            processing = "first two synchronized force-platform contacts, each normalized heel-strike-to-toe-off to 101 points; mean and population standard deviation across trials"
        force_mean, force_std = _mean_std(force_traces)
        if moment_available:
            moment_mean, moment_std = _mean_std(moment_traces)
        else:
            moment_mean = np.full_like(force_mean, np.nan)
            moment_std = np.full_like(force_mean, np.nan)
        source_motions = sorted(row["motion"] for row in group)
        payload = _base_payload(
            dataset="vielemeyer",
            subject=subject,
            motion_type=condition,
            condition=condition,
            direction="up" if condition.endswith("up") else "down",
            phase_percent=np.linspace(0.0, 100.0, PHASE_SAMPLES),
            phase_definition="contact time from heel strike to toe off (0-100%)",
            stride_positions=("trial_average",),
            stride_sides=("two_consecutive_contacts",),
            source_motions=source_motions,
        )
        payload.update(
            {
                "grf_available": np.array(True),
                "grf_force_mean": force_mean[None],
                "grf_force_std": force_std[None],
                "grf_moment_available": np.array(moment_available),
                "grf_moment_mean": moment_mean[None],
                "grf_moment_std": moment_std[None],
                "grf_valid": np.isfinite(force_mean[None]).all(axis=-1),
                "grf_channels": np.asarray(("contact1", "contact2")),
                "grf_axes": np.asarray(axes),
                "grf_force_units": np.array(force_units),
                "grf_moment_units": np.array(moment_units),
                "grf_coordinate_frame": np.array(coordinate_frame),
                "grf_processing": np.array(processing),
                "grf_source_kind": np.array(source_kind),
                "grf_source_paths": np.asarray(source_paths),
                "grf_n_source_traces": np.full((1, 2), len(force_traces), dtype=np.int64),
            }
        )
        path = _trace_path(output_root, "vielemeyer", subject, condition)
        write_trial_average(path, payload)
        matches.extend(
            _index_rows(
                group,
                dataset="vielemeyer",
                subject=subject,
                motion_type=condition,
                condition=condition,
                direction=str(np.asarray(payload["direction"]).reshape(())),
                trace_path=path,
                payload=payload,
            )
        )
        aggregates.append(_aggregate_row(path, payload))
    return matches, aggregates


def _aggregate_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": str(np.asarray(payload["dataset"]).reshape(())),
        "subject": str(np.asarray(payload["subject"]).reshape(())),
        "motion_type": str(np.asarray(payload["motion_type"]).reshape(())),
        "condition": str(np.asarray(payload["condition"]).reshape(())),
        "direction": str(np.asarray(payload["direction"]).reshape(())),
        "trace_path": str(path.resolve()),
        "source_motion_count": int(np.asarray(payload["source_motion_count"]).reshape(())),
        "phase_samples": len(payload["phase_percent"]),
        "stride_positions": len(payload["stride_positions"]),
        "emg_available": bool(np.asarray(payload["emg_available"]).reshape(())),
        "grf_available": bool(np.asarray(payload["grf_available"]).reshape(())),
        "emg_source_kind": str(np.asarray(payload["emg_source_kind"]).reshape(())),
        "grf_source_kind": str(np.asarray(payload["grf_source_kind"]).reshape(())),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty index {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_trial_averages(
    *,
    data_root: str | Path,
    artifact_root: str | Path,
    output_root: str | Path | None = None,
    datasets: Sequence[str] = ("gait120", "darmstadt", "vielemeyer"),
) -> dict[str, Any]:
    """Build all requested trace archives and their motion-to-trace registry."""

    data = Path(data_root).resolve()
    artifacts = Path(artifact_root).resolve()
    output = (
        Path(output_root).resolve() if output_root is not None else artifacts / "biomechanics" / "validation-traces"
    )
    builders = {
        "gait120": _build_gait120,
        "darmstadt": _build_darmstadt,
        "vielemeyer": _build_vielemeyer,
    }
    unknown = sorted(set(datasets) - set(builders))
    if unknown:
        raise ValueError(f"unsupported biomechanical datasets: {', '.join(unknown)}")
    matches: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_matches, dataset_aggregates = builders[dataset](
            data_root=data,
            artifact_root=artifacts,
            output_root=output,
        )
        matches.extend(dataset_matches)
        aggregates.extend(dataset_aggregates)
    matches.sort(key=lambda row: (row["dataset"], row["subject"], row["motion_type"], row["motion"]))
    aggregates.sort(key=lambda row: (row["dataset"], row["subject"], row["motion_type"]))
    _write_csv(output / "motion_trace_matches.csv", matches)
    _write_csv(output / "aggregates.csv", aggregates)
    report = {
        "schema_version": TRIAL_AVERAGE_SCHEMA_VERSION,
        "datasets": list(datasets),
        "aggregate_traces": len(aggregates),
        "matched_motions": len(matches),
        "emg_aggregate_traces": sum(bool(row["emg_available"]) for row in aggregates),
        "grf_aggregate_traces": sum(bool(row["grf_available"]) for row in aggregates),
        "output_root": str(output),
    }
    report_path = output / "validation.json"
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report


def _parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[3]
    roots = StorageRoots.from_environment(repository_root)
    parser = argparse.ArgumentParser(
        prog="terra biomechanics",
        description="Build subject/motion-type trial-averaged EMG and GRF validation traces.",
    )
    parser.add_argument("--data-root", type=Path, default=roots.data_root)
    parser.add_argument("--artifact-root", type=Path, default=roots.artifact_root)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=("gait120", "darmstadt", "vielemeyer"),
        dest="datasets",
        help="dataset to build; repeat to select multiple (default: all three)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = None if argv is None else list(argv)
    if arguments is not None and arguments[:1] == ["compare"]:
        from terra.biomechanics_validation import main as comparison_main

        return comparison_main(arguments[1:])
    if arguments is not None and arguments[:1] == ["plot"]:
        from terra.biomechanics_figures import main as figure_main

        return figure_main(arguments[1:])
    if arguments is not None and arguments[:1] == ["cohort"]:
        from terra.biomechanics_cohort import main as cohort_main

        return cohort_main(arguments[1:])
    if arguments is not None and arguments[:1] == ["build"]:
        arguments = arguments[1:]
    args = _parser().parse_args(arguments)
    report = build_trial_averages(
        data_root=args.data_root,
        artifact_root=args.artifact_root,
        output_root=args.output_root,
        datasets=tuple(args.datasets or ("gait120", "darmstadt", "vielemeyer")),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
