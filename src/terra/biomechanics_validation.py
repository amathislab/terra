"""Roll out a policy and compare muscle activation and GRFs with measured traces.

The comparison deliberately has no temporal-shift optimization.  Every simulated
signal is sampled on every fully completed experimental phase window recorded in
the synchronized motion sidecar, normalized per the declared measurement
convention, and only then averaged over completed gaits. A gait completed before
an episode terminates remains a valid physiological observation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

STANDARD_GRAVITY_M_S2 = 9.80665
COMPARISON_SCHEMA_VERSION = 5
# One 50-Hz producer boundary trim plus one 100-Hz source-to-SMPL
# end-point quantization interval, with 5 ms numerical allowance.
SOURCE_CLOCK_BOUNDARY_TOLERANCE_S = 0.035

# A fixed anatomical mapping from recorded superficial muscles to MyoFullBody
# actuators.  Bilateral actuator names are formed by appending ``_l``/``_r``.
EMG_ACTUATOR_BASES: dict[str, tuple[str, ...]] = {
    "bicepsfemoris": ("bflh", "bfsh"),
    "rectusfemoris": ("recfem",),
    "vastuslateralis": ("vaslat",),
    "vastusmedialis": ("vasmed",),
    "tibialisanterior": ("tibant",),
    "semitendinosus": ("semiten",),
    "gastrocnemiusmedialis": ("gasmed",),
    "gastrocnemiuslateralis": ("gaslat",),
    "soleus": ("soleus",),
    "soleusmedialis": ("soleus",),
    "soleuslateralis": ("soleus",),
    "peroneuslongus": ("perlong",),
    "peroneusbrevis": ("perbrev",),
}

# Masses from the release authors' anthropometrics.ini, used to normalize
# their calculated GRF data: github.com/jvielemeyer/human-ramp-walking.
# Complete-subject calculated traces
# are already normalized by body weight; these masses make the N-valued Ref03
# and Ref04 fallback traces compatible with the same comparison.
_VIELEMEYER_MASS_KG = {
    "Ref01": 65.0,
    "Ref02": 93.0,
    "Ref03": 62.0,
    "Ref04": 70.0,
    "Ref05": 65.0,
    "Ref06": 66.5,
    "Ref07": 90.0,
    "Ref08": 67.0,
    "Ref09": 57.0,
    "Ref10": 68.0,
    "Ref11": 65.0,
    "Ref12": 96.0,
    "Ref13": 85.0,
}


@dataclass(frozen=True)
class TraceMatch:
    """One exact converted-motion to experimental-trace registry match."""

    dataset: str
    subject: str
    motion_type: str
    condition: str
    direction: str
    motion: str
    sidecar_path: Path
    trace_path: Path


@dataclass(frozen=True)
class PhaseWindow:
    """A source-clock interval mapped to a position in the averaged trace."""

    position_index: int
    label: str
    side: str
    start_time_s: float
    end_time_s: float
    # ``None`` selects every target GRF channel, while an empty tuple selects
    # none.  Explicit selection lets Gait120 retain all EMG windows but compare
    # GRFs only where a complete force-plate contact was recorded.
    grf_channel_indices: tuple[int, ...] | None = None


@dataclass
class RolloutCapture:
    """Unmodified samples recorded from one CPU MuJoCo episode."""

    seed: int
    success: bool
    absorbing: bool
    coverage: float
    return_per_frame: float
    time_s: np.ndarray
    actuator_names: tuple[str, ...]
    activation: np.ndarray
    grf_world_n: np.ndarray
    root_position_m: np.ndarray
    qpos: np.ndarray | None = None
    # Retarget producer timestamps, indexed by the reference frame actually
    # visited by the environment. These must not be inferred from episode length.
    source_time_s: np.ndarray | None = None
    reference_frame: np.ndarray | None = None


class NoSuccessfulRolloutError(RuntimeError):
    """All requested policy episodes failed before physiological scoring."""

    def __init__(self, failed_rollouts: int) -> None:
        self.successful_rollouts = 0
        self.failed_rollouts = failed_rollouts
        super().__init__("no successful rollout completed; physiological metrics cannot be computed")


class NoCompletedGaitError(NoSuccessfulRolloutError):
    """No episode reached the end of any requested experimental gait window."""

    def __init__(self, captures: list[RolloutCapture]) -> None:
        completed_episodes = sum(capture.success for capture in captures)
        self.successful_rollouts = completed_episodes
        self.failed_rollouts = len(captures) - completed_episodes
        self.measured_rollouts = 0
        self.measured_early_terminated_rollouts = 0
        self.completed_gaits = 0
        RuntimeError.__init__(
            self,
            "no rollout completed a requested gait; physiological metrics cannot be computed",
        )


def _scalar(data: dict[str, np.ndarray], key: str) -> str:
    return str(np.asarray(data[key]).reshape(()))


def _canonical_motion(value: str | Path) -> str:
    text = str(value).replace("\\", "/").removesuffix(".npz").rstrip("/")
    marker = "/MyoFullBody/terra/"
    if marker in text:
        text = text.split(marker, 1)[1]
    return text


def resolve_trace_match(
    motion: str | Path,
    *,
    trace_root: str | Path,
    artifact_root: str | Path,
) -> TraceMatch:
    """Resolve a logical motion, converted NPZ, or retarget-cache NPZ exactly."""

    trace_root = Path(trace_root).resolve()
    registry = trace_root / "motion_trace_matches.csv"
    if not registry.is_file():
        raise FileNotFoundError(f"biomechanics trace registry not found: {registry}")
    with registry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    query = _canonical_motion(motion)
    candidates = [row for row in rows if query == row["motion"] or query.endswith(f"/{row['motion']}")]
    if len(candidates) != 1:
        suffix = "" if candidates else " no"
        raise ValueError(f"motion {motion!s} has{suffix} unique registry match ({len(candidates)} found)")
    row = candidates[0]
    trace_path = Path(row["trace_path"])
    if not trace_path.is_file():
        trace_path = trace_root / row["dataset"] / row["subject"] / f"{_safe_name(row['motion_type'])}.npz"
    sidecar_path = Path(row["motion_biomechanics_path"])
    if not sidecar_path.is_file():
        relative = Path(f"{row['motion']}_biomechanics.npz")
        sidecar_path = Path(artifact_root).resolve() / row["dataset"] / "smplh" / relative
    if not trace_path.is_file():
        raise FileNotFoundError(f"matched experimental trace not found: {trace_path}")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"matched motion biomechanics sidecar not found: {sidecar_path}")
    return TraceMatch(
        dataset=row["dataset"],
        subject=row["subject"],
        motion_type=row["motion_type"],
        condition=row["condition"],
        direction=row["direction"],
        motion=row["motion"],
        sidecar_path=sidecar_path.resolve(),
        trace_path=trace_path.resolve(),
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_trace(path: str | Path) -> dict[str, np.ndarray]:
    """Load an averaged trace without retaining an open NPZ handle."""

    from terra.datasets.biomechanics_averages import validate_trial_average

    validate_trial_average(path)
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def experimental_mass_kg(match: TraceMatch, data_root: str | Path) -> tuple[float, str]:
    """Return the ground-truth subject mass and its release metadata source."""

    data_root = Path(data_root)
    if match.dataset == "gait120":
        path = data_root / "Gait120-EMG" / "subject_metadata.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            row = next((value for value in csv.DictReader(handle) if value["Subject"] == match.subject), None)
        if row is None:
            raise ValueError(f"no Gait120 mass for {match.subject} in {path}")
        return float(row["BodyMass (Kg)"]), str(path.resolve())
    if match.dataset == "darmstadt":
        from scipy.io import loadmat

        path = (
            data_root
            / "Darmstadt-Stair-Ambulation"
            / "darmstadt-processed"
            / "FullyProcessed"
            / "Data_fully_processed.mat"
        )
        data = loadmat(path, struct_as_record=False, squeeze_me=True)["data"]
        masses = np.asarray(data.info.subjects_mass, dtype=np.float64).reshape(-1)
        return float(masses[int(match.subject.removeprefix("D")) - 1]), str(path.resolve())
    if match.dataset == "vielemeyer":
        try:
            return _VIELEMEYER_MASS_KG[match.subject], "Vielemeyer release code, anthropometrics.ini"
        except KeyError as exc:
            raise ValueError(f"no Vielemeyer mass for {match.subject}") from exc
    raise ValueError(f"unsupported biomechanics dataset {match.dataset!r}")


def _validate_windows(windows: list[PhaseWindow], motion_duration_s: float) -> list[PhaseWindow]:
    if not windows:
        raise ValueError("motion has no biomechanical comparison phase windows")
    tolerance = SOURCE_CLOCK_BOUNDARY_TOLERANCE_S
    for window in windows:
        if (
            not np.isfinite((window.start_time_s, window.end_time_s)).all()
            or window.start_time_s < -tolerance
            or window.end_time_s <= window.start_time_s
            or window.end_time_s > motion_duration_s + tolerance
        ):
            raise ValueError(
                f"phase window {window.label} [{window.start_time_s:.6g}, {window.end_time_s:.6g}] "
                f"is outside motion [0, {motion_duration_s:.6g}]"
            )
    return windows


def _gait120_windows(sidecar: dict[str, np.ndarray], data_root: Path) -> list[PhaseWindow]:
    from terra.datasets.biomechanics_averages import gait120_cycle_windows, gait120_grf_channels

    target_channels = {label: index for index, label in enumerate(gait120_grf_channels(sidecar))}
    return [
        PhaseWindow(
            position_index=0,
            label=window.source_path.stem,
            side="right",
            start_time_s=window.start_time_s,
            end_time_s=window.end_time_s,
            grf_channel_indices=tuple(target_channels[side] for side in window.complete_grf_sides),
        )
        for window in gait120_cycle_windows(sidecar, data_root=data_root)
    ]


def _darmstadt_windows(
    match: TraceMatch,
    sidecar: dict[str, np.ndarray],
    data_root: Path,
) -> list[PhaseWindow]:
    from scipy.io import loadmat

    from terra.datasets.biomechanics_averages import _darmstadt_stride_map, _mat_field, _mat_trials

    parts = Path(match.motion).parts
    configuration = int(parts[-3].removeprefix("config"))
    trial = int(parts[-2].removeprefix("trial"))
    subject = int(match.subject.removeprefix("D"))
    path = (
        data_root
        / "Darmstadt-Stair-Ambulation"
        / "touchdowns"
        / "Processed"
        / "Touchdowns"
        / f"Touchdowns{subject}.mat"
    )
    touchdowns = loadmat(path, struct_as_record=False, squeeze_me=True)[f"TD_{match.direction}"]
    trial_data = _mat_trials(touchdowns, configuration)[trial - 1]
    crop_start = int(np.asarray(sidecar["source_crop_start_frame"]).reshape(()))
    fps = float(np.asarray(sidecar["source_marker_fps"]).reshape(()))
    base_configuration = {"riser_0.10": 1, "riser_0.17": 2, "riser_0.24": 3}[match.condition]
    windows = []
    for position, (mapped_configuration, side_letter, touchdown_index) in enumerate(
        _darmstadt_stride_map(match.direction, base_configuration)
    ):
        if mapped_configuration != configuration:
            continue
        side = "left" if side_letter == "l" else "right"
        values = _mat_field(trial_data, f"td{side_letter.upper()}").reshape(-1)
        if touchdown_index + 1 >= len(values):
            raise ValueError(f"missing touchdown {touchdown_index + 1} for {match.motion}, {side}")
        start = (round(float(values[touchdown_index])) - 1 - crop_start) / fps
        end = (round(float(values[touchdown_index + 1])) - 1 - crop_start) / fps
        windows.append(PhaseWindow(position, f"stride{position + 1}", side, start, end))
    return windows


def _vielemeyer_event_metadata(match: TraceMatch, data_root: Path) -> dict[str, Any]:
    """Load the release's annotated events, including when only ZIPs are staged."""
    import ezc3d

    subject = int(match.subject.removeprefix("Ref"))
    filename = Path(match.motion).name.removesuffix("_stageii") + ".c3d"
    member = f"Ref_{subject}/{match.motion_type}/{filename}"
    root = data_root / "Vielemeyer-Ramp-Walking"
    path = root / "raw" / member
    if path.is_file():
        source = ezc3d.c3d(str(path))
    else:
        archive = root / f"Ref_{subject}_c3d.zip"
        if not archive.is_file():
            archive = root / f"Ref_{subject}_incomplete_c3d.zip"
        with zipfile.ZipFile(archive) as bundle, tempfile.TemporaryDirectory(prefix="terra-c3d-events-") as temporary:
            # The incomplete release archive keeps the ordinary ``Ref_N/``
            # member prefix; ``_incomplete`` appears only in the ZIP filename.
            # Accept the legacy alternative as a compatibility fallback.
            candidates = (
                member,
                member.replace(f"Ref_{subject}/", f"Ref_{subject}_incomplete/"),
            )
            selected = next((candidate for candidate in candidates if candidate in bundle.namelist()), None)
            if selected is None:
                raise KeyError(f"none of {candidates!r} exists in {archive}")
            extracted = Path(temporary) / filename
            extracted.write_bytes(bundle.read(selected))
            source = ezc3d.c3d(str(extracted))
    events = source["parameters"]["EVENT"]
    return {
        "labels": events["LABELS"]["value"],
        "sides": events["CONTEXTS"]["value"],
        "time_s": np.asarray(events["TIMES"]["value"])[0] * 60.0 + np.asarray(events["TIMES"]["value"])[1],
        "fps": float(source["header"]["points"]["frame_rate"]),
        "first_frame": int(source["header"]["points"]["first_frame"]),
        "frame_count": int(source["data"]["points"].shape[-1]),
    }


def _vielemeyer_windows(sidecar: dict[str, np.ndarray], events: dict[str, Any]) -> list[PhaseWindow]:
    """Match the first two annotated full stances, never a boundary fragment.

    Release normalization rounds events to the marker clock and slices
    [heel_strike:toe_off], excluding the toe-off sample.
    """
    time_s = np.asarray(sidecar["grf_native_time_s"], dtype=np.float64)
    valid = np.asarray(sidecar["grf_platform_valid_native"], dtype=bool)
    assignments: dict[int, str] = {}
    for value in np.asarray(sidecar["grf_platform_assignment"]).reshape(-1):
        match = re.fullmatch(r"platform(\d+):(left|right|inactive)", str(value))
        if match is not None:
            assignments[int(match.group(1)) - 1] = match.group(2)
    fps = events["fps"]
    event_frames = np.rint(np.asarray(events["time_s"]) * fps).astype(int) - events["first_frame"]
    annotated = sorted(zip(event_frames, events["labels"], events["sides"], strict=True))
    contacts = []
    for start, label, declared_side in annotated:
        if label != "Foot Strike" or start < 0:
            continue
        ends = [
            frame for frame, kind, side in annotated if frame > start and kind == "Foot Off" and side == declared_side
        ]
        if not ends or ends[0] >= events["frame_count"]:
            continue
        end = ends[0] - 1
        side = declared_side.casefold()
        start_s, end_s = start / fps, end / fps
        within = (time_s >= start_s) & (time_s <= end_s)
        candidates = [
            (float(np.mean(valid[within, p])), p) for p in range(valid.shape[1]) if assignments.get(p) == side
        ]
        if not candidates or max(candidates)[0] < 0.8:
            raise ValueError(f"annotated Vielemeyer stance lacks complete assigned force data: {side}/{start_s}")
        platform = max(candidates)[1]
        contacts.append((start_s, end_s, platform, side))
    contacts.sort()
    if len(contacts) < 2:
        raise ValueError("Vielemeyer source has fewer than two complete annotated force-platform contacts")
    return [
        PhaseWindow(
            position_index=0,
            label=f"contact{channel + 1}_platform{platform + 1}",
            side=side,
            start_time_s=float(start),
            end_time_s=float(end),
            grf_channel_indices=(channel,),
        )
        for channel, (start, end, platform, side) in enumerate(contacts[:2])
    ]


def phase_windows(match: TraceMatch, data_root: str | Path) -> list[PhaseWindow]:
    """Recover dataset-specific phase windows on the synchronized motion clock."""

    with np.load(match.sidecar_path, allow_pickle=False) as source:
        sidecar = {key: np.asarray(source[key]) for key in source.files}
    motion_time = np.asarray(sidecar["motion_time_s"], dtype=np.float64)
    if match.dataset == "gait120":
        windows = _gait120_windows(sidecar, Path(data_root))
    elif match.dataset == "darmstadt":
        windows = _darmstadt_windows(match, sidecar, Path(data_root))
    elif match.dataset == "vielemeyer":
        windows = _vielemeyer_windows(sidecar, _vielemeyer_event_metadata(match, Path(data_root)))
    else:
        raise ValueError(f"unsupported biomechanics dataset {match.dataset!r}")
    return _validate_windows(windows, float(motion_time[-1]))


def _resample_window(
    time_s: np.ndarray,
    values: np.ndarray,
    window: PhaseWindow,
    phase_percent: np.ndarray,
    *,
    source_boundary: bool = False,
) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    target_time = window.start_time_s + np.asarray(phase_percent, dtype=np.float64) / 100.0 * (
        window.end_time_s - window.start_time_s
    )
    tolerance = (
        SOURCE_CLOCK_BOUNDARY_TOLERANCE_S
        if source_boundary
        else 0.51 * float(np.median(np.diff(time_s)))
        if len(time_s) > 1
        else 0.0
    )
    if target_time[0] < time_s[0] - tolerance or target_time[-1] > time_s[-1] + tolerance:
        raise ValueError(
            f"successful rollout does not cover {window.label}: rollout ends at {time_s[-1]:.6g}s, "
            f"window ends at {target_time[-1]:.6g}s"
        )
    outside = (target_time < time_s[0] - 1e-9) | (target_time > time_s[-1] + 1e-9)
    target_time = np.clip(target_time, time_s[0], time_s[-1])
    flat = values.reshape(len(values), -1)
    sampled = np.stack([np.interp(target_time, time_s, flat[:, column]) for column in range(flat.shape[1])], axis=-1)
    if source_boundary:
        # Missing solver boundary frames are missing observations, not zeros or
        # extrapolated plateaus, and must not compress the remaining phase grid.
        sampled[outside] = np.nan
    return sampled.reshape((len(target_time), *values.shape[1:]))


def _rollout_phase_windows(
    capture: RolloutCapture,
    match: TraceMatch,
    windows: list[PhaseWindow],
) -> list[PhaseWindow]:
    """Describe source windows on the rollout clock without signal-driven shifts.

    Live captures carry exact producer timestamps. The bounded legacy fallback
    only supports old synthetic callers; schema-4 evaluation always supplies the
    verified source clock and samples it directly in ``match_rollout_to_trace``.
    """

    if capture.source_time_s is not None:
        source = np.asarray(capture.source_time_s, dtype=np.float64)
        elapsed = np.asarray(capture.time_s, dtype=np.float64)
        if source.shape != elapsed.shape or len(source) < 2 or not np.isfinite(source).all():
            raise ValueError("rollout source timestamps are missing or misaligned")
        if np.any(np.diff(source) <= 0.0) or np.any(np.diff(elapsed) <= 0.0):
            raise ValueError("rollout source and simulation timestamps must increase strictly")
        # Only bounded producer/source boundary losses are eligible for
        # clipping. A wrong or truncated reference must never be stretched to
        # cover the requested experiment.
        mapped = []
        for window in windows:
            if (
                window.start_time_s < source[0] - SOURCE_CLOCK_BOUNDARY_TOLERANCE_S - 1e-9
                or window.end_time_s > source[-1] + SOURCE_CLOCK_BOUNDARY_TOLERANCE_S + 1e-9
            ):
                raise ValueError(f"reference source clock does not cover {window.label}")
            mapped.append(
                replace(
                    window,
                    start_time_s=float(np.interp(window.start_time_s, source, elapsed)),
                    end_time_s=float(np.interp(window.end_time_s, source, elapsed)),
                )
            )
        return mapped
    if match.dataset != "gait120":
        time_s = np.asarray(capture.time_s, dtype=np.float64)
        if not windows or time_s.ndim != 1 or len(time_s) < 2:
            return list(windows)
        sample_period = float(np.median(np.diff(time_s)))
        start_overrun = max(float(time_s[0] - window.start_time_s) for window in windows)
        end_overrun = max(float(window.end_time_s - time_s[-1]) for window in windows)
        if start_overrun <= 0.0 and end_overrun <= 0.0:
            return list(windows)
        # Marker fitting may retain a source-clock boundary up to the accepted
        # conversion tolerance beyond the retargeted motion, while natural
        # completion records one fewer control transition. Clip only that small,
        # fully explained boundary mismatch; _resample_window still rejects any
        # larger overrun or a genuinely truncated rollout.
        tolerance = SOURCE_CLOCK_BOUNDARY_TOLERANCE_S + 1.01 * sample_period
        minimum_complete_coverage = 1.0 - 1.0 / len(time_s)
        if (
            start_overrun <= tolerance
            and end_overrun <= tolerance
            and capture.coverage + 1e-9 >= minimum_complete_coverage
        ):
            return [
                replace(
                    window,
                    start_time_s=max(window.start_time_s, float(time_s[0])),
                    end_time_s=min(window.end_time_s, float(time_s[-1])),
                )
                for window in windows
            ]
        return list(windows)
    if not windows:
        raise ValueError("Gait120 motion has no source phase windows")
    time_s = np.asarray(capture.time_s, dtype=np.float64)
    if time_s.ndim != 1 or len(time_s) < 2 or not np.isfinite(time_s).all() or np.any(np.diff(time_s) <= 0.0):
        raise ValueError("rollout time must contain at least two finite, strictly increasing samples")
    # A natural trajectory completion records one fewer transition than reference
    # frames.  Do not let the clock projection hide a genuinely truncated rollout.
    minimum_complete_coverage = 1.0 - 1.0 / len(time_s)
    if capture.coverage + 1e-9 < minimum_complete_coverage:
        raise ValueError(
            f"successful Gait120 rollout is incomplete: coverage {capture.coverage:.6g}, "
            f"expected at least {minimum_complete_coverage:.6g}"
        )
    source_start = min(window.start_time_s for window in windows)
    source_end = max(window.end_time_s for window in windows)
    source_duration = source_end - source_start
    rollout_start = float(time_s[0])
    rollout_end = float(time_s[-1])
    if not np.isfinite((source_start, source_end)).all() or source_duration <= 0.0:
        raise ValueError("Gait120 source phase-window clock has non-positive duration")
    if abs(source_duration - (rollout_end - rollout_start)) > 0.065:
        raise ValueError("Gait120 source/rollout duration mismatch; refusing endpoint stretching")
    scale = (rollout_end - rollout_start) / source_duration
    return [
        replace(
            window,
            start_time_s=rollout_start + (window.start_time_s - source_start) * scale,
            end_time_s=rollout_start + (window.end_time_s - source_start) * scale,
        )
        for window in windows
    ]


def completed_phase_windows(
    capture: RolloutCapture,
    match: TraceMatch,
    windows: list[PhaseWindow],
) -> list[tuple[int, PhaseWindow, PhaseWindow]]:
    """Return fully observed gait windows and their rollout-clock mappings.

    The tuple members are the source-window index, the source-clock window, and
    the corresponding rollout-clock window. Eligibility depends only on the
    verified clocks, never on episode success or the simulated signals.
    """

    if not windows:
        return []
    if capture.source_time_s is not None:
        source = np.asarray(capture.source_time_s, dtype=np.float64)
        elapsed = np.asarray(capture.time_s, dtype=np.float64)
        if source.shape != elapsed.shape or len(source) < 2 or not np.isfinite(source).all():
            raise ValueError("rollout source timestamps are missing or misaligned")
        if np.any(np.diff(source) <= 0.0) or np.any(np.diff(elapsed) <= 0.0):
            raise ValueError("rollout source and simulation timestamps must increase strictly")
        # The end tolerance represents producer-boundary trimming at natural
        # completion. An absorbing episode must actually reach the gait end.
        end_tolerance = SOURCE_CLOCK_BOUNDARY_TOLERANCE_S if capture.success else 0.0
        eligible = [
            (index, window)
            for index, window in enumerate(windows)
            if window.start_time_s >= source[0] - SOURCE_CLOCK_BOUNDARY_TOLERANCE_S - 1e-9
            and window.end_time_s <= source[-1] + end_tolerance + 1e-9
        ]
        mapped = _rollout_phase_windows(capture, match, [window for _, window in eligible])
        return [
            (index, window, rollout_window)
            for (index, window), rollout_window in zip(eligible, mapped, strict=True)
        ]

    if capture.success:
        mapped = _rollout_phase_windows(capture, match, windows)
        return [
            (index, window, rollout_window)
            for index, (window, rollout_window) in enumerate(zip(windows, mapped, strict=True))
        ]

    # Legacy captures have no verified producer timestamps. Test each window
    # independently and retain only windows whose rollout-clock interval can be
    # sampled without stretching a truncated episode.
    completed: list[tuple[int, PhaseWindow, PhaseWindow]] = []
    for index, window in enumerate(windows):
        try:
            mapped = _rollout_phase_windows(capture, match, [window])[0]
            _resample_window(capture.time_s, capture.activation, mapped, np.asarray((0.0, 100.0)))
        except ValueError:
            continue
        completed.append((index, window, mapped))
    return completed


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def emg_actuator_indices(
    actuator_names: tuple[str, ...],
    muscle: str,
    side: str,
) -> tuple[int, ...]:
    """Apply the frozen anatomical mapping and fail on a missing model actuator."""

    muscle_key = _key(muscle)
    if muscle_key not in EMG_ACTUATOR_BASES:
        raise ValueError(f"no fixed MyoFullBody actuator mapping for EMG muscle {muscle!r}")
    if side not in {"left", "right"}:
        raise ValueError(f"EMG mapping requires left/right, got {side!r}")
    by_name = {_key(name): index for index, name in enumerate(actuator_names)}
    expected = [f"{base}_{side[0]}" for base in EMG_ACTUATOR_BASES[muscle_key]]
    missing = [name for name in expected if _key(name) not in by_name]
    if missing:
        raise ValueError(f"MyoFullBody model is missing mapped actuator(s): {', '.join(missing)}")
    return tuple(by_name[_key(name)] for name in expected)


def _target_force_frame(
    dataset: str,
    force_world: np.ndarray,
    root_position: np.ndarray,
) -> np.ndarray:
    """Transform simulated world forces to the trace's declared force frame."""

    force_world = np.asarray(force_world, dtype=np.float64)
    if dataset != "darmstadt":
        return force_world
    valid_root = np.asarray(root_position)[np.isfinite(root_position).all(axis=-1)]
    if len(valid_root) < 2:
        raise ValueError("cannot define force frame from fewer than two finite root samples")
    displacement = np.asarray(valid_root[-1, :2] - valid_root[0, :2], dtype=np.float64)
    norm = float(np.linalg.norm(displacement))
    if norm < 1e-4:
        raise ValueError("cannot define Darmstadt forward/left force frame from a stationary rollout window")
    forward = displacement / norm
    left = np.asarray((-forward[1], forward[0]))
    result = np.empty_like(force_world)
    result[..., 0] = force_world[..., 0] * forward[0] + force_world[..., 1] * forward[1]
    result[..., 1] = force_world[..., 0] * left[0] + force_world[..., 1] * left[1]
    result[..., 2] = force_world[..., 2]
    return result


def _selected_grf_channels(window: PhaseWindow, channel_count: int) -> tuple[int, ...]:
    selected = tuple(range(channel_count)) if window.grf_channel_indices is None else window.grf_channel_indices
    if len(set(selected)) != len(selected) or any(channel < 0 or channel >= channel_count for channel in selected):
        raise ValueError(f"phase window {window.label} has invalid GRF channel selection {selected}")
    return selected


def _mapped_grf_side(window: PhaseWindow, declared: str) -> str:
    if declared in {"left", "right", "combined"}:
        return declared
    if declared == "ipsilateral" or window.grf_channel_indices is not None:
        return window.side
    return declared


def match_rollout_to_trace(
    capture: RolloutCapture,
    match: TraceMatch,
    trace: dict[str, np.ndarray],
    windows: list[PhaseWindow],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase-normalize every fully completed window from one rollout."""

    completed = completed_phase_windows(capture, match, windows)
    if not completed:
        raise ValueError("rollout contains no fully completed requested gait window")
    phase = np.asarray(trace["phase_percent"], dtype=np.float64)
    positions = len(np.asarray(trace["stride_positions"]).reshape(-1))
    muscles = [str(value) for value in np.asarray(trace["emg_muscles"]).reshape(-1)]
    emg_sides = [str(value) for value in np.asarray(trace["emg_channel_sides"]).reshape(-1)]
    grf_channels = [str(value) for value in np.asarray(trace["grf_channels"]).reshape(-1)]
    emg_values: list[list[list[np.ndarray]]] = [[[] for _channel in muscles] for _position in range(positions)]
    emg_sources = np.full((positions, len(muscles)), "", dtype="U256")
    grf_values: list[list[list[np.ndarray]]] = [[[] for _channel in grf_channels] for _position in range(positions)]
    source_boundary = capture.source_time_s is not None
    sample_time = capture.source_time_s if source_boundary else capture.time_s
    for _window_index, source_window, rollout_window in completed:
        window = source_window if source_boundary else rollout_window
        activation = _resample_window(sample_time, capture.activation, window, phase, source_boundary=source_boundary)
        force = _resample_window(sample_time, capture.grf_world_n, window, phase, source_boundary=source_boundary)
        root = _resample_window(sample_time, capture.root_position_m, window, phase, source_boundary=source_boundary)
        force = _target_force_frame(match.dataset, force, root)
        for channel, (muscle, declared_side) in enumerate(zip(muscles, emg_sides, strict=True)):
            side = window.side if declared_side == "ipsilateral" else declared_side
            indices = emg_actuator_indices(capture.actuator_names, muscle, side)
            emg_values[window.position_index][channel].append(np.mean(activation[:, indices], axis=-1))
            emg_sources[window.position_index, channel] = "+".join(capture.actuator_names[index] for index in indices)
        for channel in _selected_grf_channels(window, len(grf_channels)):
            declared = grf_channels[channel]
            # Vielemeyer release traces name consecutive contacts rather than
            # anatomical feet.  Their synchronized phase windows carry the
            # force-platform-to-foot assignment recovered from the source
            # sidecar, so that assignment is authoritative for contact-indexed
            # target channels.
            side = _mapped_grf_side(window, declared)
            if side == "combined":
                grf_values[window.position_index][channel].append(np.sum(force, axis=1))
                continue
            if side not in {"left", "right"}:
                raise ValueError(f"cannot map target GRF channel {declared!r} to a simulated foot")
            side_index = ("left", "right").index(side)
            grf_values[window.position_index][channel].append(force[:, side_index])
    emg = np.full((positions, len(phase), len(muscles)), np.nan, dtype=np.float64)
    grf = np.full((positions, len(phase), len(grf_channels), 3), np.nan, dtype=np.float64)
    for position in range(positions):
        for channel in range(len(muscles)):
            if emg_values[position][channel]:
                emg[position, :, channel] = _nan_mean_std(np.stack(emg_values[position][channel]))[0]
        for channel in range(len(grf_channels)):
            if grf_values[position][channel]:
                grf[position, :, channel] = _nan_mean_std(np.stack(grf_values[position][channel]))[0]
    return emg, grf, emg_sources


def _shape_normalize(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full_like(np.asarray(values, dtype=np.float64), np.nan)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if not mask.any():
        return result
    low = float(np.min(np.asarray(values)[mask]))
    high = float(np.max(np.asarray(values)[mask]))
    if high - low <= 1e-12:
        result[mask] = 0.0
    else:
        result[mask] = (np.asarray(values)[mask] - low) / (high - low)
    return result


def _normalize_emg_array(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(values)
    result = np.full_like(values, np.nan)
    for position in range(values.shape[0]):
        for channel in range(values.shape[2]):
            result[position, :, channel] = _shape_normalize(
                values[position, :, channel], np.asarray(valid)[position, :, channel]
            )
    return result


def _nan_mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    total = np.where(finite, values, 0.0).sum(axis=0)
    mean = np.divide(total, count, out=np.full(values.shape[1:], np.nan), where=count > 0)
    residual = np.where(finite, values - mean, 0.0)
    std = np.sqrt(np.divide((residual**2).sum(axis=0), count, out=np.full(values.shape[1:], np.nan), where=count > 0))
    return mean, std, count


def _pearson(first: np.ndarray, second: np.ndarray, valid: np.ndarray) -> float:
    mask = np.asarray(valid, dtype=bool) & np.isfinite(first) & np.isfinite(second)
    if mask.sum() < 3:
        return float("nan")
    first_values = np.asarray(first, dtype=np.float64)[mask]
    second_values = np.asarray(second, dtype=np.float64)[mask]
    if np.std(first_values) <= 1e-12 or np.std(second_values) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(first_values, second_values)[0, 1])


def _json_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _correlation_support(first: np.ndarray, second: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(valid, dtype=bool) & np.isfinite(first) & np.isfinite(second)
    count = int(mask.sum())
    reason = ""
    if count < 3:
        reason = "insufficient_joint_samples"
    elif np.std(np.asarray(second)[mask]) <= 1e-12:
        reason = "constant_experimental_trace"
    elif np.std(np.asarray(first)[mask]) <= 1e-12:
        reason = "constant_policy_trace"
    return {"paired_phase_samples": count, "correlation_undefined_reason": reason}


def _mean_metric(rows: list[dict[str, Any]], signal: str, key: str, *, median: bool = False) -> float | None:
    values = np.asarray(
        [float(row[key]) for row in rows if row["signal"] == signal and row.get(key) not in {None, ""}],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return float(np.median(values) if median else np.mean(values))


def analyze_rollouts(
    captures: list[RolloutCapture],
    match: TraceMatch,
    trace: dict[str, np.ndarray],
    windows: list[PhaseWindow],
    *,
    model_mass_kg: float,
    experimental_mass_kg_value: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Normalize, average, and score every fully completed gait window."""

    successful = [capture for capture in captures if capture.success]
    completed = [
        (capture, window_index, source_window, rollout_window)
        for capture in captures
        for window_index, source_window, rollout_window in completed_phase_windows(capture, match, windows)
    ]
    if not completed:
        raise NoCompletedGaitError(captures)
    if not np.isfinite(model_mass_kg) or model_mass_kg <= 0:
        raise ValueError("model mass must be finite and positive")
    if not np.isfinite(experimental_mass_kg_value) or experimental_mass_kg_value <= 0:
        raise ValueError("experimental mass must be finite and positive")
    measured_capture_ids = {id(capture) for capture, *_rest in completed}
    measured = [capture for capture in captures if id(capture) in measured_capture_ids]
    for capture in measured:
        if not all(
            np.isfinite(values).all()
            for values in (
                capture.activation,
                capture.grf_world_n,
                capture.root_position_m,
            )
        ):
            raise ValueError("measured gait contains non-finite simulation samples")
    # Each completed gait is a separate observation. This gives two completed
    # gaits in one episode twice the weight of one completed gait, independently
    # of whether the episode later reaches natural completion.
    matched = [
        match_rollout_to_trace(capture, match, trace, [source_window])
        for capture, _window_index, source_window, _rollout_window in completed
    ]
    emg_trials_raw = np.stack([value[0] for value in matched])
    grf_trials_n = np.stack([value[1] for value in matched])
    emg_mapping = np.full(np.asarray(matched[0][2]).shape, "", dtype="U256")
    for _emg, _grf, mapping in matched:
        populated = np.asarray(mapping) != ""
        conflict = populated & (emg_mapping != "") & (emg_mapping != mapping)
        if conflict.any():
            raise RuntimeError("EMG actuator mapping changed between completed gaits")
        emg_mapping[populated] = np.asarray(mapping)[populated]
    target_emg_valid = np.asarray(trace["emg_valid"], dtype=bool)
    emg_trials_normalized = np.stack(
        [_normalize_emg_array(value, target_emg_valid & np.isfinite(value)) for value in emg_trials_raw]
    )
    model_body_weight_n = float(model_mass_kg) * STANDARD_GRAVITY_M_S2
    experimental_body_weight_n = float(experimental_mass_kg_value) * STANDARD_GRAVITY_M_S2
    grf_trials_bw = grf_trials_n / model_body_weight_n

    emg_mean_raw, emg_std_raw, emg_count = _nan_mean_std(emg_trials_raw)
    emg_mean, emg_std, _ = _nan_mean_std(emg_trials_normalized)
    grf_mean_n, grf_std_n, grf_count_components = _nan_mean_std(grf_trials_n)
    grf_mean_bw, grf_std_bw, _ = _nan_mean_std(grf_trials_bw)

    target_emg = np.asarray(trace["emg_mean"], dtype=np.float64)
    target_emg_std = np.asarray(trace["emg_std"], dtype=np.float64)
    target_emg_normalized = _normalize_emg_array(target_emg, target_emg_valid)
    target_grf = np.asarray(trace["grf_force_mean"], dtype=np.float64)
    target_grf_std = np.asarray(trace["grf_force_std"], dtype=np.float64)
    force_units = _scalar(trace, "grf_force_units").strip().casefold()
    if force_units == "body weight":
        target_grf_bw = target_grf.copy()
        target_grf_std_bw = target_grf_std.copy()
    elif force_units == "n":
        target_grf_bw = target_grf / experimental_body_weight_n
        target_grf_std_bw = target_grf_std / experimental_body_weight_n
    elif target_grf.size:
        raise ValueError(f"unsupported target GRF units {_scalar(trace, 'grf_force_units')!r}")
    else:
        target_grf_bw = target_grf.copy()
        target_grf_std_bw = target_grf_std.copy()

    phase = np.asarray(trace["phase_percent"], dtype=np.float64)
    positions = [str(value) for value in np.asarray(trace["stride_positions"]).reshape(-1)]
    stride_sides = [str(value) for value in np.asarray(trace["stride_sides"]).reshape(-1)]
    emg_channels = [str(value) for value in np.asarray(trace["emg_channels"]).reshape(-1)]
    muscles = [str(value) for value in np.asarray(trace["emg_muscles"]).reshape(-1)]
    emg_sides = [str(value) for value in np.asarray(trace["emg_channel_sides"]).reshape(-1)]
    grf_channels = [str(value) for value in np.asarray(trace["grf_channels"]).reshape(-1)]
    grf_mapped_sides = np.full((len(positions), len(grf_channels)), "", dtype="U8")
    for window in windows:
        for channel in _selected_grf_channels(window, len(grf_channels)):
            declared = grf_channels[channel]
            side = _mapped_grf_side(window, declared)
            existing = str(grf_mapped_sides[window.position_index, channel])
            if existing and existing != side:
                raise ValueError(
                    f"target GRF slot {positions[window.position_index]}/{declared} maps to both {existing} and {side}"
                )
            grf_mapped_sides[window.position_index, channel] = side
    grf_axes = [str(value) for value in np.asarray(trace["grf_axes"]).reshape(-1)]
    normal_candidates = [index for index, axis in enumerate(grf_axes) if axis in {"z", "terra_z", "up"}]
    if grf_channels and len(normal_candidates) != 1:
        raise ValueError(f"cannot identify one normal/up target GRF axis from {grf_axes}")
    normal_axis = normal_candidates[0] if normal_candidates else 2
    cyclic_peak = any(
        token in _scalar(trace, "phase_definition").casefold() for token in ("gait cycle", "contact time")
    )
    metric_rows: list[dict[str, Any]] = []
    for position, position_label in enumerate(positions):
        for channel, channel_label in enumerate(emg_channels):
            valid = target_emg_valid[position, :, channel] & np.isfinite(emg_mean[position, :, channel])
            if not valid.any():
                continue
            generated = emg_mean[position, :, channel]
            target = target_emg_normalized[position, :, channel]
            valid_indices = np.flatnonzero(valid)
            generated_peak = float(phase[valid_indices[np.argmax(generated[valid])]])
            target_peak = float(phase[valid_indices[np.argmax(target[valid])]])
            peak_error = abs(generated_peak - target_peak)
            if cyclic_peak:
                peak_error = min(peak_error, 100.0 - peak_error)
            side = stride_sides[position] if emg_sides[channel] == "ipsilateral" else emg_sides[channel]
            metric_rows.append(
                {
                    "signal": "emg",
                    "dataset": match.dataset,
                    "subject": match.subject,
                    "motion_type": match.motion_type,
                    "position": position_label,
                    "channel": channel_label,
                    "side": side,
                    "muscle": muscles[channel],
                    "mapped_actuators": str(emg_mapping[position, channel]),
                    "successful_rollouts": len(successful),
                    "measured_rollouts": len(measured),
                    "completed_gaits": int(np.max(emg_count[position, :, channel])),
                    "waveform_correlation": _json_number(_pearson(generated, target, valid)),
                    **_correlation_support(generated, target, valid),
                    "peak_phase_error_percent": peak_error,
                    "peak_phase_error_is_circular": cyclic_peak,
                    "normal_grf_rmse_bw": None,
                    "normal_grf_rmse_percent_body_weight": None,
                    "impulse_error_bw_phase": None,
                    "impulse_relative_error_percent": None,
                }
            )
        for channel, channel_label in enumerate(grf_channels):
            valid = (
                np.asarray(trace["grf_valid"], dtype=bool)[position, :, channel]
                & np.isfinite(grf_mean_bw[position, :, channel, normal_axis])
                & np.isfinite(target_grf_bw[position, :, channel, normal_axis])
            )
            if not valid.any():
                continue
            generated = grf_mean_bw[position, :, channel, normal_axis]
            target = target_grf_bw[position, :, channel, normal_axis]
            difference = generated[valid] - target[valid]
            rmse = float(np.sqrt(np.mean(difference**2)))
            phase_fraction = phase[valid] / 100.0
            generated_impulse = float(np.trapezoid(generated[valid], x=phase_fraction))
            target_impulse = float(np.trapezoid(target[valid], x=phase_fraction))
            impulse_error = abs(generated_impulse - target_impulse)
            relative_impulse = (
                impulse_error / abs(target_impulse) * 100.0 if abs(target_impulse) > 1e-12 else float("nan")
            )
            side = str(grf_mapped_sides[position, channel])
            metric_rows.append(
                {
                    "signal": "grf",
                    "dataset": match.dataset,
                    "subject": match.subject,
                    "motion_type": match.motion_type,
                    "position": position_label,
                    "channel": channel_label,
                    "side": side,
                    "muscle": "",
                    "mapped_actuators": "",
                    "successful_rollouts": len(successful),
                    "measured_rollouts": len(measured),
                    "completed_gaits": int(np.max(grf_count_components[position, :, channel, :])),
                    "waveform_correlation": _json_number(_pearson(generated, target, valid)),
                    **_correlation_support(generated, target, valid),
                    "peak_phase_error_percent": None,
                    "peak_phase_error_is_circular": None,
                    "normal_grf_rmse_bw": rmse,
                    "normal_grf_rmse_percent_body_weight": rmse * 100.0,
                    "impulse_error_bw_phase": impulse_error,
                    "impulse_relative_error_percent": _json_number(relative_impulse),
                }
            )

    arrays = {
        "schema_version": np.array(COMPARISON_SCHEMA_VERSION, dtype=np.int64),
        "motion": np.array(match.motion),
        "dataset": np.array(match.dataset),
        "subject": np.array(match.subject),
        "motion_type": np.array(match.motion_type),
        "phase_percent": phase.astype(np.float32),
        "phase_definition": np.asarray(trace["phase_definition"]),
        "phase_window_labels": np.asarray([window.label for window in windows]),
        "source_phase_window_start_time_s": np.asarray([window.start_time_s for window in windows], dtype=np.float64),
        "source_phase_window_end_time_s": np.asarray([window.end_time_s for window in windows], dtype=np.float64),
        "completed_gait_source_window_index": np.asarray(
            [window_index for _capture, window_index, _source, _rollout in completed], dtype=np.int64
        ),
        "completed_gait_position_index": np.asarray(
            [source.position_index for _capture, _index, source, _rollout in completed], dtype=np.int64
        ),
        "completed_gait_source_start_time_s": np.asarray(
            [source.start_time_s for _capture, _index, source, _rollout in completed], dtype=np.float64
        ),
        "completed_gait_source_end_time_s": np.asarray(
            [source.end_time_s for _capture, _index, source, _rollout in completed], dtype=np.float64
        ),
        "completed_gait_rollout_start_time_s": np.asarray(
            [rollout.start_time_s for _capture, _index, _source, rollout in completed], dtype=np.float64
        ),
        "completed_gait_rollout_end_time_s": np.asarray(
            [rollout.end_time_s for _capture, _index, _source, rollout in completed], dtype=np.float64
        ),
        "stride_positions": np.asarray(trace["stride_positions"]),
        "stride_sides": np.asarray(trace["stride_sides"]),
        "emg_channels": np.asarray(trace["emg_channels"]),
        "emg_muscles": np.asarray(trace["emg_muscles"]),
        "emg_mapped_actuators": emg_mapping,
        "target_emg_raw": target_emg.astype(np.float32),
        "target_emg_std_raw": target_emg_std.astype(np.float32),
        "target_emg_valid": target_emg_valid,
        "target_emg_shape_normalized": target_emg_normalized.astype(np.float32),
        "generated_emg_trials_raw": emg_trials_raw.astype(np.float32),
        "generated_emg_trials_shape_normalized": emg_trials_normalized.astype(np.float32),
        "generated_emg_gaits_raw": emg_trials_raw.astype(np.float32),
        "generated_emg_gaits_shape_normalized": emg_trials_normalized.astype(np.float32),
        "generated_emg_mean_raw": emg_mean_raw.astype(np.float32),
        "generated_emg_std_raw": emg_std_raw.astype(np.float32),
        "generated_emg_mean_shape_normalized": emg_mean.astype(np.float32),
        "generated_emg_std_shape_normalized": emg_std.astype(np.float32),
        "generated_emg_sample_count": emg_count.astype(np.int64),
        "grf_channels": np.asarray(trace["grf_channels"]),
        "grf_axes": np.asarray(trace["grf_axes"]),
        "grf_mapped_sides": grf_mapped_sides,
        "target_grf_raw": target_grf.astype(np.float32),
        "target_grf_std_raw": target_grf_std.astype(np.float32),
        "target_grf_body_weight": target_grf_bw.astype(np.float32),
        "target_grf_std_body_weight": target_grf_std_bw.astype(np.float32),
        "target_grf_valid": np.asarray(trace["grf_valid"], dtype=bool),
        "generated_grf_trials_n": grf_trials_n.astype(np.float32),
        "generated_grf_trials_body_weight": grf_trials_bw.astype(np.float32),
        "generated_grf_gaits_n": grf_trials_n.astype(np.float32),
        "generated_grf_gaits_body_weight": grf_trials_bw.astype(np.float32),
        "generated_grf_mean_n": grf_mean_n.astype(np.float32),
        "generated_grf_std_n": grf_std_n.astype(np.float32),
        "generated_grf_mean_body_weight": grf_mean_bw.astype(np.float32),
        "generated_grf_std_body_weight": grf_std_bw.astype(np.float32),
        "generated_grf_sample_count": np.min(grf_count_components, axis=-1).astype(np.int64),
        "model_mass_kg": np.array(model_mass_kg, dtype=np.float64),
        "experimental_mass_kg": np.array(experimental_mass_kg_value, dtype=np.float64),
        "successful_seed": np.asarray([capture.seed for capture in successful], dtype=np.int64),
        "measured_seed": np.asarray([capture.seed for capture in measured], dtype=np.int64),
        "completed_gait_seed": np.asarray(
            [capture.seed for capture, _index, _source, _rollout in completed], dtype=np.int64
        ),
        "completed_gait_episode_success": np.asarray(
            [capture.success for capture, _index, _source, _rollout in completed], dtype=bool
        ),
    }
    summary = {
        "requested_rollouts": len(captures),
        "successful_rollouts": len(successful),
        "failed_rollouts": len(captures) - len(successful),
        "measured_rollouts": len(measured),
        "unmeasured_rollouts": len(captures) - len(measured),
        "measured_early_terminated_rollouts": sum(not capture.success for capture in measured),
        "completed_gaits": len(completed),
        "completed_gaits_in_early_terminated_rollouts": sum(
            not capture.success for capture, _index, _source, _rollout in completed
        ),
        "emg_metric_traces": sum(row["signal"] == "emg" for row in metric_rows),
        "grf_metric_traces": sum(row["signal"] == "grf" for row in metric_rows),
        "mean_emg_zero_lag_waveform_correlation": _mean_metric(metric_rows, "emg", "waveform_correlation"),
        "median_emg_peak_phase_error_percent": _mean_metric(
            metric_rows, "emg", "peak_phase_error_percent", median=True
        ),
        "mean_normal_grf_waveform_correlation": _mean_metric(metric_rows, "grf", "waveform_correlation"),
        "mean_normal_grf_rmse_bw": _mean_metric(metric_rows, "grf", "normal_grf_rmse_bw"),
        "mean_normal_grf_rmse_percent_body_weight": _mean_metric(
            metric_rows, "grf", "normal_grf_rmse_percent_body_weight"
        ),
        "mean_impulse_error_bw_phase": _mean_metric(metric_rows, "grf", "impulse_error_bw_phase"),
        "mean_impulse_relative_error_percent": _mean_metric(metric_rows, "grf", "impulse_relative_error_percent"),
    }
    return arrays, metric_rows, summary


def _actuator_names(model: Any, mujoco_module: Any) -> tuple[str, ...]:
    names = []
    for index in range(int(model.nu)):
        name = mujoco_module.mj_id2name(model, mujoco_module.mjtObj.mjOBJ_ACTUATOR, index)
        if not name:
            raise ValueError(f"MuJoCo actuator {index} has no name")
        names.append(str(name))
    if len(set(names)) != len(names):
        raise ValueError("MuJoCo actuator names are not unique")
    return tuple(names)


def _activation_snapshot(model: Any, data: Any) -> np.ndarray:
    """Read actuator activation states using their declared state addresses."""

    addresses = np.asarray(model.actuator_actadr, dtype=np.int64).reshape(-1)
    values = np.full(int(model.nu), np.nan, dtype=np.float64)
    activation = np.asarray(data.act, dtype=np.float64)
    active = addresses >= 0
    values[active] = activation[addresses[active]]
    return values


def _body_descends_from(model: Any, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return ancestor == 0


def _foot_contact_layout(model: Any, mujoco_module: Any) -> tuple[dict[int, str], set[int]]:
    foot_bodies: dict[int, str] = {}
    robot_roots: set[int] = set()
    for side, names in (("left", ("calcn_l", "toes_l")), ("right", ("calcn_r", "toes_r"))):
        found = []
        for name in names:
            body = int(mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_BODY, name))
            if body >= 0:
                found.append(body)
                foot_bodies[body] = side
                top = body
                while int(model.body_parentid[top]) > 0:
                    top = int(model.body_parentid[top])
                robot_roots.add(top)
        if not found:
            raise ValueError(f"MyoFullBody model has no {side} calcn/toes body")
    geom_side: dict[int, str] = {}
    for geom in range(int(model.ngeom)):
        body = int(model.geom_bodyid[geom])
        matches = {side for foot_body, side in foot_bodies.items() if _body_descends_from(model, body, foot_body)}
        if len(matches) == 1:
            geom_side[geom] = matches.pop()
    return geom_side, robot_roots


def _foot_grf_snapshot(
    model: Any,
    data: Any,
    mujoco_module: Any,
    geom_side: dict[int, str],
    robot_roots: set[int],
) -> np.ndarray:
    """Sum terrain contact forces acting on each foot in the world frame."""

    result = np.zeros((2, 3), dtype=np.float64)
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        side1 = geom_side.get(geom1)
        side2 = geom_side.get(geom2)
        if (side1 is None) == (side2 is None):
            continue
        foot_is_second = side2 is not None
        side = side2 if foot_is_second else side1
        other_geom = geom1 if foot_is_second else geom2
        other_body = int(model.geom_bodyid[other_geom])
        if any(_body_descends_from(model, other_body, root) for root in robot_roots):
            # Exclude robot self-contact; the comparison is foot--terrain GRF.
            continue
        wrench = np.zeros(6, dtype=np.float64)
        mujoco_module.mj_contactForce(model, data, contact_index, wrench)
        frame_rows = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        force_on_geom2_world = frame_rows.T @ wrench[:3]
        force_on_foot = force_on_geom2_world if foot_is_second else -force_on_geom2_world
        result[("left", "right").index(str(side))] += force_on_foot
    return result


def _capture_rollout(
    env: Any,
    agent_conf: Any,
    train_state: Any,
    policy: Any,
    *,
    seed: int,
    required_duration_s: float,
    reference_source_time_s: np.ndarray,
) -> RolloutCapture:
    """Run one fixed-start episode while retaining muscle states and contact GRFs."""

    import jax
    import jax.numpy as jnp
    import mujoco

    from musclemimic.algorithms.common.inference import initialize_policy_state

    env.th.fixed_start_conf = [0, 0]
    observation = env.reset(key=jax.random.key(seed))
    history = _observation_history(env, agent_conf.config)
    if history is not None:
        observation = history.reset(observation)
    policy_state, policy_reset = initialize_policy_state(agent_conf, observation)
    rng = jax.random.key(seed)
    model = env.model
    data = env.data
    names = _actuator_names(model, mujoco)
    geom_side, robot_roots = _foot_contact_layout(model, mujoco)
    control_dt = float(env.dt)
    trajectory_length = int(env.th.len_trajectory(0))
    reference_source_time_s = np.asarray(reference_source_time_s, dtype=np.float64)
    if reference_source_time_s.shape != (trajectory_length,):
        raise ValueError("reference source clock does not match the loaded trajectory")
    initial_state = env._additional_carry.traj_state
    if int(initial_state.traj_no) != 0 or int(initial_state.subtraj_step_no) != 0:
        raise RuntimeError("biomechanics rollout did not reset to reference frame zero")
    horizon = max(trajectory_length + 2, int(np.ceil(required_duration_s / control_dt)) + 2)
    times = [0.0]
    activations = [_activation_snapshot(model, data)]
    grfs = [_foot_grf_snapshot(model, data, mujoco, geom_side, robot_roots)]
    roots = [np.asarray(data.qpos[:3], dtype=np.float64).copy()]
    qposes = [np.asarray(data.qpos, dtype=np.float64).copy()]
    reference_frames = [0]
    episode_return = 0.0
    episode_length = 0
    absorbing_result = False
    success = False
    for _step in range(horizon):
        rng, action_rng = jax.random.split(rng)
        action, train_state, policy_state = policy(
            train_state,
            observation,
            action_rng,
            policy_state,
            policy_reset,
        )
        observation, reward, absorbing, done, _info = env.step(jnp.atleast_2d(action))
        if history is not None:
            observation = history.step(observation)
        episode_length += 1
        reference_frame = int(_info["subtraj_step_no"])
        if int(_info["traj_no"]) != 0 or reference_frame != reference_frames[-1] + 1:
            raise RuntimeError("biomechanics reference skipped, repeated, or changed trajectory")
        reference_frames.append(reference_frame)
        episode_return += float(np.asarray(reward).item())
        times.append(episode_length * control_dt)
        activations.append(_activation_snapshot(model, data))
        grfs.append(_foot_grf_snapshot(model, data, mujoco, geom_side, robot_roots))
        roots.append(np.asarray(data.qpos[:3], dtype=np.float64).copy())
        qposes.append(np.asarray(data.qpos, dtype=np.float64).copy())
        done_value = bool(np.asarray(done).item())
        absorbing_result = bool(np.asarray(absorbing).item())
        policy_reset = jnp.asarray([done_value], dtype=bool)
        if done_value:
            success = not absorbing_result and reference_frame == trajectory_length - 1
            break
    coverage = min(float(episode_length) / max(trajectory_length, 1), 1.0)
    return RolloutCapture(
        seed=seed,
        success=success,
        absorbing=absorbing_result,
        coverage=coverage,
        return_per_frame=episode_return / max(episode_length, 1),
        time_s=np.asarray(times, dtype=np.float64),
        actuator_names=names,
        activation=np.asarray(activations, dtype=np.float64),
        grf_world_n=np.asarray(grfs, dtype=np.float64),
        root_position_m=np.asarray(roots, dtype=np.float64),
        qpos=np.asarray(qposes, dtype=np.float64),
        source_time_s=reference_source_time_s[np.asarray(reference_frames)],
        reference_frame=np.asarray(reference_frames, dtype=np.int64),
    )


def _observation_history(env: Any, config: Any):
    """Recreate the checkpoint's PPO observation-history buffer, when enabled."""

    from musclemimic.algorithms.ppo.inference import ObservationHistoryBuffer

    history_length = int(config.experiment.get("len_obs_history", 1))
    if history_length <= 1:
        return None
    split_goal = bool(config.experiment.get("split_goal", False))
    state_indices = None
    goal_indices = None
    if split_goal:
        goal_indices = np.asarray(env.obs_container.get_obs_ind_by_group("goal"), dtype=int)
        if goal_indices.size == 0:
            raise ValueError("split_goal=True requires goal observations")
        state_mask = np.ones(int(env.info.observation_space.shape[0]), dtype=bool)
        state_mask[goal_indices] = False
        state_indices = np.arange(state_mask.size, dtype=int)[state_mask]
    return ObservationHistoryBuffer(
        history_length,
        split_goal=split_goal,
        state_indices=state_indices,
        goal_indices=goal_indices,
    )


def _build_cpu_environment(config: Any, motion_paths: list[str]):
    """Build a headless native-MuJoCo environment for the requested motions."""

    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from loco_mujoco.task_factories import TaskFactory
    from terra.rl.hooks import TerraValidationVideoRecorder

    recorder = TerraValidationVideoRecorder(
        video_dir="/tmp/terra-unused-biomechanics-video",
        frequency=1,
        length=1,
        deterministic=True,
    )
    holder = SimpleNamespace(config=config)
    env_params = recorder._build_env_params(holder, "biomechanics_validation")
    env_params["headless"] = True
    env_params["visualize_goal"] = False
    env_params.pop("recorder_params", None)
    goal_params = dict(env_params.get("goal_params", {}))
    goal_params["visualize_goal"] = False
    env_params["goal_params"] = goal_params

    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    validation_dataset = config.experiment.validation.get("amass_dataset_conf", None)
    if validation_dataset is not None:
        task_params["amass_dataset_conf"] = OmegaConf.to_container(validation_dataset, resolve=True)
    dataset_config = dict(task_params["amass_dataset_conf"])
    dataset_config["rel_dataset_path"] = list(motion_paths)
    dataset_config["dataset_group"] = None
    dataset_config["max_motions"] = None
    task_params["amass_dataset_conf"] = dataset_config
    # The checkpoint cache identifies its original cohort, not this per-case
    # override. On shared terrain it otherwise silently returns another motion.
    task_params["trajectory_cache_root"] = ""
    task_params["trajectory_cache_key"] = ""
    task_params.pop("trajectory_handler", None)

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    return factory.make(**env_params, **task_params)


def _verified_reference(env: Any, match: TraceMatch, config: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Check loaded arrays against the requested file and recover its source clock."""

    from dataclasses import asdict

    from loco_mujoco.trajectory import Trajectory
    from loco_mujoco.trajectory.dataclasses import interpolate_trajectories
    from loco_mujoco.trajectory.handler import TrajectoryHandler
    from terra.evaluation.timeline import load_trajectory_timeline

    path = Path(env.paired_motion_path).resolve()
    expected_suffix = Path(f"{match.motion}.npz")
    if not path.as_posix().endswith("/" + expected_suffix.as_posix()):
        raise RuntimeError(f"paired motion path does not match {match.motion}: {path}")
    source = Trajectory.load(path, backend=np)
    expected_data, expected_info = TrajectoryHandler.filter_and_extend(source.data, source.info, env.model)
    if not np.isclose(1.0 / expected_info.frequency, env.dt, atol=1e-9):
        expected_data, expected_info = interpolate_trajectories(expected_data, expected_info, 1.0 / env.dt, backend=np)
    actual = env.th.traj.data
    for field in ("split_points", "qpos", "qvel"):
        expected_values = np.asarray(getattr(expected_data, field))
        actual_values = np.asarray(getattr(actual, field))
        if expected_values.shape != actual_values.shape or not np.allclose(
            expected_values, actual_values, rtol=0.0, atol=1e-6
        ):
            raise RuntimeError(f"loaded reference {field} does not match the requested motion {match.motion}")
    if int(env.th.n_trajectories) != 1:
        raise RuntimeError("biomechanics requires exactly one verified reference trajectory")
    dataset_config = config.experiment.validation.get("amass_dataset_conf", None)
    if dataset_config is None:
        dataset_config = config.experiment.task_factory.params.amass_dataset_conf
    analysis_path = path.with_name(path.stem + "_analysis.npz")
    # The synchronized motion lives beside its sidecar. Its dataset source root
    # can be recovered without relying on host-specific producer paths.
    source_root = match.sidecar_path
    for _part in Path(match.motion).parts:
        source_root = source_root.parent
    timeline = load_trajectory_timeline(path, analysis_path, match.motion, source_root=source_root)
    with np.load(match.sidecar_path, allow_pickle=False) as sidecar:
        sidecar_time = np.asarray(sidecar["motion_time_s"], dtype=np.float64)
        sidecar_motion = str(np.asarray(sidecar["motion"]).reshape(()))
    if sidecar_motion != match.motion or not np.isclose(sidecar_time[-1], timeline.source_end_s, atol=1e-8):
        raise RuntimeError("retarget source timeline disagrees with the experimental motion sidecar")
    if timeline.output_start_s < -1e-9 or timeline.output_end_s > timeline.source_end_s + 1e-9:
        raise RuntimeError("retarget output extends outside the experimental motion")
    source_times = timeline.output_times()
    if len(source_times) != len(actual.qpos):
        source_times = np.linspace(source_times[0], source_times[-1], len(actual.qpos))
    reference = {
        "motion": match.motion,
        "retargeting_method": str(dataset_config.retargeting_method),
        "cache_root": str(Path(dataset_config.cache_root).resolve()),
        "trajectory_path": str(path),
        "trajectory_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "loaded_qpos_sha256": hashlib.sha256(np.asarray(actual.qpos).tobytes()).hexdigest(),
        "loaded_frames": len(actual.qpos),
        "control_dt_s": float(env.dt),
        "materialized_cache_enabled": False,
        "arrays_verified": True,
        "timeline": asdict(timeline),
    }
    return source_times, reference


def _patch_cache_root(config: Any, cache_root: str | Path | None) -> None:
    if cache_root is None:
        return
    value = str(Path(cache_root).expanduser().resolve())
    locations = [config.experiment.task_factory.params.get("amass_dataset_conf", None)]
    locations.append(config.experiment.validation.get("amass_dataset_conf", None))
    for location in locations:
        if location is not None:
            location.cache_root = value


def _raw_rollout_arrays(captures: list[RolloutCapture]) -> dict[str, np.ndarray]:
    if not captures:
        raise ValueError("cannot serialize an empty rollout collection")
    names = captures[0].actuator_names
    if any(capture.actuator_names != names for capture in captures[1:]):
        raise RuntimeError("actuator names changed between rollouts")
    length = max(len(capture.time_s) for capture in captures)
    activation = np.full((len(captures), length, len(names)), np.nan, dtype=np.float32)
    grf = np.full((len(captures), length, 2, 3), np.nan, dtype=np.float32)
    root = np.full((len(captures), length, 3), np.nan, dtype=np.float32)
    time = np.full((len(captures), length), np.nan, dtype=np.float64)
    lengths = np.empty(len(captures), dtype=np.int64)
    for index, capture in enumerate(captures):
        current = len(capture.time_s)
        lengths[index] = current
        time[index, :current] = capture.time_s
        activation[index, :current] = capture.activation
        grf[index, :current] = capture.grf_world_n
        root[index, :current] = capture.root_position_m
    arrays = {
        "schema_version": np.array(COMPARISON_SCHEMA_VERSION, dtype=np.int64),
        "seed": np.asarray([capture.seed for capture in captures], dtype=np.int64),
        "success": np.asarray([capture.success for capture in captures], dtype=bool),
        "absorbing": np.asarray([capture.absorbing for capture in captures], dtype=bool),
        "coverage": np.asarray([capture.coverage for capture in captures], dtype=np.float64),
        "return_per_frame": np.asarray([capture.return_per_frame for capture in captures], dtype=np.float64),
        "sample_count": lengths,
        "time_s": time,
        "actuator_names": np.asarray(names),
        "muscle_activation": activation,
        "grf_world_n": grf,
        "grf_sides": np.asarray(("left", "right")),
        "root_position_m": root,
    }
    if all(capture.qpos is not None for capture in captures):
        nq = {int(np.asarray(capture.qpos).shape[1]) for capture in captures}
        if len(nq) != 1:
            raise RuntimeError("qpos width changed between rollouts")
        qpos = np.full((len(captures), length, nq.pop()), np.nan, dtype=np.float64)
        for index, capture in enumerate(captures):
            values = np.asarray(capture.qpos, dtype=np.float64)
            qpos[index, : len(values)] = values
        arrays["qpos"] = qpos
    if all(capture.source_time_s is not None for capture in captures):
        source_time = np.full((len(captures), length), np.nan, dtype=np.float64)
        reference_frame = np.full((len(captures), length), -1, dtype=np.int64)
        for index, capture in enumerate(captures):
            source_time[index, : len(capture.time_s)] = capture.source_time_s
            reference_frame[index, : len(capture.time_s)] = capture.reference_frame
        arrays["source_time_s"] = source_time
        arrays["reference_frame"] = reference_frame
    return arrays


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "signal",
        "dataset",
        "subject",
        "motion_type",
        "position",
        "channel",
        "side",
        "muscle",
        "mapped_actuators",
        "successful_rollouts",
        "measured_rollouts",
        "completed_gaits",
        "waveform_correlation",
        "paired_phase_samples",
        "correlation_undefined_reason",
        "peak_phase_error_percent",
        "peak_phase_error_is_circular",
        "normal_grf_rmse_bw",
        "normal_grf_rmse_percent_body_weight",
        "impulse_error_bw_phase",
        "impulse_relative_error_percent",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def compare_checkpoint(
    motion: str | Path,
    checkpoint: str | Path,
    *,
    output_dir: str | Path,
    data_root: str | Path,
    artifact_root: str | Path,
    trace_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    trials: int = 10,
    first_seed: int = 0,
    deterministic: bool = False,
    termination_threshold_m: float | None = None,
) -> dict[str, Any]:
    """Execute the complete policy-to-EMG/GRF comparison pipeline."""

    if trials < 1:
        raise ValueError("trials must be positive")
    if first_seed < 0:
        raise ValueError("first_seed must be non-negative")
    if termination_threshold_m is not None and termination_threshold_m <= 0:
        raise ValueError("termination_threshold_m must be positive")
    artifact_root = Path(artifact_root).resolve()
    trace_root = (
        Path(trace_root).resolve() if trace_root is not None else artifact_root / "biomechanics" / "validation-traces"
    )
    target_match = resolve_trace_match(motion, trace_root=trace_root, artifact_root=artifact_root)
    trace = load_trace(target_match.trace_path)
    for field in ("dataset", "subject", "motion_type"):
        if _scalar(trace, field) != getattr(target_match, field):
            raise ValueError(f"experimental trace {field} does not match the requested motion")
    if target_match.motion not in set(map(str, np.asarray(trace["source_motions"]).reshape(-1))):
        raise ValueError("experimental trace does not list the requested source motion")
    windows = phase_windows(target_match, data_root)
    mass_kg, mass_source = experimental_mass_kg(target_match, data_root)

    from omegaconf import OmegaConf, open_dict

    from loco_mujoco.core.stateful_object import StatefulObject
    from musclemimic.algorithms.common.inference import make_policy_action_fn
    from musclemimic.runner.engine import pick_algorithm
    from musclemimic.runner.eval_utils import align_agent_state, load_checkpoint
    from terra.rl import register_components
    from terra.rl.backend import install_backend_integrations
    from terra.rl.validation_render import _checkpoint_timestep, _upgrade_legacy_observation_config

    checkpoint_path = str(Path(checkpoint).expanduser())
    config, raw_agent_state, metadata = load_checkpoint(checkpoint_path)
    OmegaConf.set_struct(config, False)
    _upgrade_legacy_observation_config(config)
    _patch_cache_root(config, cache_root)
    algorithm_cls = pick_algorithm(config)
    register_components()
    install_backend_integrations(str(config.experiment.get("algorithm", algorithm_cls.__name__)))
    with open_dict(config.experiment.validation):
        config.experiment.validation.active = True
        config.experiment.validation.evaluate_all = True
        config.experiment.validation.start_from_beginning = True
        if termination_threshold_m is not None:
            threshold = float(termination_threshold_m)
            terminal = config.experiment.validation.get("terminal_state_params", None)
            if terminal is None:
                config.experiment.validation.terminal_state_params = {}
                terminal = config.experiment.validation.terminal_state_params
            terminal.mean_site_deviation_threshold = threshold
            terminal.core_upper_body_mean_site_deviation_threshold = threshold
            terminal.curriculum_initial_global_threshold = threshold
            terminal.core_upper_body_curriculum_initial_threshold = threshold

    saved_instances = StatefulObject._instances.copy()
    StatefulObject._instances.clear()
    env = None
    captures: list[RolloutCapture] = []
    model_mass = float("nan")
    try:
        env = _build_cpu_environment(config, [target_match.motion])
        if int(env.th.n_trajectories) != 1:
            raise RuntimeError(f"validation environment loaded {env.th.n_trajectories} motions instead of one")
        reference_source_time, reference_provenance = _verified_reference(env, target_match, config)
        env.th.random_start = False
        env.th.use_fixed_start = True
        env.th.start_from_random_step = False
        agent_conf = algorithm_cls.init_agent_conf(env, config)
        agent_state = align_agent_state(raw_agent_state, agent_conf)
        policy = make_policy_action_fn(agent_conf, deterministic=deterministic)
        import mujoco

        model_mass = float(mujoco.mj_getTotalmass(env.model))
        required_duration = max(window.end_time_s for window in windows)
        seeds = list(range(first_seed, first_seed + trials))
        for rollout_index, seed in enumerate(seeds, start=1):
            print(
                f"[BiomechanicsValidation] rollout {rollout_index}/{trials}, seed={seed}, "
                f"deterministic={deterministic}",
                flush=True,
            )
            captures.append(
                _capture_rollout(
                    env,
                    agent_conf,
                    agent_state.train_state,
                    policy,
                    seed=seed,
                    required_duration_s=required_duration,
                    reference_source_time_s=reference_source_time,
                )
            )
    finally:
        if env is not None:
            env.stop()
        StatefulObject._instances = saved_instances

    output = Path(output_dir).expanduser().resolve()
    _write_json(output / "reference.json", reference_provenance)
    _write_npz(output / "rollouts.npz", _raw_rollout_arrays(captures))
    try:
        arrays, rows, metric_summary = analyze_rollouts(
            captures,
            target_match,
            trace,
            windows,
            model_mass_kg=model_mass,
            experimental_mass_kg_value=mass_kg,
        )
    except NoSuccessfulRolloutError as exc:
        exc.reference = reference_provenance
        raise
    _write_npz(output / "comparison_traces.npz", arrays)
    _write_metrics(output / "metrics.csv", rows)
    from terra.biomechanics_figures import render_comparison_figures

    figure_report = render_comparison_figures(
        output / "comparison_traces.npz",
        output / "metrics.csv",
        output_dir=output / "figures",
    )
    report = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_global_timestep": _checkpoint_timestep(metadata),
        "termination_threshold_m": float(
            config.experiment.validation.terminal_state_params.mean_site_deviation_threshold
        ),
        "motion": target_match.motion,
        "dataset": target_match.dataset,
        "subject": target_match.subject,
        "motion_type": target_match.motion_type,
        "reference": reference_provenance,
        "source_sha256": {
            "evaluation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "experimental_trace": hashlib.sha256(target_match.trace_path.read_bytes()).hexdigest(),
            "motion_biomechanics_sidecar": hashlib.sha256(target_match.sidecar_path.read_bytes()).hexdigest(),
        },
        "success_criterion": "non-absorbing natural completion at the verified reference's final frame",
        "measurement_criterion": (
            "every experimental gait window whose full source-clock interval was observed before termination; "
            "episode completion is not required"
        ),
        "experimental_trace": str(target_match.trace_path),
        "motion_biomechanics_sidecar": str(target_match.sidecar_path),
        "policy_mode": "deterministic" if deterministic else "stochastic",
        "seeds": [capture.seed for capture in captures],
        "model_mass_kg": model_mass,
        "experimental_mass_kg": mass_kg,
        "experimental_mass_source": mass_source,
        "phase_windows": [
            {
                "position_index": window.position_index,
                "label": window.label,
                "side": window.side,
                "start_time_s": window.start_time_s,
                "end_time_s": window.end_time_s,
                "grf_channel_indices": (
                    list(window.grf_channel_indices) if window.grf_channel_indices is not None else None
                ),
            }
            for window in windows
        ],
        "normalization": {
            "emg": "per-completed-gait, per-position, per-channel min-max shape normalization before gait averaging",
            "grf": "simulation divided by model body weight; experiment divided by subject body weight unless release already normalized",
            "impulse": "absolute and relative area error over normalized phase (BW x phase fraction)",
            "temporal_alignment": (
                "zero lag using producer source timestamps and recorded reference frame indices; "
                "missing producer/source boundary frames (maximum 35 ms) masked without rescaling phase"
            ),
        },
        "figures": figure_report["figures"],
        "summary": metric_summary,
        "artifacts": {
            "rollouts": str(output / "rollouts.npz"),
            "comparison_traces": str(output / "comparison_traces.npz"),
            "metrics": str(output / "metrics.csv"),
            "figures": figure_report["manifest"],
            "summary": str(output / "summary.json"),
        },
    }
    _write_json(output / "summary.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    from terra.paths import StorageRoots

    repository_root = Path(__file__).resolve().parents[3]
    roots = StorageRoots.from_environment(repository_root)
    parser = argparse.ArgumentParser(
        prog="terra biomechanics compare",
        description="Roll out one policy checkpoint and compare muscle activity and foot GRFs with matched measurements.",
    )
    parser.add_argument("--motion", required=True, help="logical motion id, converted NPZ, or retarget-cache NPZ")
    parser.add_argument("--checkpoint", required=True, type=Path, help="complete PPO checkpoint")
    parser.add_argument("--output-dir", required=True, type=Path, help="comparison artifact directory")
    parser.add_argument("--data-root", type=Path, default=roots.data_root)
    parser.add_argument("--artifact-root", type=Path, default=roots.artifact_root)
    parser.add_argument("--trace-root", type=Path, help="trial-averaged trace registry root")
    parser.add_argument("--cache-root", type=Path, help="override checkpoint retarget-cache root")
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="policy rollouts; every fully completed gait window is averaged",
    )
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="use policy modes instead of stochastic samples")
    parser.add_argument(
        "--termination-threshold-m",
        type=float,
        help="override both global and core upper-body mean site-deviation termination thresholds",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = compare_checkpoint(
        args.motion,
        args.checkpoint,
        output_dir=args.output_dir,
        data_root=args.data_root,
        artifact_root=args.artifact_root,
        trace_root=args.trace_root,
        cache_root=args.cache_root,
        trials=args.trials,
        first_seed=args.first_seed,
        deterministic=args.deterministic,
        termination_threshold_m=args.termination_threshold_m,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"[BiomechanicsValidation] report: {report['artifacts']['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
