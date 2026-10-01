"""Source-motion fidelity metrics for retargeted trajectories.

The functions in this module deliberately operate on already time-aligned arrays.  Dataset
drivers own source interpolation and MuJoCo forward kinematics; this module owns the two
metric definitions so that the paper report and unit tests use identical formulas.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from terra.evaluation.settings import BENCHMARK_THRESHOLDS, QUALITY_THRESHOLDS
from terra.terrain.stance import (
    DEFAULT_CONTACT_JOINTS,
    DEFAULT_MIN_STANCE_FRAMES,
    detect_stance_events,
)


@dataclass(frozen=True)
class ContactTiming:
    """Duration-weighted contact-confusion statistics over frame/probe samples."""

    true_positive_s: float
    false_positive_s: float
    false_negative_s: float
    precision: float
    recall: float
    f1: float


def _validated_weights(weights: np.ndarray, n_frames: int) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.shape != (n_frames,):
        raise ValueError(f"weights must have shape ({n_frames},), got {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("weights must be finite and non-negative")
    if not np.any(values > 0):
        raise ValueError("at least one sample must have positive weight")
    return values


def pelvis_aligned_mpjpe_mm(
    retargeted: np.ndarray,
    source: np.ndarray,
    weights: np.ndarray,
    *,
    pelvis_index: int = 0,
    include_pelvis: bool = True,
) -> float:
    """Return sample-duration-weighted pelvis-aligned MPJPE in millimetres.

    Each skeleton is expressed relative to its own pelvis on every frame before taking
    Euclidean point errors.  ``include_pelvis=True`` retains the aligned (zero-error) pelvis
    in the joint mean, as required for the paper's named set of 17 correspondences.  The
    non-pelvis value is also useful as an explicit diagnostic.
    """
    retargeted = np.asarray(retargeted, dtype=float)
    source = np.asarray(source, dtype=float)
    if retargeted.shape != source.shape or retargeted.ndim != 3 or retargeted.shape[2] != 3:
        raise ValueError(
            "retargeted and source must have matching shapes (frames, correspondences, 3)"
        )
    if not np.all(np.isfinite(retargeted)) or not np.all(np.isfinite(source)):
        raise ValueError("retargeted and source positions must be finite")
    if pelvis_index < 0 or pelvis_index >= retargeted.shape[1]:
        raise ValueError(f"invalid pelvis index {pelvis_index}")
    weights = _validated_weights(weights, len(retargeted))

    retargeted_local = retargeted - retargeted[:, pelvis_index : pelvis_index + 1]
    source_local = source - source[:, pelvis_index : pelvis_index + 1]
    distances = np.linalg.norm(retargeted_local - source_local, axis=2)
    if not include_pelvis:
        distances = np.delete(distances, pelvis_index, axis=1)
        if distances.shape[1] == 0:
            raise ValueError("cannot exclude the only correspondence")
    per_frame = np.mean(distances, axis=1)
    return float(np.dot(weights, per_frame) / np.sum(weights) * 1000.0)


def kinematic_contact_mask(
    positions: np.ndarray,
    joint_names: Sequence[str],
    fps: float,
    *,
    contact_joints: Sequence[str] = DEFAULT_CONTACT_JOINTS,
    speed_m_s: float = BENCHMARK_THRESHOLDS["source_contact_speed_m_s"],
    min_stance_s: float = QUALITY_THRESHOLDS["min_stance_s"],
) -> np.ndarray:
    """Detect a binary slow-and-low contact mask for each named probe.

    The detector and thresholds are the benchmark's existing kinematic contact convention.
    The minimum duration is expressed in seconds so the same rule applies to source and
    retargeted clocks.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape (frames, joints, 3)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must be finite")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not np.isfinite(min_stance_s) or min_stance_s < 0:
        raise ValueError("min_stance_s must be finite and non-negative")
    names = list(joint_names)
    missing = [joint for joint in contact_joints if joint not in names]
    if missing:
        raise ValueError(f"contact joints absent from positions: {missing}")

    min_frames = max(DEFAULT_MIN_STANCE_FRAMES, int(np.ceil(min_stance_s * fps - 1.0e-12)))
    events = detect_stance_events(
        positions,
        names,
        fps,
        contact_joints=contact_joints,
        speed_ms=speed_m_s,
        min_frames=min_frames,
    )
    mask = np.zeros((len(positions), len(contact_joints)), dtype=bool)
    probe_index = {joint: index for index, joint in enumerate(contact_joints)}
    for event in events:
        mask[event.start : event.end, probe_index[event.joint]] = True
    return mask


def contact_timing_statistics(
    reference: np.ndarray,
    candidate: np.ndarray,
    weights: np.ndarray,
) -> ContactTiming:
    """Return frame/probe contact precision, recall and F1 without shift tolerance."""
    reference = np.asarray(reference, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and candidate must have matching (frames, probes) shapes")
    weights = _validated_weights(weights, len(reference))[:, None]
    true_positive = float(np.sum(weights * (reference & candidate)))
    false_positive = float(np.sum(weights * (~reference & candidate)))
    false_negative = float(np.sum(weights * (reference & ~candidate)))

    predicted = true_positive + false_positive
    relevant = true_positive + false_negative
    precision = true_positive / predicted if predicted > 0 else float(relevant == 0)
    recall = true_positive / relevant if relevant > 0 else float(predicted == 0)
    denominator = 2.0 * true_positive + false_positive + false_negative
    f1 = 2.0 * true_positive / denominator if denominator > 0 else 1.0
    return ContactTiming(
        true_positive_s=true_positive,
        false_positive_s=false_positive,
        false_negative_s=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
    )


__all__ = [
    "ContactTiming",
    "contact_timing_statistics",
    "kinematic_contact_mask",
    "pelvis_aligned_mpjpe_mm",
]
