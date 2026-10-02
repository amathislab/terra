"""Build swing-foot clearance and route targets from source motion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from terra.collisions import walkable_terrain
from terra.constants import FOOT_LANDMARK_SIDES
from terra.contacts import contact_ramp
from terra.defaults import (
    DEFAULT_CLEARANCE_LOOKAHEAD,
    DEFAULT_SWING_TARGET_RAMP_FRAMES,
)
from terra.terrain import detect_stance_events, joint_surface_offsets


def _smooth_surface(surface: np.ndarray, fps: float, ramp_seconds: float) -> np.ndarray:
    """Smooth per-frame surface heights with a centered moving average.

    Args:
        surface: Raw surface heights with shape ``(T,)``, in meters.
        fps: Source frame rate in frames per second.
        ramp_seconds: Half-width of the smoothing window in seconds. A value
            of zero disables smoothing.

    Returns:
        Smoothed surface heights with shape ``(T,)``. Boundary values are
        extended to keep a constant window size.

    Raises:
        ValueError: If the duration is invalid or ``fps`` is not positive.
    """
    ramp_seconds = float(ramp_seconds)
    if not np.isfinite(ramp_seconds) or ramp_seconds < 0:
        raise ValueError(f"surface ramp duration must be finite and non-negative, got {ramp_seconds}")
    if ramp_seconds == 0:
        return surface
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"surface smoothing needs a positive fps, got {fps}")
    # Conventional half-up rounding avoids Python's surprising 2.5 -> 2 tie-to-even at
    # 50 Hz, while still selecting the nearest representable duration.
    ramp_frames = max(int(np.floor(fps * ramp_seconds + 0.5)), 1)
    window = 2 * ramp_frames + 1
    kernel = np.full(window, 1.0 / window)
    padded = np.pad(surface, ramp_frames, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def source_sole_clearance(
    human_joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
    terrain=None,
    lookahead: float = 0.0,
    surface_ramp_seconds: float = 0.0,
    source_offsets: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Measure source sole clearance above the nearby surface.

    Args:
        human_joints: Source joints with shape ``(T, J, 3)``.
        demo_joints: Names indexing the joint dimension.
        fps: Source frame rate in frames per second.
        terrain: Terrain to measure, or ``None`` for flat ground.
        lookahead: Horizontal reach used to query the nearby surface, in meters.
        surface_ramp_seconds: Smoothing half-width for surface heights, in seconds.
        source_offsets: Optional ankle/toe heights calibrated on flat ground.

    Returns:
        Clearance arrays in meters, keyed by ``"l"`` and ``"r"``.
    """
    return {
        side: c
        for side, (c, _) in _sole_clearance_and_surface(
            human_joints,
            demo_joints,
            fps,
            terrain,
            lookahead,
            surface_ramp_seconds,
            source_offsets=source_offsets,
        ).items()
    }


def _sole_clearance_and_surface(
    human_joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
    terrain=None,
    lookahead: float = 0.0,
    surface_ramp_seconds: float = 0.0,
    source_offsets: Mapping[str, float] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Measure the lowest sole clearance and associated surface height.

    Args:
        human_joints: Source joints with shape ``(T, J, 3)``.
        demo_joints: Names indexing the joint dimension.
        fps: Source frame rate in frames per second.
        terrain: Terrain surface, or ``None`` for flat ground.
        lookahead: Horizontal reach used for surface queries, in meters.
        surface_ramp_seconds: Smoothing half-width for surface heights, in seconds.
        source_offsets: Optional ankle/toe heights calibrated on flat ground.

    Returns:
        Per-side tuples containing source clearance and surface height.
    """
    demo_joints = list(demo_joints)
    probes = [j for j in FOOT_LANDMARK_SIDES if j in demo_joints]
    offsets = (
        joint_surface_offsets(detect_stance_events(human_joints, demo_joints, fps, contact_joints=tuple(probes)))
        if source_offsets is None
        else {str(name): float(value) for name, value in source_offsets.items()}
    )

    ground = walkable_terrain(terrain)
    out = {}
    for side in ("l", "r"):
        clearances, surfaces = [], []
        for joint in probes:
            if FOOT_LANDMARK_SIDES[joint] != side:
                continue
            p = human_joints[:, demo_joints.index(joint)]
            surface = (
                np.zeros(len(p))
                if ground is None
                else np.asarray(ground.height_near(p[:, 0], p[:, 1], lookahead), dtype=float)
            )
            surface = _smooth_surface(surface, fps, surface_ramp_seconds)
            surfaces.append(np.broadcast_to(surface, (len(p),)))
            clearances.append(p[:, 2] - offsets.get(joint, 0.0) - surface)
        if not clearances:
            continue
        c = np.stack(clearances)
        lowest = np.argmin(c, axis=0)
        idx = np.arange(c.shape[1])
        out[side] = (c[lowest, idx], np.stack(surfaces)[lowest, idx])
    return out


def sole_clearance_targets(
    human_joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
    terrain=None,
    lookahead: float = 0.0,
    fraction: float = 1.0,
    cap: float = np.inf,
    min_clearance: float = 0.0,
    contact: dict[str, np.ndarray] | None = None,
    ramp_frames: int = 0,
    surface_ramp_seconds: float = 0.0,
    source_offsets: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Build absolute sole-height targets for swing-foot clearance.

    Args:
        human_joints: Source joints with shape ``(T, J, 3)``.
        demo_joints: Names indexing the joint dimension.
        fps: Source frame rate in frames per second.
        terrain: Terrain surface, or ``None`` for flat ground.
        lookahead: Horizontal reach used for surface queries, in meters.
        fraction: Fraction of source clearance to retain.
        cap: Maximum requested clearance in meters.
        min_clearance: Minimum additional sole height during swing, in meters.
        contact: Per-side Boolean contact arrays. Required when
            ``min_clearance`` is positive.
        ramp_frames: Frames used to ease the minimum clearance at swing boundaries.
        surface_ramp_seconds: Smoothing half-width for surface heights, in seconds.
        source_offsets: Optional ankle/toe heights calibrated on flat ground.

    Returns:
        Per-side absolute sole heights in meters. Inactive frames contain
        negative infinity.

    Raises:
        ValueError: If a required contact schedule is missing.
    """
    swing = {}
    if min_clearance > 0.0:
        if contact is None:
            raise ValueError("min_clearance needs the contact schedule to know what is a swing")
        swing = {side: contact_ramp(~np.asarray(c, dtype=bool), ramp_frames) for side, c in contact.items()}

    out = {}
    pairs = _sole_clearance_and_surface(
        human_joints,
        demo_joints,
        fps,
        terrain,
        lookahead,
        surface_ramp_seconds,
        source_offsets=source_offsets,
    )
    for side, (clearance, surface) in pairs.items():
        margin = np.clip(fraction * clearance, 0.0, cap)
        target = np.where(margin > 0.0, surface + margin, -np.inf)

        if min_clearance > 0.0:
            if side not in swing:
                raise ValueError(f"min_clearance needs a contact schedule for side {side!r}")
            n = len(target)
            ramp = swing[side][:n]
            # Above the source's *own* sole, not above the surface within reach: the latter
            # turns "be above the step" into a riser-sized lift held for as long as the step
            # is in reach.
            out[side] = np.where(
                (ramp > 0.0) & ~np.isfinite(target),
                surface + clearance + min_clearance * ramp,
                target,
            )
        else:
            out[side] = target
    return out


def swing_foot_target_offsets(
    human_joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
    terrain,
    contact: dict[str, np.ndarray],
    lift: float,
    lookahead: float = DEFAULT_CLEARANCE_LOOKAHEAD,
    ramp_frames: int = DEFAULT_SWING_TARGET_RAMP_FRAMES,
    source_offsets: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Build vertical foot-landmark offsets near a tread riser.

    Args:
        human_joints: Source joints with shape ``(T, J, 3)``.
        demo_joints: Names indexing the joint dimension.
        fps: Source frame rate in frames per second.
        terrain: Terrain in the source-motion frame.
        contact: Boolean contact arrays keyed by toe name.
        lift: Maximum landmark translation in meters.
        lookahead: Horizontal reach used to detect nearby risers, in meters.
        ramp_frames: Frames used to ease offsets at route boundaries.
        source_offsets: Optional ankle/toe heights calibrated on flat ground.

    Returns:
        Per-side vertical offset arrays in meters.

    Raises:
        ValueError: If a toe contact schedule is missing.
    """
    n = len(human_joints)
    out = {"l": np.zeros(n), "r": np.zeros(n)}
    if lift <= 0.0 or terrain is None or terrain.is_flat:
        return out

    names = list(demo_joints)
    clear = source_sole_clearance(
        human_joints,
        names,
        fps,
        terrain,
        lookahead=lookahead,
        source_offsets=source_offsets,
    )
    for toe, side in (("L_Toe", "l"), ("R_Toe", "r")):
        if toe not in contact:
            raise ValueError(f"swing target lift needs contact schedule for {toe!r}")
        swing = ~np.asarray(contact[toe], dtype=bool)[:n]
        # A tiny negative is ordinary floating-point noise on a sole exactly at tread
        # height, not evidence that it is beside the face. Keep a micrometre deadband so
        # the route has actually vanished by touchdown.
        beside_riser = np.asarray(clear[side][:n]) < -1e-6
        # Ramp the final activity mask, not only the contact schedule. `beside_riser` is
        # itself a hard terrain-height gate and can flip in the middle of a swing. Ramping
        # before applying it leaves that edge discontinuous and turns the full route cost on
        # in one frame.
        out[side] = float(lift) * contact_ramp(swing & beside_riser, ramp_frames)
    return out
