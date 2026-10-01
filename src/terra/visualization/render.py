"""Rendering a terrain-retargeted motion so that its failures are visible in the picture.

The default playback camera frames the whole body from ~4 m, at which a foot is a dozen
pixels and a heel dragging along a beam is indistinguishable from a heel clearing it. That
is how a scraping beam crossing survived visual review. Every choice here follows from that:

* **two panes** - the whole body for gait and posture, and a low camera at foot height,
  which is the only view in which support, clearance and slip are legible;
* **a camera that faces the travel direction broadside**, computed per motion. A fixed
  azimuth looks straight down a staircase, where the flight occludes the feet;
* **the verdict burned into the frame** - a per-frame readout of each sole's clearance and a
  coloured band for whichever failure mode is active. The bands consume the unified
  evaluator's per-frame results, so a tinted frame corresponds to an evaluator verdict.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Must precede `import mujoco`: the GL backend is chosen when the module loads, and a later
# setenv leaves `mjr_makeContext` with no platform library. osmesa because it is the only
# backend that initialises on this host - egl raises EGLError, glfw has no display.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
from PIL import Image, ImageDraw

#: Failure mode -> band colour. Ordered by which one to show when several are active: a
#: pose that is geometrically impossible outranks a foot in the wrong place.
BANDS = [
    ("legs interpenetrating", "selfpen", (150, 90, 220)),
    ("through the surface", "pen", (220, 60, 60)),
    ("dragging or scraping", "drag", (235, 140, 40)),
    ("slipping", "slip", (65, 135, 225)),
    ("floating", "float", (235, 205, 60)),
    ("contact gap (>2 mm)", "contact_gap", (245, 225, 145)),
    ("forefoot up", "foreup", (60, 190, 190)),
    ("hindfoot up", "hindup", (95, 205, 155)),
]


def trajectory_paths(
    motion: str,
    model: str = "MyoFullBody",
    method: str = "terra",
    *,
    cache_root: str | Path,
) -> tuple[Path, Path]:
    """Return package-owned trajectory and terrain paths for one artifact set."""

    from terra.artifacts import normalize_motion_name

    relative = normalize_motion_name(motion)
    base = Path(cache_root).expanduser().resolve() / model / method / relative
    trajectory = base.parent / f"{base.name}.npz"
    return trajectory, base.parent / f"{base.name}_terrain.json"


@dataclass
class Flags:
    """Per-frame masks, one per failure mode, plus the live clearance readout."""

    n: int
    masks: dict[str, np.ndarray]
    vert: dict[str, np.ndarray]
    stance: dict[str, np.ndarray]

    @classmethod
    def empty(cls, n: int) -> Flags:
        return cls(
            n,
            {k: np.zeros(n, bool) for _, k, _ in BANDS},
            {"left": np.full(n, np.nan), "right": np.full(n, np.nan)},
            {"left": np.zeros(n, bool), "right": np.zeros(n, bool)},
        )

    @classmethod
    def load(cls, motion: str, score_dir: Path, n: int, *, method: str | None = None) -> Flags:
        """Read one motion's per-frame unified-evaluator verdicts.

        Args:
            motion: AMASS motion name.
            score_dir: Method-specific ``frames/`` directory from
                :mod:`terra.evaluation.cli`.
            n: Exact number of trajectory frames expected by the scored artifact.
            method: Expected cache method. If supplied, annotations for another method are
                rejected.

        Returns:
            Validated masks. Missing or incompatible scores raise; untinted rendering is
            available only by intentionally omitting ``score_dir`` from :func:`render_motion`.
        """
        from terra.evaluation.annotations import (
            AnnotationError,
            validate_frame_archive,
        )

        stem = motion.replace("/", "__")
        frames_path = score_dir / "frames" / f"{stem}.npz"
        if not frames_path.exists():
            raise AnnotationError(f"scored frame annotations are missing: {frames_path}")
        out = cls.empty(n)
        with np.load(frames_path, allow_pickle=False) as frame:
            validate_frame_archive(frame, motion=motion, n_frames=n, method=method)
            clearance = np.asarray(frame["sole_vertical_clearance_m"], dtype=float)
            stance = np.asarray(frame["source_stance"], dtype=bool)
            out.vert["left"], out.vert["right"] = clearance[:, 0], clearance[:, 1]
            out.stance["left"], out.stance["right"] = stance[:, 0], stance[:, 1]
            out.masks["pen"] = np.asarray(frame["environment_penetrating"], dtype=bool) | np.any(
                np.asarray(frame["support_penetrating"], dtype=bool), axis=1
            )
            out.masks["selfpen"] = np.asarray(frame["self_collision_bad"], dtype=bool)
            out.masks["drag"] = np.asarray(frame["swing_clearance_failure"], dtype=bool) | np.asarray(
                frame["swing_scraping"], dtype=bool
            )
            out.masks["slip"] = np.asarray(frame["skating"], dtype=bool) | np.asarray(
                frame["stance_slip_failure"], dtype=bool
            )
            out.masks["float"] = np.any(np.asarray(frame["support_floating"], dtype=bool), axis=1)
            out.masks["contact_gap"] = np.asarray(frame["contact_gap"], dtype=bool)
            out.masks["foreup"] = np.asarray(frame["forefoot_up"], dtype=bool)
            out.masks["hindup"] = np.asarray(frame["hindfoot_up"], dtype=bool)
        return out


def _camera(lookat, distance, elevation, azimuth) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.elevation = elevation
    cam.azimuth = azimuth
    return cam


def broadside_azimuth(pelvis_xy: np.ndarray, fallback: float = 120.0) -> float:
    """Camera azimuth, in degrees, looking across the direction of travel.

    Direction is the net pelvis displacement over the clip rather than a per-frame
    velocity, so a turn in the middle does not swing the camera around.

    Args:
        pelvis_xy: Pelvis position per frame, shape (n, 2), in world coordinates.
        fallback: Returned when the pelvis moves less than 0.3 m in total, where there is
            no travel direction to be broadside to.

    Returns:
        The azimuth to pass to `_camera`.
    """
    delta = pelvis_xy[-1] - pelvis_xy[0]
    if np.linalg.norm(delta) < 0.3:  # standing still: nothing to be broadside to
        return fallback
    return float(np.degrees(np.arctan2(delta[1], delta[0])) + 90.0)


def bounded_render_stride(
    n_frames: int,
    requested_stride: int,
    max_rendered_frames: int | None,
) -> int:
    """Return a positive stride that covers the full clip within a frame cap."""

    if n_frames < 1:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    if requested_stride < 1:
        raise ValueError(f"stride must be positive, got {requested_stride}")
    if max_rendered_frames is None:
        return requested_stride
    if max_rendered_frames < 1:
        raise ValueError(f"max_rendered_frames must be positive or null, got {max_rendered_frames}")
    return max(requested_stride, math.ceil(n_frames / max_rendered_frames))


def _annotate(
    frame: np.ndarray, width: int, i: int, n: int, flags: Flags, caption: str, *, include_closeup: bool = True
) -> None:
    """Draw the caption, the live clearance readout and any active failure band, in place."""
    active = [(label, colour) for label, key, colour in BANDS if flags.masks[key][i]]
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    if active and include_closeup:
        # One band per active mode, stacked down the foot pane's top edge so that two
        # simultaneous failures do not hide each other.
        for k, (label, colour) in enumerate(active):
            y = k * 14
            draw.rectangle([width, y, 2 * width, y + 13], fill=colour)
            draw.text((width + 6, y + 2), label, fill=(20, 20, 20))

    left, right = flags.vert["left"][i], flags.vert["right"][i]
    readout = "  ".join(f"{s} {v * 1000:+5.0f}" for s, v in (("L", left), ("R", right)) if np.isfinite(v))
    expected = "/".join(side for side, name in (("L", "left"), ("R", "right")) if flags.stance[name][i]) or "none"
    # White-on-sky is unreadable in exactly the frames worth reading; back the text.
    draw.rectangle([0, 0, width, 30], fill=(15, 15, 20))
    draw.text((6, 3), caption, fill=(240, 240, 240))
    draw.text((6, 17), f"frame {i:5d}/{n}", fill=(190, 190, 190))
    if readout and include_closeup:
        h = frame.shape[0]
        draw.rectangle([width, h - 16, 2 * width, h], fill=(15, 15, 20))
        draw.text(
            (width + 6, h - 13),
            f"sole over surface (mm): {readout}   expected stance: {expected}",
            fill=(240, 240, 240),
        )
    frame[:] = np.asarray(img)


def render_motion(
    motion: str,
    out_path: str | Path,
    *,
    model: str = "MyoFullBody",
    method: str = "terra",
    terrain_method: str | None = None,
    force_flat: bool = False,
    score_dir: Path | None = None,
    stride: int = 3,
    max_rendered_frames: int | None = None,
    width: int = 480,
    height: int = 360,
    fps: int | None = None,
    caption: str = "",
    elevation: float = -10.0,
    include_closeup: bool = True,
    cache_root: str | Path,
) -> dict:
    """Render one retargeted motion to `out_path`. Returns a small dict of what it did.

    Args:
        motion: AMASS motion name; must already be retargeted.
        out_path: Output `.mp4`.
        model: Environment name, also the cache directory.
        method: Cache subdirectory holding the retargeted motion.
        terrain_method: Optional separate cache namespace holding the terrain metadata.
        force_flat: Ignore every terrain metadata file and render a plain floor. Benchmark Flat
            rows set this because some immutable exploratory caches retain stale terrain files.
        score_dir: Method-specific unified-evaluator quality directory containing
            per-frame verdicts. Without it the video renders untinted.
        stride: Render every Nth trajectory frame.
        max_rendered_frames: If set, increase ``stride`` as needed to cover the
            complete trajectory with at most this many rendered frames. This
            preserves full-duration review of long recordings without creating
            tens of thousands of PNGs.
        width: Width of each rendered pane.
        fps: Output frame rate. Default keeps playback real-time for the stride used.
        caption: Overlaid on the wide pane; the motion name by default.
        height: Height of the rendered frame.
        elevation: Camera elevation for the wide pane, in degrees.
        include_closeup: Append the low terrain-contact pane after the wide pane. The
            default preserves the existing terrain-diagnostic layout.
        cache_root: Explicit published-artifact root.

    Returns:
        A dict with the motion name, the output path, the trajectory and rendered frame
        counts, the output frame rate, the number of terrain boxes and the camera azimuth.

    Raises:
        FileNotFoundError: If the motion has not been retargeted.
    """
    from loco_mujoco.core.terrain import TerrainSpec
    from loco_mujoco.trajectory import Trajectory
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    traj_path, terrain_path = trajectory_paths(motion, model, method, cache_root=cache_root)
    if terrain_method is not None:
        terrain_path = trajectory_paths(motion, model, terrain_method, cache_root=cache_root)[1]
    if not traj_path.exists():
        raise FileNotFoundError(f"No retargeted motion at {traj_path}")
    terrain = (
        TerrainSpec() if force_flat else TerrainSpec.load(str(terrain_path)) if terrain_path.exists() else TerrainSpec()
    )

    env_kwargs = {} if terrain.is_flat else {"terrain_type": "BoxTerrain", "terrain_params": terrain.to_env_params()}
    env = MyoFullBody(**env_kwargs, th_params={"random_start": False, "fixed_start_conf": (0, 0)})
    m = env._model
    data = mujoco.MjData(m)
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    traj = Trajectory.load(str(traj_path))
    qpos = np.asarray(traj.data.qpos)
    n = len(qpos)
    stride = bounded_render_stride(n, stride, max_rendered_frames)
    src_fps = float(traj.info.frequency) if getattr(traj.info, "frequency", None) else 100.0
    fps = fps or max(5, round(src_fps / stride))

    flags = Flags.load(motion, score_dir, n, method=method) if score_dir is not None else Flags.empty(n)

    # The camera needs the pelvis path before rendering starts, so run the kinematics once.
    pelvis_xyz = np.empty((n, 3))
    for i in range(n):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(m, data)
        pelvis_xyz[i] = data.xpos[pelvis]
    azimuth = broadside_azimuth(pelvis_xyz[:, :2])

    renderer = None
    tmp = Path(tempfile.mkdtemp(prefix="render_terrain_"))
    try:
        renderer = mujoco.Renderer(m, height=height, width=width)
        opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(opt)
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        # The environment's skybox is a branded 1024px texture whose lettering sits directly
        # over the body in the wide pane. Off, the background is flat and the silhouette reads.
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False

        idx = range(0, n, stride)
        for k, i in enumerate(idx):
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(m, data)
            cx, cy, cz = pelvis_xyz[i]
            cameras = [_camera([cx, cy, cz - 0.25], 3.6, elevation, azimuth)]
            if include_closeup:
                # The foot pane tracks the *surface under the pelvis*, not a fixed height: on a
                # staircase a camera at z=0.06 is looking at the bottom step by the third tread.
                surface = float(terrain.height_at(cx, cy))
                cameras.append(_camera([cx, cy, surface + 0.08], 1.4, -6, azimuth))

            panes = []
            for cam in cameras:
                renderer.update_scene(data, camera=cam, scene_option=opt)
                panes.append(renderer.render().copy())

            frame = panes[0] if len(panes) == 1 else np.concatenate(panes, axis=1)
            _annotate(
                frame,
                width,
                i,
                n,
                flags,
                caption or motion,
                include_closeup=include_closeup,
            )
            Image.fromarray(frame).save(tmp / f"{k:06d}.png")

        out_path = Path(out_path).absolute()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Encode beside the temporary frames and replace the destination only after ffmpeg
        # succeeds. An interrupted render can therefore never leave a truncated mp4.
        encoded = tmp / "render.mp4"
        # imageio's bundled binary rather than whatever is on PATH: the conda ffmpeg here is
        # built without libx264, so the encode fails after every frame has been rendered.
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            exe = "ffmpeg"
        subprocess.run(
            [
                exe,
                "-y",
                "-v",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(tmp / "%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(encoded),
            ],
            check=True,
        )
        # ``TMPDIR`` may be a fast export volume while the requested report lives in the
        # workspace. ``Path.replace`` is an atomic rename only within one filesystem and
        # raises EXDEV across those mounts after the expensive render has completed. Stage a
        # complete copy beside the destination, then atomically publish it there.
        staged = out_path.with_name(f".{out_path.name}.staging-{os.getpid()}")
        try:
            shutil.copy2(encoded, staged)
            os.replace(staged, out_path)
        finally:
            staged.unlink(missing_ok=True)
        return {
            "motion": motion,
            "out": str(out_path),
            "n_frames": n,
            "n_rendered": len(idx),
            "fps": fps,
            "stride": stride,
            "n_boxes": len(terrain),
            "azimuth": azimuth,
            "n_panes": 2 if include_closeup else 1,
        }
    finally:
        if renderer is not None:
            renderer.close()
        shutil.rmtree(tmp, ignore_errors=True)
