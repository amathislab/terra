"""Package-owned source loading and timeline construction for metric evaluators."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _scalar(value):
    value = np.asarray(value)
    return value.item() if value.shape == () else value


def _source_from_root(root: Path, motion: str) -> Path:
    root = root.expanduser().resolve()
    relative = Path(motion.removesuffix(".npz").lstrip("/") + ".npz")
    source = (root / relative).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"motion escapes the configured source root: {motion!r}") from exc
    return source


def load_source_motion(
    motion: str,
    analysis_path: Path | None = None,
    *,
    source_root: Path | None = None,
) -> dict[str, object]:
    """Load a named source from the evaluator root or its recorded source path.

    Every current artifact records its original ``source_path``. Dataset evaluation may
    instead provide an explicit source root for relocatable motion collections.
    """

    candidates: list[Path] = []
    if source_root is not None:
        candidates.append(_source_from_root(source_root, motion))

    if analysis_path is not None and analysis_path.is_file():
        with np.load(analysis_path, allow_pickle=False) as analysis:
            if "source_path" in analysis.files:
                candidates.append(Path(str(_scalar(analysis["source_path"]))).expanduser().resolve())

    from terra.smplh import load_smplh_motion

    for candidate in candidates:
        if candidate.is_file():
            return load_smplh_motion(candidate)

    rendered = ", ".join(str(path) for path in candidates) or "no explicit source path"
    raise FileNotFoundError(f"source motion {motion!r} was not found ({rendered})")


def load_trajectory_timeline(
    trajectory_path: Path,
    analysis_path: Path,
    motion: str,
    *,
    source_root: Path | None = None,
):
    """Load a complete timeline or construct one from producer metadata.

    Dataset retargeting records the method's frame rate/count and its central-difference
    trim. The source archive supplies the source clock and the trajectory archive supplies
    the output clock.
    """

    from musclemimic.utils.retarget.benchmark_timeline import MotionTimeline

    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        output_frames = len(trajectory["qpos"])
        output_fps = float(_scalar(trajectory["frequency"]))

    required = set(MotionTimeline.__dataclass_fields__)
    producer_fields = {
        "native_fps",
        "native_frame_count",
        "trim_start_frames",
        "trim_end_frames",
    }
    analysis_values: dict[str, object] = {}
    if analysis_path.is_file():
        with np.load(analysis_path, allow_pickle=False) as analysis:
            fields = (required | producer_fields) & set(analysis.files)
            analysis_values = {key: _scalar(analysis[key]) for key in fields}

    if required <= set(analysis_values):
        timeline = MotionTimeline(**{key: analysis_values[key] for key in required})
    else:
        source = load_source_motion(motion, analysis_path, source_root=source_root)
        source_frames = len(source["pose_aa"])
        source_fps = float(np.asarray(source["fps"]).reshape(()))
        native_fps = float(analysis_values.get("native_fps", source_fps))
        native_frames = int(analysis_values.get("native_frame_count", source_frames - 2))
        trim_start = int(analysis_values.get("trim_start_frames", 1))
        trim_end = int(analysis_values.get("trim_end_frames", 1))
        timeline = MotionTimeline(
            source_start_s=0.0,
            source_end_s=(source_frames - 1) / source_fps,
            source_fps=source_fps,
            source_frame_count=source_frames,
            output_start_s=trim_start / native_fps,
            output_end_s=(trim_start + native_frames - 1) / native_fps,
            output_fps=output_fps,
            output_frame_count=output_frames,
            native_fps=native_fps,
            native_frame_count=native_frames,
            trim_start_frames=trim_start,
            trim_end_frames=trim_end,
        )

    timeline.validate()
    if timeline.output_frame_count != output_frames:
        raise ValueError(
            f"declared output timeline has {timeline.output_frame_count} frames, "
            f"trajectory has {output_frames}: {trajectory_path}"
        )
    if not np.isclose(timeline.output_fps, output_fps, rtol=0.0, atol=1e-6):
        raise ValueError(
            "declared output frequency disagrees with trajectory: "
            f"{timeline.output_fps} vs {output_fps}: {trajectory_path}"
        )
    return timeline


__all__ = ["load_source_motion", "load_trajectory_timeline"]
