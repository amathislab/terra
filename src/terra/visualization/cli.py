"""Render cached motions, with optional evaluator failure annotations."""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

from terra._revision import write_git_commit

#: Failure counts, in diagnostic priority order, with the column holding
#: each. `passed` is the headline but says nothing about *what* went wrong.
MODES = [
    ("legs", "selfpen_worst_mm"),
    ("through-surface", "n_penetrating"),
    ("dragging", "n_dragging"),
    ("floating", "n_floating"),
    ("slipping", "n_slipping"),
]


def _configure_render_environment() -> None:
    """Select deterministic headless rendering before MuJoCo is imported."""

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    # Parallelism is across motions. Giving every software rasterizer and BLAS
    # instance its own machine-sized thread pool causes severe oversubscription.
    for variable in (
        "LP_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", "1")


def _worker(job: tuple[str, str, str, int, int | None, int, str, str | None, str, bool]) -> dict:
    _configure_render_environment()
    motion, out_path, caption, stride, max_frames, width, method, score_dir, cache_root, force_flat = job
    from terra.visualization.render import render_motion

    try:
        info = render_motion(
            motion,
            out_path,
            method=method,
            score_dir=Path(score_dir) if score_dir is not None else None,
            force_flat=force_flat,
            stride=stride,
            max_rendered_frames=max_frames,
            width=width,
            height=int(width * 0.75),
            caption=caption,
            cache_root=cache_root,
        )
        info["error"] = ""
        return info
    # Deliberately broad: one motion that will not render must not stop the batch.
    except Exception as exc:
        return {"motion": motion, "out": out_path, "error": f"{type(exc).__name__}: {exc}"[:300]}


def is_playable(path: Path) -> bool:
    """Whether `path` is a video that decodes, not merely a file that exists.

    A render interrupted during the encode leaves an mp4 that every existence check
    accepts and no player opens, and the resume path would skip it forever.

    Args:
        path: Candidate video file.

    Returns:
        True if the container reports a positive duration.
    """
    if not path.exists():
        return False
    try:
        import imageio_ffmpeg

        reader = imageio_ffmpeg.read_frames(str(path))
        meta = next(reader)
        reader.close()
        return float(meta.get("duration", 0)) > 0
    # Any decode failure is the answer to the question being asked.
    except Exception:
        return False


def write_index(
    out_dir: Path,
    rows: list[dict],
    rendered: dict[str, str],
    *,
    force_flat: bool = False,
) -> None:
    """`INDEX.md`: every clip, worst first, with the reasons it failed beside it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_git_commit(out_dir)
    scored = [r for r in rows if r.get("passed") not in (None, "")]

    def rank(r):
        # Sort by number of failing phases, then by the worst self-collision - the mode
        # that makes a pose impossible rather than merely wrong.
        return (-int(r["n_fails"] or 0), -float(r["selfpen_worst_mm"] or 0))

    terrain_classes = {r.get("terrain_class") for r in rows}

    def link_target(relative_path: str) -> str:
        """Encode characters that make a relative Markdown destination ambiguous."""

        return quote(relative_path, safe="/")

    if force_flat:
        title = "Retargeted flat motions"
    elif "flat" in terrain_classes:
        title = "Retargeted motion review"
    else:
        title = "Retargeted non-flat motions"
    scene = (
        "Every clip is rendered on an explicitly plain floor; cached terrain metadata is ignored."
        if force_flat
        else "Each clip uses its cached reconstructed scene."
    )
    lines = [
        f"# {title}",
        "",
        f"{len(rendered)} clips, worst first. {scene} Each is two panes - whole body, and a camera "
        "at foot height tracking the surface under the pelvis - with a coloured band on the "
        "foot pane whenever the scored verdict says that frame is failing, and a live "
        "sole-over-surface readout in millimetres.",
        "",
        "Bands: **purple** legs interpenetrating, **red** through the surface, "
        "**orange** dragging or scraping, **blue** slipping, **yellow** floating, "
        "**teal** forefoot up, **green** hindfoot up.",
        "",
        "| # | clip | source | class | fails | floating | penetrating | slipping | fore-up | hind-up | dragging | scraping | self-pen |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(scored, key=rank):
        rel = rendered.get(r["motion"])
        if not rel:
            continue
        name = r["motion"].split("/")[-1]
        spen = float(r["selfpen_worst_mm"] or 0)
        review_index = int(r["review_index"]) + 1 if r.get("review_index", "").isdigit() else ""
        source_dataset = r.get("source_dataset") or r.get("dataset", "")
        lines.append(
            f"| {review_index} | [{name}]({link_target(rel)}) | {source_dataset} | {r['terrain_class']} "
            f"| {r['n_fails']} | {r['n_floating']} "
            f"| {r['n_penetrating']} | {r['n_slipping']} | {r.get('n_forefoot_up', '')} "
            f"| {r.get('n_hindfoot_up', '')} "
            f"| {r['n_dragging']} | {r.get('n_scraping', '')} | {spen:.0f} mm |"
        )
    unscored = [m for m in rendered if m not in {r["motion"] for r in scored}]
    if unscored:
        lines += ["", "## Rendered but not scored", ""]
        lines += [f"- [{m}]({link_target(rendered[m])})" for m in sorted(unscored)]
    (out_dir / "INDEX.md").write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="terra visualize",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, help="CSV containing a motion column for a collection")
    score_group = p.add_mutually_exclusive_group(required=True)
    score_group.add_argument(
        "--scores",
        type=Path,
        help="matching unified-evaluator quality directory",
    )
    score_group.add_argument(
        "--without-scores",
        action="store_true",
        help="Intentionally render without evaluator annotations or failure tinting.",
    )
    p.add_argument("--out", type=Path, required=True, help="artifact directory for MP4 files and INDEX.md")
    p.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="published retargeting cache containing the selected method",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--class", dest="klass", default=None, help="Only this terrain class")
    p.add_argument(
        "--motion",
        action="append",
        default=None,
        help="Motion ID in the cache; repeatable. Filters --manifest when supplied.",
    )
    p.add_argument("--stride", type=int, default=3, help="Render every Nth trajectory frame")
    p.add_argument(
        "--max-rendered-frames",
        type=int,
        default=None,
        help="Increase stride for long clips so the full motion uses at most this many frames.",
    )
    p.add_argument("--width", type=int, default=480, help="Width of each of the two panes")
    p.add_argument("--redo", action="store_true", help="Re-render clips that already exist")
    p.add_argument(
        "--force-flat",
        action="store_true",
        help="Ignore terrain metadata and render every requested motion on a plain floor.",
    )
    p.add_argument("--index-only", action="store_true", help="Rewrite INDEX.md and stop")
    p.add_argument(
        "--method",
        default="terra",
        help="Cache subdirectory holding the retargeted motions. The same method's unified "
        "CSV and per-frame verdicts are selected so tinting matches the trajectory.",
    )
    args = p.parse_args(argv)
    if args.manifest is None and not args.motion:
        p.error("pass --motion or --manifest")
    if args.workers < 1 or args.stride < 1 or args.width < 1:
        p.error("--workers, --stride, and --width must be positive")
    if args.max_rendered_frames is not None and args.max_rendered_frames < 1:
        p.error("--max-rendered-frames must be positive")

    from terra.paths import StorageRoots

    roots = StorageRoots.from_environment(Path.cwd())
    args.manifest = roots.resolve_input(args.manifest, base=Path.cwd())
    args.cache_root = roots.resolve_artifact(args.cache_root, base=Path.cwd())
    args.out = roots.resolve_artifact(args.out, base=Path.cwd())
    if args.scores is not None:
        args.scores = roots.resolve_artifact(args.scores, base=Path.cwd())
    assert args.cache_root is not None and args.out is not None
    if args.manifest is not None and not args.manifest.is_file():
        p.error(f"manifest does not exist: {args.manifest}")
    _configure_render_environment()

    if args.manifest is not None:
        with args.manifest.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "motion" not in reader.fieldnames:
                p.error(f"manifest has no motion column: {args.manifest}")
            manifest_rows = list(reader)
        selection = f"manifest {args.manifest}"
    else:
        manifest_rows = [{"motion": motion} for motion in args.motion]
        selection = "--motion selection"
    motion_ids = [row.get("motion", "").strip() for row in manifest_rows]
    if not motion_ids or any(not motion for motion in motion_ids):
        p.error(f"{selection} is empty or contains an empty motion ID")
    if len(motion_ids) != len(set(motion_ids)):
        p.error(f"{selection} contains duplicate motion IDs")
    if args.motion is not None:
        args.motion = [motion.strip() for motion in args.motion]
    manifest = {motion: row | {"motion": motion} for motion, row in zip(motion_ids, manifest_rows, strict=True)}

    quality: dict[str, dict[str, str]] = {}
    score_dir: Path | None = None
    if not args.without_scores:
        assert args.scores is not None
        scores = args.scores
        quality_csv = scores / (
            "quality.csv" if args.method == "terra" else f"quality_{args.method.removeprefix('terra_')}.csv"
        )
        score_dir = scores if args.method == "terra" else scores / args.method
        if not quality_csv.is_file():
            raise SystemExit(
                f"unified evaluator scores are missing: {quality_csv}; run terra evaluate or pass --without-scores"
            )
        quality = {row["motion"]: row for row in csv.DictReader(quality_csv.open())}
        if not quality:
            raise SystemExit(f"unified evaluator score sheet is empty: {quality_csv}")

    motions = args.motion or [m for m, r in manifest.items() if not args.klass or r.get("terrain_class") == args.klass]
    unknown = sorted(set(motions) - set(manifest))
    if unknown:
        p.error(f"requested motion is absent from the manifest: {unknown[0]}")

    from terra.visualization.render import trajectory_paths

    ready, missing = [], []
    for m in motions:
        trajectory, _terrain = trajectory_paths(m, method=args.method, cache_root=args.cache_root)
        (ready if trajectory.exists() else missing).append(m)
    if missing:
        print(f"{len(missing)} of {len(motions)} motions are not retargeted yet; rendering the rest.")

    def dest(motion: str) -> Path:
        cls = manifest.get(motion, {}).get("terrain_class", "unclassified")
        return args.out / cls / (motion.replace("/", "__") + ".mp4")

    rendered = {m: str(dest(m).relative_to(args.out)) for m in ready if is_playable(dest(m))}
    # Evaluator rows contain metrics, while the explicit review manifest owns
    # presentation metadata such as terrain class and review order.  Preserve
    # both when a motion has scores instead of replacing the manifest row.
    rows = [manifest[m] | quality.get(m, {"n_fails": "", "passed": ""}) for m in ready]

    if args.index_only:
        args.out.mkdir(parents=True, exist_ok=True)
        write_index(
            args.out,
            [r for r in rows if r.get("n_fails") != ""],
            rendered,
            force_flat=args.force_flat,
        )
        print(f"wrote {args.out / 'INDEX.md'} ({len(rendered)} clips)")
        return 0

    todo = [m for m in ready if args.redo or m not in rendered]
    print(
        f"{len(ready)} motions ready, {len(ready) - len(todo)} already rendered, "
        f"{len(todo)} to render on {args.workers} workers"
    )

    jobs = []
    cache_root = str(args.cache_root)
    for m in todo:
        # A forced render is a replacement, not a second candidate for the same filename.
        # Invalidate the old clip before dispatch so a failed replacement is retried on the
        # next ordinary resume instead of being mistaken for current output.
        if args.redo:
            dest(m).unlink(missing_ok=True)
        row = quality.get(m, {})
        cls = manifest.get(m, {}).get("terrain_class", "?")
        manifest_row = manifest.get(m, {})
        source_dataset = manifest_row.get("source_dataset") or manifest_row.get("dataset", "?")
        review_index = manifest_row.get("review_index", "")
        review_number = f"{int(review_index) + 1}/{len(motions)} " if review_index.isdigit() else ""
        verdict = ""
        if row.get("passed") not in (None, ""):
            verdict = " PASS" if row["passed"] == "1" else f" FAIL x{row['n_fails']}"
        jobs.append(
            (
                m,
                str(dest(m)),
                f"{review_number}{source_dataset} | {m}  [{cls}]{verdict}",
                args.stride,
                args.max_rendered_frames,
                args.width,
                args.method,
                str(score_dir) if score_dir is not None else None,
                cache_root,
                args.force_flat,
            )
        )

    t0 = time.time()
    n_ok = 0
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, j): j[0] for j in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                info = fut.result()
                if not info["error"]:
                    n_ok += 1
                    rendered[info["motion"]] = str(Path(info["out"]).relative_to(args.out))
                print(
                    f"[{i}/{len(jobs)} {(time.time() - t0) / 60:5.1f}m] "
                    f"{'ok  ' if not info['error'] else 'FAIL'} {info['motion']} "
                    f"{info.get('n_rendered', '')} frames {info['error']}",
                    flush=True,
                )

    write_index(
        args.out,
        [r for r in rows if r.get("n_fails") != ""],
        rendered,
        force_flat=args.force_flat,
    )
    print(f"\n{n_ok}/{len(jobs)} rendered in {(time.time() - t0) / 60:.1f} min\n-> {args.out}/  (see INDEX.md)")
    return 0 if n_ok == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
