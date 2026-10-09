from __future__ import annotations

import pytest

from terra.visualization.cli import main as visualize
from terra.visualization.cli import write_index
from terra.visualization.render import bounded_render_stride, trajectory_paths


def test_bounded_render_stride_preserves_short_clip_stride():
    assert bounded_render_stride(250, 3, 1200) == 3


def test_bounded_render_stride_caps_long_clip_while_covering_all_frames():
    stride = bounded_render_stride(13_180, 3, 1200)

    assert stride == 11
    assert len(range(0, 13_180, stride)) <= 1200


@pytest.mark.parametrize(
    ("n_frames", "stride", "cap"),
    ((0, 3, 1200), (100, 0, 1200), (100, 3, 0)),
)
def test_bounded_render_stride_rejects_invalid_values(n_frames, stride, cap):
    with pytest.raises(ValueError):
        bounded_render_stride(n_frames, stride, cap)


def test_trajectory_paths_support_explicit_cache_namespaces(tmp_path):
    trajectory, terrain = trajectory_paths(
        "Study/Subject/Trial",
        method="candidate",
        cache_root=tmp_path,
    )

    assert trajectory == tmp_path / "MyoFullBody/candidate/Study/Subject/Trial.npz"
    assert terrain == tmp_path / "MyoFullBody/candidate/Study/Subject/Trial_terrain.json"


def test_visualize_requires_explicit_inputs():
    with pytest.raises(SystemExit) as error:
        visualize([])

    assert error.value.code == 2


def test_visualize_can_index_an_explicit_unscored_manifest(tmp_path):
    manifest = tmp_path / "cohort.csv"
    manifest.write_text("motion,terrain_class\nStudy/Subject/Trial,ramp\n")
    output = tmp_path / "review"

    assert (
        visualize(
            [
                "--manifest",
                str(manifest),
                "--without-scores",
                "--cache-root",
                str(tmp_path / "cache"),
                "--out",
                str(output),
                "--index-only",
            ]
        )
        == 0
    )
    assert (output / "INDEX.md").is_file()
    assert len((output / "GIT_COMMIT").read_text().strip()) == 40


def test_index_percent_encodes_spaces_in_video_links(tmp_path):
    motion = "Study/Subject/sit down"
    write_index(
        tmp_path,
        [{"motion": motion, "terrain_class": "chair_sit", "passed": ""}],
        {motion: "chair_sit/Study__Subject__sit down.mp4"},
    )

    index = (tmp_path / "INDEX.md").read_text()
    assert "(chair_sit/Study__Subject__sit%20down.mp4)" in index


def test_visualize_index_merges_manifest_metadata_into_scores(tmp_path, monkeypatch):
    motion = "Study/Subject/Trial"
    manifest = tmp_path / "cohort.csv"
    manifest.write_text(f"motion,terrain_class,review_index,source_dataset\n{motion},stairs_up,0,Study\n")
    scores = tmp_path / "scores"
    scores.mkdir()
    scores.joinpath("quality.csv").write_text(
        "motion,passed,n_fails,selfpen_worst_mm,n_floating,n_penetrating,"
        "n_slipping,n_dragging,n_scraping\n"
        f"{motion},0,1,0,0,1,0,0,0\n"
    )
    trajectory, _terrain = trajectory_paths(motion, cache_root=tmp_path / "cache")
    trajectory.parent.mkdir(parents=True)
    trajectory.touch()
    monkeypatch.setattr("terra.visualization.cli.is_playable", lambda _path: True)
    output = tmp_path / "review"

    assert (
        visualize(
            [
                "--manifest",
                str(manifest),
                "--scores",
                str(scores),
                "--cache-root",
                str(tmp_path / "cache"),
                "--out",
                str(output),
                "--index-only",
            ]
        )
        == 0
    )

    index = (output / "INDEX.md").read_text()
    assert "| 1 | [Trial]" in index
    assert "| Study | stairs_up | 1 |" in index


@pytest.mark.parametrize("use_manifest", [False, True])
def test_visualize_selects_a_cached_motion_by_name(tmp_path, use_manifest):
    import imageio_ffmpeg
    import numpy as np

    motion = "upstairs07_poses"
    output = tmp_path / "videos"
    trajectory, _terrain = trajectory_paths(motion, cache_root=tmp_path / "cache")
    trajectory.parent.mkdir(parents=True)
    trajectory.touch()
    video = output / "unclassified" / f"{motion}.mp4"
    video.parent.mkdir(parents=True)
    writer = imageio_ffmpeg.write_frames(str(video), (16, 16), fps=5)
    writer.send(None)
    try:
        for _ in range(5):
            writer.send(np.zeros((16, 16, 3), dtype=np.uint8))
    finally:
        writer.close()
    selection = ["--motion", motion]
    if use_manifest:
        manifest = tmp_path / "motions.csv"
        manifest.write_text(f"motion\n{motion}\nother_motion\n")
        selection += ["--manifest", str(manifest)]

    assert (
        visualize(
            [
                *selection,
                "--without-scores",
                "--cache-root",
                str(tmp_path / "cache"),
                "--out",
                str(output),
                "--index-only",
            ]
        )
        == 0
    )
    index = (output / "INDEX.md").read_text()
    assert f"[{motion}](unclassified/{motion}.mp4)" in index
    assert "other_motion" not in index


@pytest.mark.parametrize("selection", [[], ["--motion", ""], ["--motion", "a", "--motion", "a"]])
def test_visualize_rejects_missing_empty_or_duplicate_selection(tmp_path, selection):
    with pytest.raises(SystemExit) as error:
        visualize(
            [
                *selection,
                "--without-scores",
                "--cache-root",
                str(tmp_path / "cache"),
                "--out",
                str(tmp_path / "videos"),
                "--index-only",
            ]
        )
    assert error.value.code == 2
