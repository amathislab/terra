from __future__ import annotations

import numpy as np
import pytest

from terra.figures.geometry import (
    alignment_yaw,
    cell_origin,
    history_alpha_by_frame,
    resolve_frames,
    resolve_highlight_frame,
)
from terra.figures.manifest import CellSpec, LayoutSpec, load_manifest


def _write_manifest(path, cells: str) -> None:
    path.write_text(
        """version = 1
[figure]
name = "test"
[layout]
rows = 2
columns = 2
width = 800
height = 400
"""
        + cells
    )


def test_load_manifest_accepts_explicit_and_fractional_frames(tmp_path):
    path = tmp_path / "figure.toml"
    _write_manifest(
        path,
        """
[[cells]]
row = 0
column = 0
motion = "Study/Subject/Walk"
frames = [10, 20, 30]
[[cells]]
row = 1
column = 1
motion = "Study/Subject/Sit"
fractions = [0.1, 0.5, 0.9]
""",
    )

    manifest = load_manifest(path)

    assert manifest.name == "test"
    assert manifest.cells[0].frames == (10, 20, 30)
    assert manifest.lighting.key_samples == 3
    assert resolve_frames(manifest.cells[1], 101) == (10, 50, 90)


def test_explicit_highlight_can_be_between_ghost_frames(tmp_path):
    path = tmp_path / "figure.toml"
    _write_manifest(
        path,
        """
[[cells]]
row = 0
column = 0
motion = "Study/Subject/Walk"
frames = [10, 20, 30, 40]
highlight_frame = 20
loop_frames = [5, 45]
""",
    )

    cell = load_manifest(path).cells[0]
    frames = resolve_frames(cell, 50)

    assert resolve_highlight_frame(cell, frames) == 20
    assert cell.loop_frames == (5, 45)
    assert history_alpha_by_frame(frames, 20, (0.2, 0.4, 0.6)) == {
        40: 0.2,
        10: 0.4,
        30: 0.6,
    }


def test_camera_lookat_supports_horizontal_pan(tmp_path):
    path = tmp_path / "figure.toml"
    _write_manifest(
        path,
        """
camera_lookat_x = 0.25
camera_lookat_y = -0.5
camera_lookat_z = -1.0
horizon_extent = 120.0
[[cells]]
row = 0
column = 0
motion = "Study/Subject/Walk"
frames = [10]
""",
    )

    layout = load_manifest(path).layout

    assert (layout.camera_lookat_x, layout.camera_lookat_y, layout.camera_lookat_z) == (0.25, -0.5, -1.0)
    assert layout.horizon_extent == 120.0


def test_explicit_highlight_must_be_one_of_the_frames(tmp_path):
    path = tmp_path / "figure.toml"
    _write_manifest(
        path,
        """
[[cells]]
row = 0
column = 0
motion = "Study/Subject/Walk"
frames = [10, 20, 30]
highlight_frame = 40
""",
    )

    with pytest.raises(ValueError, match="highlight_frame must be included"):
        load_manifest(path)


@pytest.mark.parametrize("loop_frames", ("[20]", "[30, 10]", "[11, 40]"))
def test_loop_frames_must_be_ordered_and_encompass_display_frames(tmp_path, loop_frames):
    path = tmp_path / "figure.toml"
    _write_manifest(
        path,
        f"""
[[cells]]
row = 0
column = 0
motion = "Study/Subject/Walk"
frames = [10, 20, 30]
loop_frames = {loop_frames}
""",
    )

    with pytest.raises(ValueError, match="loop_frames"):
        load_manifest(path)


def test_load_manifest_rejects_duplicate_slots(tmp_path):
    path = tmp_path / "figure.toml"
    _write_manifest(
        path,
        """
[[cells]]
row = 0
column = 0
motion = "Study/A"
frames = [1]
[[cells]]
row = 0
column = 0
motion = "Study/B"
frames = [2]
""",
    )

    with pytest.raises(ValueError, match="unique row/column"):
        load_manifest(path)


def test_resolve_frames_rejects_an_out_of_range_explicit_frame():
    cell = CellSpec(0, 0, "Study/A", "A", (0, 10), None, None, None)

    with pytest.raises(ValueError, match="outside"):
        resolve_frames(cell, 10)


def test_alignment_yaw_maps_principal_motion_to_positive_x():
    path = np.column_stack((np.linspace(0, 1, 20), np.linspace(0, 1, 20)))

    assert alignment_yaw(path, (0, 19), None) == pytest.approx(-45.0)


def test_cell_origin_centers_the_declared_grid():
    layout = LayoutSpec(3, 5, 100, 100, (4.2, 2.7), (4.7, 3.3), 0.06, 1, 120, -28, 25, 0.75, 35)

    assert cell_origin(layout, 1, 2) == pytest.approx((0.0, 0.0, 0.0))
    assert cell_origin(layout, 0, 0) == pytest.approx((-9.4, 3.3, 0.0))


def test_explicit_yaw_wins_over_path_alignment():
    path = np.column_stack((np.linspace(0, 1, 20), np.linspace(0, 1, 20)))
    cell = CellSpec(0, 0, "Study/A", "A", (0, 19), None, 12.0, None)

    assert alignment_yaw(path, cell.frames, cell.yaw_degrees) == 12.0


def test_load_manifest_rejects_an_invalid_light_rig(tmp_path):
    path = tmp_path / "figure.toml"
    _write_manifest(
        path,
        """
[lighting]
ambient_strength = 0.21
key_direction = [0.0, 0.0, 0.0]
[[cells]]
row = 0
column = 0
motion = "Study/A"
frames = [1]
""",
    )

    with pytest.raises(ValueError, match="key_direction must be non-zero"):
        load_manifest(path)


def test_load_manifest_accepts_explicit_ambient_strength(tmp_path):
    path = tmp_path / "figure.toml"
    _write_manifest(
        path,
        """
[lighting]
ambient_strength = 0.21
direct_light_scale = 1.2
[[cells]]
row = 0
column = 0
motion = "Study/A"
frames = [1]
""",
    )

    lighting = load_manifest(path).lighting

    assert lighting.ambient_strength == pytest.approx(0.21)
    assert lighting.direct_light_scale == pytest.approx(1.2)
