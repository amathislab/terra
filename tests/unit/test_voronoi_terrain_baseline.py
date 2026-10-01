"""Contract tests for Voronoi terrain reconstruction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from terra.benchmarking.terrain import VoronoiConfig, fit_voronoi_terrain_from_motion
from terra.benchmarking.terrain.voronoi import (
    _cluster_edges,
    _CollisionTimeline,
    _edge_causes_outside_collision,
    _TerrainEdge,
)

FPS = 20.0
FRAMES = 21
NAMES = ["L_Toe", "R_Toe", "Pelvis", "Blocker"]


def motion(**positions: tuple[float, float, float]) -> np.ndarray:
    defaults = {
        "L_Toe": (-0.1, 0.1, 0.0),
        "R_Toe": (0.1, -0.1, 0.0),
        "Pelvis": (0.0, 0.0, 0.9),
        "Blocker": (10.0, 10.0, 2.0),
    }
    defaults.update(positions)
    out = np.zeros((FRAMES, len(NAMES), 3), dtype=float)
    for name, point in defaults.items():
        out[:, NAMES.index(name)] = point
    return out


def no_collision_points(n_frames: int = FRAMES) -> tuple[np.ndarray, tuple[str, ...]]:
    return np.empty((n_frames, 0, 3), dtype=float), ()


def fit_one(joints: np.ndarray, **kwargs):
    collision, collision_names = no_collision_points(len(joints))
    input_stage = kwargs.pop("input_stage", "post_gmr_g1")
    return fit_voronoi_terrain_from_motion(
        joints,
        NAMES,
        FPS,
        terrain_links=("L_Toe",),
        pelvis_link=None,
        collision_points=collision,
        collision_point_names=collision_names,
        input_stage=input_stage,
        **kwargs,
    )


def height_at(terrain, x: float, y: float) -> float:
    return float(terrain.height_at(x, y))


def nonfinite_motion(value: np.ndarray) -> np.ndarray:
    out = value.copy()
    out[0, 0, 0] = np.nan
    return out


def test_stationary_low_acceleration_contact_builds_square_horizontal_plateau():
    terrain, report = fit_one(motion(L_Toe=(0.0, 0.0, 0.2)))

    assert report["n_contact_events"] == 1
    assert report["n_contact_edges"] == FRAMES
    assert report["n_candidate_cells"] == 100
    assert len(terrain.boxes) == 1
    box = terrain.boxes[0]
    assert box.pos == pytest.approx((0.0, 0.0, 0.1))
    assert box.size == pytest.approx((0.5, 0.5, 0.1))
    assert box.yaw == box.pitch == 0.0
    assert box.name == "terrain_box_voronoi_0000"
    assert height_at(terrain, 0.45, 0.45) == pytest.approx(0.2)
    assert height_at(terrain, 0.55, 0.55) == 0.0


def test_contact_requires_both_velocity_and_acceleration_thresholds():
    joints = motion(L_Toe=(0.0, 0.0, 0.2))
    t = (np.arange(FRAMES) - FRAMES // 2) / FPS
    joints[:, NAMES.index("L_Toe"), 0] = 2.0 * t**2  # zero speed at apex, 4 m/s^2 acceleration
    config = replace(VoronoiConfig(), min_contact_duration_s=0.0)

    rejected, rejected_report = fit_one(joints, config=config)
    accepted, accepted_report = fit_one(
        joints,
        config=replace(config, acceleration_threshold_m_s2=5.0),
    )

    assert rejected.is_flat
    assert rejected_report["n_contact_edges"] == 0
    assert not accepted.is_flat
    assert accepted_report["n_contact_edges"] >= 1


def test_nearby_similar_height_plateaus_merge_to_weighted_mean():
    joints = motion(L_Toe=(-0.4, 0.0, 0.20), R_Toe=(0.4, 0.0, 0.26))
    collision, collision_names = no_collision_points()
    terrain, report = fit_voronoi_terrain_from_motion(
        joints,
        NAMES,
        FPS,
        terrain_links=("L_Toe", "R_Toe"),
        pelvis_link=None,
        collision_points=collision,
        collision_point_names=collision_names,
        input_stage="post_gmr_g1",
    )

    assert report["n_height_clusters"] == 1
    assert report["height_clusters"][0]["height"] == pytest.approx(0.23)
    assert height_at(terrain, -0.35, 0.05) == pytest.approx(0.23)
    assert height_at(terrain, 0.35, 0.05) == pytest.approx(0.23)


def test_height_at_exact_merge_tolerance_does_not_merge():
    tolerance = VoronoiConfig().height_merge_tolerance_m
    joints = motion(L_Toe=(-0.4, 0.0, 0.2), R_Toe=(0.4, 0.0, 0.2 + tolerance))
    collision, collision_names = no_collision_points()
    _, report = fit_voronoi_terrain_from_motion(
        joints,
        NAMES,
        FPS,
        terrain_links=("L_Toe", "R_Toe"),
        pelvis_link=None,
        collision_points=collision,
        collision_point_names=collision_names,
        input_stage="post_gmr_g1",
    )

    assert report["n_height_clusters"] == 2


def test_similar_height_but_distant_plateaus_do_not_merge():
    joints = motion(L_Toe=(-1.2, 0.0, 0.2), R_Toe=(1.2, 0.0, 0.2))
    collision, collision_names = no_collision_points()
    _, report = fit_voronoi_terrain_from_motion(
        joints,
        NAMES,
        FPS,
        terrain_links=("L_Toe", "R_Toe"),
        pelvis_link=None,
        collision_points=collision,
        collision_point_names=collision_names,
        input_stage="post_gmr_g1",
    )

    assert report["n_height_clusters"] == 2


def test_different_height_overlap_uses_nearest_contact_ownership():
    joints = motion(L_Toe=(-0.2, 0.0, 0.2), R_Toe=(0.2, 0.0, 0.4))
    collision, collision_names = no_collision_points()
    terrain, report = fit_voronoi_terrain_from_motion(
        joints,
        NAMES,
        FPS,
        terrain_links=("L_Toe", "R_Toe"),
        pelvis_link=None,
        collision_points=collision,
        collision_point_names=collision_names,
        input_stage="post_gmr_g1",
    )

    assert report["n_height_clusters"] == 2
    assert height_at(terrain, -0.15, 0.05) == pytest.approx(0.2)
    assert height_at(terrain, 0.15, 0.05) == pytest.approx(0.4)


def test_each_contact_edge_keeps_its_subcell_plateau_footprint():
    joints = motion(L_Toe=(0.0, 0.0, 0.2))
    joints[:, NAMES.index("L_Toe"), 0] = np.linspace(0.001, 0.099, FRAMES)

    _, report = fit_one(joints)

    assert report["n_contact_edges"] == FRAMES
    assert report["n_candidate_cells"] == 110


def test_slow_contact_height_drift_becomes_multiple_bounded_terraces():
    frames = 101
    joints = np.repeat(motion(L_Toe=(0.0, 0.0, 0.1))[:1], frames, axis=0)
    time = np.arange(frames) / FPS
    joints[:, NAMES.index("L_Toe"), 0] = 0.10 * time
    joints[:, NAMES.index("L_Toe"), 2] = 0.10 + 0.05 * time
    collision, collision_names = no_collision_points(frames)
    terrain, report = fit_voronoi_terrain_from_motion(
        joints,
        NAMES,
        FPS,
        terrain_links=("L_Toe",),
        pelvis_link=None,
        collision_points=collision,
        collision_point_names=collision_names,
        input_stage="post_gmr_g1",
    )

    assert report["n_contact_edges"] == frames
    assert report["n_height_clusters"] >= 3
    assert all(
        cluster["height_max"] - cluster["height_min"] < VoronoiConfig().height_merge_tolerance_m
        for cluster in report["height_clusters"]
    )
    assert height_at(terrain, -0.45, 0.05) < height_at(terrain, 0.95, 0.05)


def test_nearest_component_merge_is_reflection_equivariant():
    def cluster_heights(y_sign: float):
        edges = [
            _TerrainEdge("A", 0, np.array([0.5, 0.0, 0.0]), 0, 1),
            _TerrainEdge("A", 1, np.array([0.6, -0.8 * y_sign, 0.0]), 0, 2),
            _TerrainEdge("B", 2, np.array([0.65, -0.3 * y_sign, 0.18]), 2, 3),
            _TerrainEdge("E", 3, np.array([0.7, 0.0, 0.09]), 3, 4),
        ]
        return sorted(cluster.height for cluster in _cluster_edges(edges, VoronoiConfig()))

    assert cluster_heights(1.0) == pytest.approx([0.03, 0.18])
    assert cluster_heights(-1.0) == pytest.approx([0.03, 0.18])


def test_global_nearest_merge_is_independent_of_axis_reflection():
    def cluster_heights(x_sign: float):
        edges = [
            _TerrainEdge("A", 0, np.array([0.0 * x_sign, 0.0, 0.0]), 0, 1),
            _TerrainEdge("B", 1, np.array([0.4 * x_sign, 0.0, 0.09]), 1, 2),
            _TerrainEdge("C", 2, np.array([1.0 * x_sign, 0.0, 0.18]), 2, 3),
        ]
        return sorted(cluster.height for cluster in _cluster_edges(edges, VoronoiConfig()))

    assert cluster_heights(1.0) == pytest.approx([0.045, 0.18])
    assert cluster_heights(-1.0) == pytest.approx([0.045, 0.18])


def test_equal_distance_tie_break_is_independent_of_axis_reflection():
    def cluster_heights(x_sign: float):
        edges = [
            _TerrainEdge("A", 0, np.array([0.0 * x_sign, 0.0, 0.0]), 0, 1),
            _TerrainEdge("B", 1, np.array([0.5 * x_sign, 0.0, 0.09]), 1, 2),
            _TerrainEdge("C", 2, np.array([1.0 * x_sign, 0.0, 0.18]), 2, 3),
        ]
        return sorted(cluster.height for cluster in _cluster_edges(edges, VoronoiConfig()))

    assert cluster_heights(1.0) == pytest.approx([0.045, 0.18])
    assert cluster_heights(-1.0) == pytest.approx([0.045, 0.18])


def test_equal_distance_tie_break_is_independent_of_time_reversal():
    points = (
        (0.0, 0.0, 0.09),
        (0.0, 0.5, 0.09),
        (0.0, 1.0, 0.0),
        (0.5, 0.0, 0.09),
        (0.5, 0.5, 0.18),
    )

    def cluster_heights(reverse: bool):
        edges = [
            _TerrainEdge("L_Toe", len(points) - 1 - index if reverse else index, np.array(point), 0, len(points))
            for index, point in enumerate(points)
        ]
        config = replace(VoronoiConfig(), spatial_merge_distance_m=0.71)
        return sorted((cluster.height, len(cluster.edges)) for cluster in _cluster_edges(edges, config))

    assert cluster_heights(False) == pytest.approx(cluster_heights(True))


def test_dense_equal_signature_ties_are_reflection_invariant():
    points = (
        (0.0, 0.0, 0.18),
        (0.0, 0.5, 0.09),
        (0.5, 0.0, 0.0),
        (0.5, 1.0, 0.09),
        (1.0, 0.5, 0.09),
    )

    def cluster_heights(x_sign: float):
        edges = [
            _TerrainEdge("L_Toe", index, np.array([x_sign * x, y, z]), 0, len(points))
            for index, (x, y, z) in enumerate(points)
        ]
        config = replace(VoronoiConfig(), spatial_merge_distance_m=0.71)
        return sorted((cluster.height, len(cluster.edges)) for cluster in _cluster_edges(edges, config))

    assert cluster_heights(1.0) == pytest.approx(cluster_heights(-1.0))


def test_collision_carving_preserves_an_interior_hole():
    joints = motion(L_Toe=(0.0, 0.0, 0.2))
    collision = np.full((FRAMES, 1, 3), (10.0, 10.0, 2.0), dtype=float)
    collision[FRAMES // 2, 0] = (0.05, 0.05, 0.05)
    terrain, report = fit_voronoi_terrain_from_motion(
        joints,
        NAMES,
        FPS,
        terrain_links=("L_Toe",),
        pelvis_link=None,
        collision_points=collision,
        collision_point_names=("Blocker",),
        input_stage="post_gmr_g1",
    )

    assert report["n_cells_carved_to_floor"] > 0
    assert len(terrain.boxes) > 1
    assert height_at(terrain, 0.05, 0.05) == 0.0
    assert height_at(terrain, 0.35, 0.35) == pytest.approx(0.2)
    assert float(terrain.penetration(np.array([[0.05, 0.05, 0.05]]))[0]) == 0.0


def test_interaction_edge_pruning_precedes_plateau_reconstruction():
    edge = _TerrainEdge("L_Toe", 12, np.array([0.05, 0.05, 0.2]), 10, 15)
    collision = _CollisionTimeline(
        frames=np.array([0]),
        prefix_min_z=np.array([0.05]),
        suffix_min_z=np.array([0.05]),
    )
    config = VoronoiConfig(plateau_side_m=1.0)

    # A collision elsewhere in the future 1 m plateau cannot prune the graph edge:
    # SceneBot creates and prunes the interaction graph before it reconstructs terrain.
    assert not _edge_causes_outside_collision(edge, {(4, 4): collision}, config)
    assert _edge_causes_outside_collision(edge, {(0, 0): collision}, config)


@pytest.mark.parametrize("blocker_z", [0.20, 0.25])
def test_point_on_or_above_plateau_does_not_carve(blocker_z):
    joints = motion(L_Toe=(0.0, 0.0, 0.2))
    collision = np.full((FRAMES, 1, 3), (10.0, 10.0, 2.0), dtype=float)
    collision[FRAMES // 2, 0] = (0.05, 0.05, blocker_z)
    terrain, report = fit_voronoi_terrain_from_motion(
        joints,
        NAMES,
        FPS,
        terrain_links=("L_Toe",),
        pelvis_link=None,
        collision_points=collision,
        collision_point_names=("Blocker",),
        input_stage="post_gmr_g1",
    )

    assert report["n_cells_carved_to_floor"] == 0
    assert height_at(terrain, 0.05, 0.05) == pytest.approx(0.2)


def test_collision_carving_can_reveal_a_lower_overlapping_plateau():
    joints = motion(L_Toe=(0.0, 0.0, 0.4), R_Toe=(0.25, 0.0, 0.2))
    collision = np.full((FRAMES, 1, 3), (10.0, 10.0, 2.0), dtype=float)
    collision[FRAMES // 2, 0] = (0.05, 0.05, 0.25)
    terrain, report = fit_voronoi_terrain_from_motion(
        joints,
        NAMES,
        FPS,
        terrain_links=("L_Toe", "R_Toe"),
        pelvis_link=None,
        collision_points=collision,
        collision_point_names=("Blocker",),
        input_stage="post_gmr_g1",
    )

    assert report["n_cells_lowered"] > 0
    assert height_at(terrain, 0.05, 0.05) == pytest.approx(0.2)


def test_pelvis_guard_suppresses_standing_but_keeps_separated_contact():
    collision, collision_names = no_collision_points()
    standing, standing_report = fit_voronoi_terrain_from_motion(
        motion(Pelvis=(0.0, 0.0, 0.4)),
        NAMES,
        FPS,
        terrain_links=("Pelvis",),
        collision_points=collision,
        collision_point_names=collision_names,
        input_stage="post_gmr_g1",
    )
    seated, seated_report = fit_voronoi_terrain_from_motion(
        motion(Pelvis=(0.5, 0.0, 0.4)),
        NAMES,
        FPS,
        terrain_links=("Pelvis",),
        collision_points=collision,
        collision_point_names=collision_names,
        input_stage="post_gmr_g1",
    )

    assert standing.is_flat
    assert standing_report["contact_detection"]["Pelvis"]["pelvis_frames_suppressed"] == FRAMES
    assert not seated.is_flat
    assert seated_report["contact_detection"]["Pelvis"]["accepted_frames"] == FRAMES


def test_pelvis_guard_uses_foot_midpoint_for_wide_stance():
    joints = motion(L_Toe=(-0.3, 0.0, 0.0), R_Toe=(0.3, 0.0, 0.0), Pelvis=(0.0, 0.0, 0.9))
    collision, collision_names = no_collision_points()
    terrain, report = fit_voronoi_terrain_from_motion(
        joints,
        NAMES,
        FPS,
        terrain_links=("Pelvis",),
        collision_points=collision,
        collision_point_names=collision_names,
        input_stage="post_gmr_g1",
    )

    assert terrain.is_flat
    assert report["contact_detection"]["Pelvis"]["pelvis_frames_suppressed"] == FRAMES


def test_shared_height_rule_changes_only_voronoi_detected_pelvis_heights():
    joints = motion(Pelvis=(0.5, 0.0, 0.8))
    collision, collision_names = no_collision_points()
    observed = []

    def resolve(intervals):
        observed.append(intervals)
        return [0.42]

    terrain, report = fit_voronoi_terrain_from_motion(
        joints,
        NAMES,
        FPS,
        terrain_links=("Pelvis",),
        pelvis_link="Pelvis",
        collision_points=collision,
        collision_point_names=collision_names,
        pelvis_support_height_resolver=resolve,
        pelvis_support_height_source="shared_posed_posterior_body_surface",
        input_stage="fitted_smplh_landmarks",
    )

    assert observed == [((0, FRAMES),)]
    assert report["contact_detection"]["Pelvis"]["accepted_runs"] == 1
    assert report["support_intervals"] == [
        {
            "link": "Pelvis",
            "start": 0,
            "end": FRAMES,
            "kind": "pelvis",
            "evidence_stage": "accepted_before_edge_pruning",
            "surface_height_m": pytest.approx(0.42),
            "height_source": "shared_posed_posterior_body_surface",
        }
    ]
    assert report["resolved_pelvis_support_heights_m"] == pytest.approx((0.42,))
    assert height_at(terrain, 0.5, 0.0) == pytest.approx(0.42)


def test_shared_height_rule_rejects_a_height_count_mismatch():
    collision, collision_names = no_collision_points()

    with pytest.raises(ValueError, match="one height per Voronoi-detected interval"):
        fit_voronoi_terrain_from_motion(
            motion(Pelvis=(0.5, 0.0, 0.8)),
            NAMES,
            FPS,
            terrain_links=("Pelvis",),
            pelvis_link="Pelvis",
            collision_points=collision,
            collision_point_names=collision_names,
            pelvis_support_height_resolver=lambda _intervals: [],
        )


def test_moving_link_with_no_candidate_contact_returns_flat_terrain():
    joints = motion(L_Toe=(0.0, 0.0, 0.2))
    joints[:, NAMES.index("L_Toe"), 0] = np.arange(FRAMES) / FPS

    terrain, report = fit_one(joints)

    assert terrain.is_flat
    assert report["n_contact_events"] == 0
    assert report["n_boxes"] == 0


def test_permuting_joint_and_link_order_preserves_geometry():
    joints = motion(L_Toe=(-0.3, 0.0, 0.2), R_Toe=(0.3, 0.0, 0.4))
    collision, collision_names = no_collision_points()
    first, _ = fit_voronoi_terrain_from_motion(
        joints,
        NAMES,
        FPS,
        terrain_links=("L_Toe", "R_Toe"),
        pelvis_link=None,
        collision_points=collision,
        collision_point_names=collision_names,
        input_stage="post_gmr_g1",
    )
    permutation = [2, 1, 3, 0]
    names = [NAMES[index] for index in permutation]
    second, _ = fit_voronoi_terrain_from_motion(
        joints[:, permutation],
        names,
        FPS,
        terrain_links=("R_Toe", "L_Toe"),
        pelvis_link=None,
        collision_points=collision,
        collision_point_names=collision_names,
        input_stage="post_gmr_g1",
    )

    assert first.boxes == second.boxes


def test_report_describes_voronoi_and_kinematic_contacts():
    terrain, report = fit_one(motion(L_Toe=(0.0, 0.0, 0.2)), input_stage="fitted_smplh_landmarks")

    assert report["model"] == "voronoi"
    assert report["contact_source"] == "kinematic"
    assert "acceleration_threshold_m_s2" in report["algorithm_details"]
    assert "grid_size_m" in report["voronoi_parameters"]
    assert terrain.provenance["method"] == "Voronoi"
    assert terrain.provenance["input_stage"] == "fitted_smplh_landmarks"
    assert terrain.provenance["config"] == report["config"]
    assert report["config"]["max_merge_candidate_pairs"] == 2_000_000


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"grid_size_m": 0.0}, "grid_size_m must be positive"),
        ({"plateau_side_m": np.inf}, "plateau_side_m must be positive"),
        ({"collision_xy_radius_m": -0.1}, "collision_xy_radius_m must be finite and non-negative"),
        ({"max_raster_cells": 0}, "max_raster_cells must be a positive integer"),
        ({"max_merge_candidate_pairs": 0}, "max_merge_candidate_pairs must be a positive integer"),
    ],
)
def test_config_rejects_nonphysical_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        VoronoiConfig(**kwargs)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value[:2], r"shape \(T>=3, J, 3\)"),
        (lambda value: np.concatenate([value, value[:, :1]], axis=1), "names for the joint array"),
        (nonfinite_motion, "finite values"),
    ],
)
def test_motion_contract_is_validated(mutator, message):
    joints = mutator(motion())
    with pytest.raises(ValueError, match=message):
        fit_voronoi_terrain_from_motion(joints, NAMES, FPS, terrain_links=("L_Toe",))


def test_negative_support_height_requires_explicit_floor_normalization():
    with pytest.raises(ValueError, match="cannot represent negative support heights"):
        fit_one(motion(L_Toe=(0.0, 0.0, -0.2)))


def test_raster_size_guard_is_explicit():
    with pytest.raises(ValueError, match="exceeded max_raster_cells"):
        fit_one(
            motion(L_Toe=(0.0, 0.0, 0.2)),
            config=replace(VoronoiConfig(), max_raster_cells=50),
        )


def test_merge_pair_cap_preserves_prior_result_at_the_exact_limit():
    edges = [
        _TerrainEdge(
            "L_Toe",
            frame,
            np.array([0.01 * frame, 0.0, 0.2]),
            0,
            4,
        )
        for frame in range(4)
    ]
    expected = _cluster_edges(edges, VoronoiConfig())
    stats = {}
    bounded = _cluster_edges(
        edges,
        replace(VoronoiConfig(), max_merge_candidate_pairs=6),
        execution_stats=stats,
    )

    assert [(cluster.height, len(cluster.edges)) for cluster in bounded] == pytest.approx(
        [(cluster.height, len(cluster.edges)) for cluster in expected]
    )
    assert stats == {
        "n_neighbor_candidates": 6,
        "n_candidate_pairs": 6,
        "max_merge_candidate_pairs": 6,
    }


def test_merge_pair_cap_fails_before_materializing_neighbors(monkeypatch):
    from scipy import spatial

    real_tree = spatial.cKDTree

    class CountOnlyTree:
        def __init__(self, *args, **kwargs):
            self.delegate = real_tree(*args, **kwargs)

        def count_neighbors(self, other, *args, **kwargs):
            return self.delegate.count_neighbors(other.delegate, *args, **kwargs)

        def query_pairs(self, *args, **kwargs):
            raise AssertionError("pair materialization must occur only below the cap")

    monkeypatch.setattr(spatial, "cKDTree", CountOnlyTree)
    edges = [
        _TerrainEdge(
            "L_Toe",
            frame,
            np.array([0.01 * frame, 0.0, 0.2]),
            0,
            4,
        )
        for frame in range(4)
    ]

    with pytest.raises(
        ValueError,
        match=r"max_merge_candidate_pairs=5 \(6 inclusive XY-neighbor pairs\)",
    ):
        _cluster_edges(
            edges,
            replace(VoronoiConfig(), max_merge_candidate_pairs=5),
        )


def test_inclusive_pair_cap_is_explicit_at_the_strict_distance_boundary():
    edges = [
        _TerrainEdge("L_Toe", 0, np.array([0.0, 0.0, 0.2]), 0, 3),
        _TerrainEdge("L_Toe", 1, np.array([0.5, 0.0, 0.2]), 0, 3),
        _TerrainEdge("L_Toe", 2, np.array([1.0, 0.0, 0.2]), 0, 3),
    ]
    stats = {}
    clusters = _cluster_edges(
        edges,
        replace(VoronoiConfig(), max_merge_candidate_pairs=3),
        execution_stats=stats,
    )

    assert len(clusters) == 1
    assert stats["n_neighbor_candidates"] == 3
    assert stats["n_candidate_pairs"] == 2
    with pytest.raises(ValueError, match="3 inclusive XY-neighbor pairs"):
        _cluster_edges(
            edges,
            replace(VoronoiConfig(), max_merge_candidate_pairs=2),
        )


def test_dense_equal_distance_graph_preserves_reflection_and_time_invariants():
    points = [(0.2 * x, 0.2 * y, 0.045 * ((x + y) % 4)) for x in range(8) for y in range(8)]
    config = replace(
        VoronoiConfig(),
        spatial_merge_distance_m=0.31,
    )

    def signature(*, reflect: bool, reverse: bool):
        edges = [
            _TerrainEdge(
                ("L_Toe", "R_Toe")[frame % 2],
                len(points) - 1 - frame if reverse else frame,
                np.array((-x if reflect else x, y, z)),
                0,
                len(points),
            )
            for frame, (x, y, z) in enumerate(points)
        ]
        return sorted((cluster.height, len(cluster.edges)) for cluster in _cluster_edges(edges, config))

    expected = signature(reflect=False, reverse=False)
    np.testing.assert_allclose(signature(reflect=True, reverse=False), expected)
    np.testing.assert_allclose(signature(reflect=False, reverse=True), expected)
    np.testing.assert_allclose(signature(reflect=True, reverse=True), expected)


def test_dense_graph_keeps_strict_component_height_span_guard():
    count = 240
    angle = np.arange(count, dtype=float)
    points = np.column_stack(
        (
            0.1 * np.cos(angle),
            0.1 * np.sin(angle),
            np.linspace(0.0, 0.36, count),
        )
    )
    edges = [_TerrainEdge("L_Toe", frame, point, 0, count) for frame, point in enumerate(points)]
    config = VoronoiConfig()

    clusters = _cluster_edges(edges, config)

    assert len(clusters) >= 4
    assert all(
        max(edge.point[2] for edge in cluster.edges) - min(edge.point[2] for edge in cluster.edges) + 1e-12
        < config.height_merge_tolerance_m
        for cluster in clusters
    )
