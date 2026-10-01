"""Voronoi terrain reconstruction based on TIP and SceneBot.

The method combines SceneBot's motion-conditioned interaction edges, pruning, plateau
merging, and collision carving with TIP's discrete height lattice and nearest-contact
Voronoi ownership. It operates independently of TERRA's terrain-family fitters.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from terra._musclemimic import BoxSpec, TerrainSpec
from terra.terrain.stance import DEFAULT_CONTACT_JOINTS

SCENEBOT_PAPER = "https://arxiv.org/abs/2606.27581v1"
TIP_PAPER = "https://arxiv.org/abs/2203.15720"

# TIP searches one stationary body point on each foot and SceneBot uses one left-foot
# and one right-foot interaction node.  The fitted toe landmarks are the available
# contact-surface proxies for those two paper-level foot links; anatomical ankle centres
# must not become separate terrain surfaces.
DEFAULT_VORONOI_TERRAIN_LINKS = ("L_Toe", "R_Toe", "Pelvis")


PelvisSupportHeightResolver = Callable[
    [tuple[tuple[int, int], ...]],
    Sequence[float],
]


@dataclass(frozen=True)
class VoronoiConfig:
    """Numerical parameters for Voronoi terrain reconstruction.

    The 0.10 m grid, 1 m plateau side/influence, strict 0.10 m height merge,
    1 m proximity, and 0.20 m pelvis/foot-midpoint separation are inherited from
    Transformer Inertial Poser (TIP). Motion-conditioned interaction edges, pruning,
    plateau merging, and collision carving follow the structure described by SceneBot.

    ``max_merge_candidate_pairs`` is a hard safety cap on the in-memory neighbor
    graph.  Motions above the cap fail closed instead of changing the algorithm or
    spilling potentially sensitive motion data to disk.
    """

    velocity_threshold_m_s: float = 0.15
    acceleration_threshold_m_s2: float = 2.0
    min_contact_duration_s: float = 0.10
    max_contact_gap_s: float = 0.04
    grid_size_m: float = 0.10
    plateau_side_m: float = 1.0
    height_merge_tolerance_m: float = 0.10
    spatial_merge_distance_m: float = 1.0
    pelvis_foot_separation_m: float = 0.20
    collision_xy_radius_m: float = 0.05
    collision_vertical_tolerance_m: float = 0.03
    minimum_box_height_m: float = 1e-4
    max_raster_cells: int = 2_000_000
    max_merge_candidate_pairs: int = 2_000_000

    def __post_init__(self) -> None:
        positive = (
            "velocity_threshold_m_s",
            "acceleration_threshold_m_s2",
            "grid_size_m",
            "plateau_side_m",
            "height_merge_tolerance_m",
            "spatial_merge_distance_m",
        )
        nonnegative = (
            "min_contact_duration_s",
            "max_contact_gap_s",
            "pelvis_foot_separation_m",
            "collision_xy_radius_m",
            "collision_vertical_tolerance_m",
            "minimum_box_height_m",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {value!r}")
            object.__setattr__(self, name, value)
        for name in nonnegative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
            object.__setattr__(self, name, value)
        for name in ("max_raster_cells", "max_merge_candidate_pairs"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class _ContactRun:
    link: str
    start: int
    end: int
    points: np.ndarray
    active_frames: np.ndarray

    @property
    def weight(self) -> int:
        return len(self.points)

    @property
    def centre_xy(self) -> np.ndarray:
        return np.median(self.points[:, :2], axis=0)

    @property
    def height(self) -> float:
        return float(np.mean(self.points[:, 2]))


@dataclass(frozen=True)
class _TerrainEdge:
    """One paper-level temporal terrain edge, retaining its original XYZ."""

    link: str
    frame: int
    point: np.ndarray
    interval_start: int
    interval_end: int


@dataclass(frozen=True)
class _HeightCluster:
    index: int
    edges: tuple[_TerrainEdge, ...]
    height: float

    @property
    def points(self) -> np.ndarray:
        return np.stack([edge.point for edge in self.edges], axis=0)


@dataclass(frozen=True)
class _CollisionTimeline:
    frames: np.ndarray
    prefix_min_z: np.ndarray
    suffix_min_z: np.ndarray

    def minimum_outside(self, start: int, end: int) -> float:
        before = int(np.searchsorted(self.frames, start, side="left"))
        after = int(np.searchsorted(self.frames, end, side="left"))
        minimum = float("inf")
        if before:
            minimum = float(self.prefix_min_z[before - 1])
        if after < len(self.frames):
            minimum = min(minimum, float(self.suffix_min_z[after]))
        return minimum


class _UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = int(self.parent[item])
        return int(item)

    def find_many(self, items: np.ndarray) -> np.ndarray:
        """Find roots in bulk without changing union semantics."""

        roots = np.asarray(items, dtype=np.int64).copy()
        while True:
            parents = self.parent[roots]
            pending = parents != roots
            if not np.any(pending):
                break
            roots[pending] = parents[pending]
        self.parent[np.asarray(items, dtype=np.int64)] = roots
        return roots

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        # Canonical roots make component identities independent of pair traversal.
        if a > b:
            a, b = b, a
        self.parent[b] = a


def _validate_inputs(
    joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
    terrain_links: Sequence[str],
    link_surface_offsets: Mapping[str, float] | None,
    collision_points: np.ndarray | None,
    collision_point_names: Sequence[str] | None,
) -> tuple[np.ndarray, list[str], float, tuple[str, ...], dict[str, float], np.ndarray, list[str] | None]:
    joints = np.asarray(joints, dtype=float)
    if joints.ndim != 3 or joints.shape[0] < 3 or joints.shape[2] != 3:
        raise ValueError(f"joints must have shape (T>=3, J, 3), got {joints.shape}")
    if not np.all(np.isfinite(joints)):
        raise ValueError("joints must contain only finite values")
    names = list(demo_joints)
    if len(names) != joints.shape[1]:
        raise ValueError(f"expected {joints.shape[1]} names for the joint array, got {len(names)}")
    if len(set(names)) != len(names):
        raise ValueError("demo_joints must not contain duplicate names")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be positive and finite, got {fps!r}")

    links = tuple(str(link) for link in terrain_links)
    if not links or len(set(links)) != len(links):
        raise ValueError("terrain_links must contain at least one unique link name")
    missing = [link for link in links if link not in names]
    if missing:
        raise ValueError(f"terrain link(s) missing from demo_joints: {missing}")

    offsets = {str(name): float(value) for name, value in dict(link_surface_offsets or {}).items()}
    unknown_offsets = sorted(set(offsets) - set(links))
    if unknown_offsets:
        raise ValueError(f"link_surface_offsets contains non-terrain link(s): {unknown_offsets}")
    if any(not np.isfinite(value) for value in offsets.values()):
        raise ValueError("link_surface_offsets must contain only finite values")
    offsets = {link: offsets.get(link, 0.0) for link in links}

    if collision_points is None:
        collision = joints
        collision_names = names
    else:
        collision = np.asarray(collision_points, dtype=float)
        if collision.ndim != 3 or collision.shape[0] != len(joints) or collision.shape[2] != 3:
            raise ValueError(
                f"collision_points must have shape (T, K, 3) with the same frame count as joints, got {collision.shape}"
            )
        if not np.all(np.isfinite(collision)):
            raise ValueError("collision_points must contain only finite values")
        collision_names = None if collision_point_names is None else list(collision_point_names)
        if collision_names is not None:
            if len(collision_names) != collision.shape[1]:
                raise ValueError("collision_point_names must contain one name per collision point")
            # Dense collision proxies commonly sample one link at several points. Repeated
            # names intentionally exempt every sample on an active interaction link.
    if collision_points is None and collision_point_names is not None and list(collision_point_names) != names:
        raise ValueError("collision_point_names may only override names when collision_points is provided")
    return joints, names, fps, links, offsets, collision, collision_names


def _close_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    if max_gap <= 0 or len(out) < 3:
        return out
    i = 0
    while i < len(out):
        if out[i]:
            i += 1
            continue
        j = i
        while j < len(out) and not out[j]:
            j += 1
        if i > 0 and j < len(out) and j - i <= max_gap:
            out[i:j] = True
        i = j
    return out


def _true_runs(mask: np.ndarray, min_frames: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        end = i + 1
        while end < len(mask) and mask[end]:
            end += 1
        if end - i >= min_frames:
            runs.append((i, end))
        i = end
    return runs


def _kinematic_contact_mask(points: np.ndarray, fps: float, config: VoronoiConfig) -> tuple[np.ndarray, dict]:
    dt = 1.0 / fps
    velocity = np.gradient(points, dt, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=2)
    speed = np.linalg.norm(velocity, axis=1)
    accel = np.linalg.norm(acceleration, axis=1)
    raw = (speed <= config.velocity_threshold_m_s) & (accel <= config.acceleration_threshold_m_s2)
    # ``max_contact_gap_s`` is a true upper bound.  Flooring avoids closing a
    # 50 ms frame when the configured maximum is 40 ms at 20 Hz.
    closed = _close_short_gaps(raw, max(0, math.floor(config.max_contact_gap_s * fps + 1e-12)))
    return closed, {
        "raw_candidate_frames": int(raw.sum()),
        "gap_closed_candidate_frames": int(closed.sum()),
        "speed_min_m_s": float(speed.min()),
        "speed_median_m_s": float(np.median(speed)),
        "acceleration_min_m_s2": float(accel.min()),
        "acceleration_median_m_s2": float(np.median(accel)),
    }


def _infer_contact_runs(
    joints: np.ndarray,
    names: Sequence[str],
    fps: float,
    terrain_links: Sequence[str],
    offsets: Mapping[str, float],
    config: VoronoiConfig,
    foot_links: Sequence[str],
    pelvis_link: str | None,
) -> tuple[list[_ContactRun], dict[str, np.ndarray], dict, list[str]]:
    min_frames = max(1, math.ceil(config.min_contact_duration_s * fps - 1e-12))
    active: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict] = {}
    warnings: list[str] = []
    runs: list[_ContactRun] = []
    foot_indices = [names.index(link) for link in foot_links if link in names]
    pelvis_guard_available = len(foot_indices) >= 2
    if pelvis_link in terrain_links and not pelvis_guard_available:
        warnings.append(
            f"pelvis link {pelvis_link!r} was disabled because all foot_links are required for the pelvis-distance guard"
        )

    for link in terrain_links:
        index = names.index(link)
        mask, diag = _kinematic_contact_mask(joints[:, index], fps, config)

        pelvis_suppressed = 0
        if link == pelvis_link:
            if not pelvis_guard_available:
                mask[:] = False
            elif config.pelvis_foot_separation_m > 0.0:
                foot_midpoint_xy = joints[:, foot_indices, :2].mean(axis=1)
                pelvis_xy = joints[:, index, :2]
                distance = np.linalg.norm(foot_midpoint_xy - pelvis_xy, axis=1)
                too_close = distance <= config.pelvis_foot_separation_m
                pelvis_suppressed = int(np.count_nonzero(mask & too_close))
                mask &= ~too_close

        intervals = _true_runs(mask, min_frames)
        accepted = np.zeros(len(joints), dtype=bool)
        corrected = joints[:, index].copy()
        corrected[:, 2] -= offsets[link]
        for start, end in intervals:
            frames = np.arange(start, end, dtype=int)
            accepted[start:end] = True
            runs.append(_ContactRun(link, start, end, corrected[start:end].copy(), frames))
        active[link] = accepted
        diagnostics[link] = {
            **diag,
            "pelvis_frames_suppressed": pelvis_suppressed,
            "accepted_frames": int(accepted.sum()),
            "accepted_runs": len(intervals),
        }

    runs.sort(key=lambda run: (run.link, run.start, run.end, *run.centre_xy, run.height))
    return runs, active, diagnostics, warnings


def _resolve_pelvis_support_heights(
    runs: Sequence[_ContactRun],
    pelvis_link: str | None,
    resolver: PelvisSupportHeightResolver | None,
) -> tuple[list[_ContactRun], tuple[float, ...] | None]:
    """Change only the heights of Voronoi-detected pelvis support runs."""

    if resolver is None:
        return list(runs), None
    if pelvis_link is None:
        raise ValueError("pelvis_support_height_resolver requires an active pelvis_link")

    pelvis_runs = [run for run in runs if run.link == pelvis_link]
    intervals = tuple((run.start, run.end) for run in pelvis_runs)
    values = np.asarray(resolver(intervals), dtype=float)
    if values.ndim != 1 or len(values) != len(intervals):
        raise ValueError(
            "pelvis support height resolver must return one height per "
            f"Voronoi-detected interval ({len(intervals)} expected, got shape {values.shape})"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("resolved pelvis support heights must be finite")

    height_by_interval = {interval: float(height) for interval, height in zip(intervals, values, strict=True)}
    resolved: list[_ContactRun] = []
    for run in runs:
        if run.link != pelvis_link:
            resolved.append(run)
            continue
        points = run.points.copy()
        points[:, 2] = height_by_interval[(run.start, run.end)]
        resolved.append(_ContactRun(run.link, run.start, run.end, points, run.active_frames))
    return resolved, tuple(float(value) for value in values)


def _axis_cells(centre: float, half_side: float, resolution: float) -> range:
    # Cell i covers [i*r, (i+1)*r] and has centre (i+0.5)*r. Select centres inside
    # the half-open plateau interval so an exactly 1 m plateau on a 0.1 m grid has
    # exactly ten cells along an aligned axis.
    lower = (centre - half_side) / resolution - 0.5
    upper = (centre + half_side) / resolution - 0.5
    first = math.ceil(lower - 1e-12)
    stop = math.ceil(upper - 1e-12)
    return range(first, max(first + 1, stop))


def _edge_plateau_cells(edge: _TerrainEdge, config: VoronoiConfig) -> tuple[tuple[int, int], ...]:
    half = config.plateau_side_m / 2.0
    x_cells = _axis_cells(float(edge.point[0]), half, config.grid_size_m)
    y_cells = _axis_cells(float(edge.point[1]), half, config.grid_size_m)
    return tuple((ix, iy) for ix in x_cells for iy in y_cells)


def _candidate_edge_cells(edges: Sequence[_TerrainEdge], config: VoronoiConfig) -> dict[tuple[int, int], None]:
    cells: dict[tuple[int, int], None] = {}
    for edge in edges:
        for cell in _edge_plateau_cells(edge, config):
            cells[cell] = None
            if len(cells) > config.max_raster_cells:
                raise ValueError(
                    f"Voronoi raster exceeded max_raster_cells={config.max_raster_cells}; "
                    "increase the guard explicitly or use a coarser grid"
                )
    return cells


def _edges_from_runs(runs: Sequence[_ContactRun]) -> list[_TerrainEdge]:
    edges = [
        _TerrainEdge(run.link, int(frame), point.copy(), run.start, run.end)
        for run in runs
        for frame, point in zip(run.active_frames, run.points, strict=True)
    ]
    edges.sort(key=lambda edge: (edge.frame, edge.link, *edge.point, edge.interval_start, edge.interval_end))
    return edges


def _edge_causes_outside_collision(
    edge: _TerrainEdge,
    collision_timelines: Mapping[tuple[int, int], _CollisionTimeline],
    config: VoronoiConfig,
) -> bool:
    """Test the interaction location before any plateau geometry is synthesized.

    SceneBot prunes interaction-graph edges before reconstructing the square plateaus.
    An edge therefore owns only its spatial interaction location at this stage, not the
    full plateau that will later receive TIP's bounded Voronoi influence.  Testing every
    future plateau cell here made the graph-pruning decision depend on the influence
    width and removed otherwise feasible foot contacts.
    """

    limit = float(edge.point[2]) - config.collision_vertical_tolerance_m
    cell = (
        math.floor(float(edge.point[0]) / config.grid_size_m),
        math.floor(float(edge.point[1]) / config.grid_size_m),
    )
    timeline = collision_timelines.get(cell)
    return (
        timeline is not None
        and timeline.minimum_outside(edge.interval_start, edge.interval_end) < limit
    )


def _prune_infeasible_edges(
    edges: Sequence[_TerrainEdge],
    collision_timelines: Mapping[tuple[int, int], _CollisionTimeline],
    config: VoronoiConfig,
) -> tuple[list[_TerrainEdge], dict]:
    """Prune an entire candidate edge before synthesis, as SceneBot Algorithm 1 does."""

    surviving: list[_TerrainEdge] = []
    pruned_by_link: dict[str, int] = defaultdict(int)
    # Stationary runs often contain hundreds of identical edges.  The collision
    # decision depends on XYZ and the run interval, not the edge timestep, so cache it.
    decisions: dict[tuple[float, float, float, int, int], bool] = {}
    for edge in edges:
        key = (*map(float, edge.point), edge.interval_start, edge.interval_end)
        collides = decisions.get(key)
        if collides is None:
            collides = _edge_causes_outside_collision(edge, collision_timelines, config)
            decisions[key] = collides
        if collides:
            pruned_by_link[edge.link] += 1
        else:
            surviving.append(edge)
    return surviving, {
        "n_candidate_edges": len(edges),
        "n_edges_pruned_outside_interval_collision": len(edges) - len(surviving),
        "n_edges_after_pruning": len(surviving),
        "pruned_edges_by_link": dict(sorted(pruned_by_link.items())),
        "force_closure_pruning": "not_applicable_terrain_only",
    }


def _cluster_edges(
    edges: Sequence[_TerrainEdge],
    config: VoronoiConfig,
    *,
    execution_stats: dict[str, int] | None = None,
) -> list[_HeightCluster]:
    """Merge adjacent edges deterministically within a bounded in-memory graph.

    SceneBot specifies only that nearby, similar-height plateaus are merged.  This
    implementation finds XY neighbors with a k-d tree, processes them globally by
    increasing squared distance, and merges components only when their total height
    span remains strictly below the configured tolerance.  Equal-distance candidates
    are handled as one invariant batch.

    The k-d tree's inclusive neighbor count is checked before materializing pairs.
    This deliberately conservative cap can reject a graph containing many pairs
    exactly on the configured distance boundary; accepted graphs retain the prior
    exact strict-distance filter and merge semantics.
    """

    stats = {} if execution_stats is None else execution_stats
    stats.clear()
    stats.update(
        {
            "n_neighbor_candidates": 0,
            "n_candidate_pairs": 0,
            "max_merge_candidate_pairs": config.max_merge_candidate_pairs,
        }
    )
    if not edges:
        return []

    ordered = sorted(
        edges,
        key=lambda edge: (
            float(edge.point[0]),
            float(edge.point[1]),
            float(edge.point[2]),
            edge.frame,
            edge.link,
            edge.interval_start,
            edge.interval_end,
        ),
    )

    # Temporal samples at exactly the same XYZ are indistinguishable for merging but
    # retain their multiplicity in the final cluster mean.
    atom_edges: dict[tuple[float, float, float], list[_TerrainEdge]] = defaultdict(list)
    for edge in ordered:
        atom_edges[tuple(map(float, edge.point))].append(edge)
    atom_points = np.asarray(sorted(atom_edges), dtype=float)
    atoms = [tuple(atom_edges[tuple(point)]) for point in atom_points]

    union = _UnionFind(len(atoms))
    minimum_height = atom_points[:, 2].tolist()
    maximum_height = minimum_height.copy()
    atom_signatures = [
        (float(atom_points[index, 2]), tuple(sorted(edge.link for edge in atom))) for index, atom in enumerate(atoms)
    ]
    component_members = {index: [signature] for index, signature in enumerate(atom_signatures)}

    def merge_if_compatible(left: int, right: int) -> None:
        left, right = union.find(left), union.find(right)
        if left == right:
            return
        combined_min = min(minimum_height[left], minimum_height[right])
        combined_max = max(maximum_height[left], maximum_height[right])
        if not combined_max - combined_min + 1e-12 < config.height_merge_tolerance_m:
            return
        union.union(left, right)
        root = union.find(left)
        other = right if root == left else left
        minimum_height[root] = combined_min
        maximum_height[root] = combined_max
        component_members[root].extend(component_members.pop(other))

    def merge_tied_pairs(pairs: np.ndarray) -> None:
        """Apply one simultaneous rounded-distance batch invariantly."""

        root_snapshot = union.find_many(np.arange(len(atoms), dtype=np.int64))
        tie_graph = _UnionFind(len(atoms))
        involved: set[int] = set()
        left = root_snapshot[pairs[:, 0]]
        right = root_snapshot[pairs[:, 1]]
        keep = left != right
        for a, b in zip(left[keep], right[keep], strict=True):
            a, b = int(a), int(b)
            tie_graph.union(a, b)
            involved.add(a)
            involved.add(b)
        if not involved:
            return

        component_signature = {root: tuple(sorted(component_members[root])) for root in involved}
        graph_components: dict[int, list[int]] = defaultdict(list)
        for root in involved:
            graph_components[tie_graph.find(root)].append(root)

        band_by_root: dict[int, int] = {}
        next_band = 0
        ordered_components = sorted(
            graph_components.values(),
            key=lambda roots: min(component_signature[root] for root in roots),
        )
        for graph_component in ordered_components:
            signature_groups: dict[tuple, list[int]] = defaultdict(list)
            for root in graph_component:
                signature_groups[component_signature[root]].append(root)
            band_min = band_max = None
            for signature in sorted(signature_groups):
                roots = signature_groups[signature]
                signature_min = minimum_height[roots[0]]
                signature_max = maximum_height[roots[0]]
                if (
                    band_min is None
                    or max(float(band_max), signature_max) - min(float(band_min), signature_min) + 1e-12
                    >= config.height_merge_tolerance_m
                ):
                    next_band += 1
                    band_min, band_max = signature_min, signature_max
                else:
                    band_min = min(float(band_min), signature_min)
                    band_max = max(float(band_max), signature_max)
                for root in roots:
                    band_by_root[root] = next_band

        for a, b in zip(left, right, strict=True):
            a, b = int(a), int(b)
            if a != b and band_by_root.get(a) == band_by_root.get(b):
                merge_if_compatible(a, b)

    def merge_singleton_pairs(pairs: np.ndarray) -> None:
        for start in range(0, len(pairs), 8_192):
            block = pairs[start : start + 8_192]
            left_roots = union.find_many(block[:, 0])
            right_roots = union.find_many(block[:, 1])
            keep = left_roots != right_roots
            for left, right in zip(block[keep, 0], block[keep, 1], strict=True):
                merge_if_compatible(int(left), int(right))

    pair_dtype = np.dtype([("key", "<f8"), ("left", "<u4"), ("right", "<u4")])

    def merge_sorted_records(records: np.ndarray) -> None:
        cursor = 0
        while cursor < len(records):
            stop = cursor + 1
            key = records["key"][cursor]
            while stop < len(records) and records["key"][stop] == key:
                stop += 1
            if stop - cursor > 1:
                pairs = np.column_stack((records["left"][cursor:stop], records["right"][cursor:stop])).astype(
                    np.int64, copy=False
                )
                merge_tied_pairs(pairs)
                cursor = stop
                continue

            unique_stop = stop
            while unique_stop < len(records) and unique_stop - cursor < 8_192:
                following = unique_stop + 1
                following_key = records["key"][unique_stop]
                while following < len(records) and records["key"][following] == following_key:
                    following += 1
                if following - unique_stop > 1:
                    break
                unique_stop = following
            pairs = np.column_stack(
                (
                    records["left"][cursor:unique_stop],
                    records["right"][cursor:unique_stop],
                )
            ).astype(np.int64, copy=False)
            merge_singleton_pairs(pairs)
            cursor = unique_stop

    if len(atoms) > 1:
        from scipy.spatial import cKDTree

        if len(atoms) >= 2**32:
            raise ValueError("Voronoi merging supports fewer than 2^32 unique XYZ atoms")
        distance = config.spatial_merge_distance_m
        distance_sq = distance * distance
        tree = cKDTree(atom_points[:, :2])
        neighbor_count = (int(tree.count_neighbors(tree, distance)) - len(atoms)) // 2
        stats["n_neighbor_candidates"] = neighbor_count
        if neighbor_count > config.max_merge_candidate_pairs:
            raise ValueError(
                "Voronoi merging exceeded "
                f"max_merge_candidate_pairs={config.max_merge_candidate_pairs} "
                f"({neighbor_count} inclusive XY-neighbor pairs); increase the "
                "explicit cap, shorten/downsample the motion, or reduce "
                "spatial_merge_distance_m"
            )

        pairs = tree.query_pairs(distance, output_type="ndarray")
        if len(pairs):
            delta = atom_points[pairs[:, 0], :2] - atom_points[pairs[:, 1], :2]
            separation_sq = np.einsum("ij,ij->i", delta, delta)
            strict = separation_sq + 1e-12 < distance_sq
            pairs = pairs[strict]
            separation_sq = separation_sq[strict]
        stats["n_candidate_pairs"] = len(pairs)

        if len(pairs):
            records = np.empty(len(pairs), dtype=pair_dtype)
            records["key"] = np.round(separation_sq, decimals=12)
            records["left"] = pairs[:, 0]
            records["right"] = pairs[:, 1]
            records = records[np.argsort(records["key"], kind="stable")]
            merge_sorted_records(records)

    spatial_groups: dict[int, list[_TerrainEdge]] = defaultdict(list)
    for index, atom in enumerate(atoms):
        spatial_groups[union.find(index)].extend(atom)

    components: list[tuple[float, tuple[_TerrainEdge, ...]]] = []
    for spatial_edges in spatial_groups.values():
        component = tuple(spatial_edges)
        components.append((float(np.mean([item.point[2] for item in component])), component))

    components.sort(
        key=lambda item: (
            item[0],
            item[1][0].frame,
            item[1][0].link,
            *item[1][0].point[:2],
        )
    )
    return [_HeightCluster(index, component, height) for index, (height, component) in enumerate(components)]


def _rasterize_plateaus(
    clusters: Sequence[_HeightCluster], config: VoronoiConfig
) -> dict[tuple[int, int], dict[int, float]]:
    proposals: dict[tuple[int, int], dict[int, float]] = {}
    half = config.plateau_side_m / 2.0
    resolution = config.grid_size_m
    for cluster in clusters:
        for edge in cluster.edges:
            centre = edge.point[:2]
            for ix in _axis_cells(float(centre[0]), half, resolution):
                cell_x = (ix + 0.5) * resolution
                for iy in _axis_cells(float(centre[1]), half, resolution):
                    key = (ix, iy)
                    cell_y = (iy + 0.5) * resolution
                    distance = float((cell_x - centre[0]) ** 2 + (cell_y - centre[1]) ** 2)
                    per_cluster = proposals.setdefault(key, {})
                    per_cluster[cluster.index] = min(
                        distance,
                        per_cluster.get(cluster.index, float("inf")),
                    )
                    if len(proposals) > config.max_raster_cells:
                        raise ValueError(
                            f"Voronoi raster exceeded max_raster_cells={config.max_raster_cells}; "
                            "increase the guard explicitly or use a coarser grid"
                        )
    return proposals


def _cell_intersects_disk(ix: int, iy: int, point_xy: np.ndarray, radius: float, resolution: float) -> bool:
    x0, x1 = ix * resolution, (ix + 1) * resolution
    y0, y1 = iy * resolution, (iy + 1) * resolution
    dx = max(x0 - point_xy[0], 0.0, point_xy[0] - x1)
    dy = max(y0 - point_xy[1], 0.0, point_xy[1] - y1)
    return dx * dx + dy * dy <= radius * radius + 1e-15


def _collision_samples_by_cell(
    proposals: Mapping[tuple[int, int], object],
    collision_points: np.ndarray,
    collision_names: Sequence[str] | None,
    config: VoronoiConfig,
) -> dict[tuple[int, int], dict[tuple[int, str | None], float]]:
    resolution = config.grid_size_m
    radius = config.collision_xy_radius_m
    samples: dict[tuple[int, int], dict[tuple[int, str | None], float]] = {}
    names = None if collision_names is None else list(collision_names)
    for frame, frame_points in enumerate(collision_points):
        for point_index, point in enumerate(frame_points):
            name = None if names is None else names[point_index]
            ix0 = math.floor((point[0] - radius) / resolution)
            ix1 = math.floor((point[0] + radius) / resolution)
            iy0 = math.floor((point[1] - radius) / resolution)
            iy1 = math.floor((point[1] + radius) / resolution)
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    key = (ix, iy)
                    if key not in proposals or not _cell_intersects_disk(ix, iy, point[:2], radius, resolution):
                        continue
                    # One lowest sample per frame/link is sufficient. Repeated names are
                    # dense samples on the same link and share the same interval
                    # exemption; unnamed proxies cannot be exempted and share one minimum.
                    per_cell = samples.setdefault(key, {})
                    sample_key = (frame, name)
                    per_cell[sample_key] = min(float(point[2]), per_cell.get(sample_key, float("inf")))
    return samples


def _collision_timelines(
    samples: Mapping[tuple[int, int], Mapping[tuple[int, str | None], float]],
) -> dict[tuple[int, int], _CollisionTimeline]:
    timelines = {}
    for cell, cell_samples in samples.items():
        minimum_by_frame: dict[int, float] = {}
        for (frame, _name), z in cell_samples.items():
            minimum_by_frame[frame] = min(z, minimum_by_frame.get(frame, float("inf")))
        frames = np.array(sorted(minimum_by_frame), dtype=int)
        heights = np.array([minimum_by_frame[int(frame)] for frame in frames], dtype=float)
        timelines[cell] = _CollisionTimeline(
            frames=frames,
            prefix_min_z=np.minimum.accumulate(heights),
            suffix_min_z=np.minimum.accumulate(heights[::-1])[::-1],
        )
    return timelines


def _cluster_contact_intervals(
    clusters: Sequence[_HeightCluster],
) -> dict[int, dict[str, tuple[tuple[int, int], ...]]]:
    allowed: dict[int, dict[str, tuple[tuple[int, int], ...]]] = {}
    for cluster in clusters:
        by_link: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for edge in cluster.edges:
            by_link[edge.link].add((edge.interval_start, edge.interval_end))
        allowed[cluster.index] = {link: tuple(sorted(intervals)) for link, intervals in by_link.items()}
    return allowed


def _proposal_collides(
    cluster_index: int,
    height: float,
    samples: Mapping[tuple[int, str | None], float] | None,
    allowed: Mapping[int, Mapping[str, Sequence[tuple[int, int]]]],
    vertical_tolerance: float,
) -> bool:
    if not samples:
        return False
    cluster_intervals = allowed.get(cluster_index, {})
    for (frame, name), point_z in samples.items():
        if point_z >= height - vertical_tolerance:
            continue
        if name is not None and any(start <= frame < end for start, end in cluster_intervals.get(name, ())):
            # The generating interaction link is permitted to occupy this candidate
            # plateau only during this candidate cluster's own interaction interval.
            continue
        return True
    return False


def _select_and_carve_cells(
    proposals: Mapping[tuple[int, int], Mapping[int, float]],
    clusters: Sequence[_HeightCluster],
    collision_samples: Mapping[tuple[int, int], Mapping[tuple[int, str | None], float]],
    config: VoronoiConfig,
) -> tuple[dict[tuple[int, int], float], dict[str, int]]:
    heights = {cluster.index: cluster.height for cluster in clusters}
    allowed = _cluster_contact_intervals(clusters)
    output: dict[tuple[int, int], float] = {}
    positive_before = 0
    lowered = 0
    carved = 0
    floor_owned = 0
    for cell, per_cluster in sorted(proposals.items()):
        ranked = sorted(per_cluster, key=lambda index: (per_cluster[index], -heights[index], index))
        initial = ranked[0]
        initial_height = heights[initial]
        if initial_height <= config.minimum_box_height_m:
            floor_owned += 1
            continue
        positive_before += 1
        samples = collision_samples.get(cell)
        if not _proposal_collides(
            initial,
            initial_height,
            samples,
            allowed,
            config.collision_vertical_tolerance_m,
        ):
            output[cell] = initial_height
            continue

        # Collision carving reveals the highest non-colliding lower proposal where
        # overlapping plateaus exist; otherwise the implicit z=0 floor is exposed.
        lower = sorted(
            (index for index in ranked[1:] if heights[index] < initial_height - 1e-12),
            key=lambda index: (-heights[index], per_cluster[index], index),
        )
        selected = next(
            (
                index
                for index in lower
                if heights[index] > config.minimum_box_height_m
                and not _proposal_collides(
                    index,
                    heights[index],
                    samples,
                    allowed,
                    config.collision_vertical_tolerance_m,
                )
            ),
            None,
        )
        if selected is None:
            carved += 1
        else:
            output[cell] = heights[selected]
            lowered += 1
    return output, {
        "n_cells_before_carving": positive_before,
        "n_cells_floor_owned": floor_owned,
        "n_cells_lowered": lowered,
        "n_cells_carved_to_floor": carved,
        "n_cells_after_carving": len(output),
    }


def _rectangles_for_height(cells: set[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    rows: dict[int, list[int]] = defaultdict(list)
    for ix, iy in cells:
        rows[iy].append(ix)

    rectangles: list[tuple[int, int, int, int]] = []
    active: dict[tuple[int, int], int] = {}
    previous_y: int | None = None
    for iy in sorted(rows):
        xs = sorted(rows[iy])
        runs: list[tuple[int, int]] = []
        start = previous = xs[0]
        for ix in xs[1:]:
            if ix == previous + 1:
                previous = ix
                continue
            runs.append((start, previous))
            start = previous = ix
        runs.append((start, previous))

        if previous_y is None or iy != previous_y + 1:
            for (x0, x1), y0 in active.items():
                rectangles.append((x0, x1, y0, previous_y if previous_y is not None else y0))
            active = {}
        current = set(runs)
        for run, y0 in list(active.items()):
            if run not in current:
                rectangles.append((run[0], run[1], y0, previous_y if previous_y is not None else y0))
                del active[run]
        for run in runs:
            active.setdefault(run, iy)
        previous_y = iy

    if previous_y is not None:
        for (x0, x1), y0 in active.items():
            rectangles.append((x0, x1, y0, previous_y))
    rectangles.sort(key=lambda rectangle: (rectangle[2], rectangle[0], rectangle[3], rectangle[1]))
    return rectangles


def _cells_to_boxes(cells: Mapping[tuple[int, int], float], resolution: float) -> tuple[BoxSpec, ...]:
    by_height: dict[float, set[tuple[int, int]]] = defaultdict(set)
    for cell, height in cells.items():
        by_height[float(height)].add(cell)
    unpacked: list[tuple[float, tuple[int, int, int, int]]] = []
    for height in sorted(by_height):
        unpacked.extend((height, rectangle) for rectangle in _rectangles_for_height(by_height[height]))
    unpacked.sort(key=lambda item: (item[1][2], item[1][0], item[0], item[1][3], item[1][1]))

    boxes = []
    for index, (height, (ix0, ix1, iy0, iy1)) in enumerate(unpacked):
        x0, x1 = ix0 * resolution, (ix1 + 1) * resolution
        y0, y1 = iy0 * resolution, (iy1 + 1) * resolution
        boxes.append(
            BoxSpec(
                pos=((x0 + x1) / 2.0, (y0 + y1) / 2.0, height / 2.0),
                size=((x1 - x0) / 2.0, (y1 - y0) / 2.0, height / 2.0),
                yaw=0.0,
                pitch=0.0,
                name=f"terrain_box_voronoi_{index:04d}",
            )
        )
    return tuple(boxes)


def fit_voronoi_terrain_from_motion(
    joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
    *,
    terrain_links: Sequence[str] = DEFAULT_VORONOI_TERRAIN_LINKS,
    pelvis_link: str | None = "Pelvis",
    link_surface_offsets: Mapping[str, float] | None = None,
    collision_points: np.ndarray | None = None,
    collision_point_names: Sequence[str] | None = None,
    pelvis_support_height_resolver: PelvisSupportHeightResolver | None = None,
    pelvis_support_height_source: str = "fixed_link_offset",
    config: VoronoiConfig | None = None,
    input_stage: str = "fitted_smplh_landmarks",
) -> tuple[TerrainSpec, dict]:
    """Construct a Voronoi 2.5D terrain from world-space joint motion.

    Args:
        joints: Link or landmark positions with shape ``(T, J, 3)``.
        demo_joints: Names indexing the joint dimension.
        fps: Motion sampling rate.
        terrain_links: Links eligible to create terrain contacts. The TERRA adapter
            defaults to toe proxies and the pelvis; wrist/object reconstruction is out
            of scope for this terrain-only baseline.
        pelvis_link: Pelvis link to guard, or ``None`` to disable pelvis-specific logic.
        link_surface_offsets: Constant vertical distance from each link trajectory to
            its contact surface. Missing entries are explicitly recorded as zero; no
            TERRA sole calibration is silently applied.
        collision_points: Optional collision points with shape ``(T, K, 3)``.
            Defaults to all input joints.
        collision_point_names: Names for custom collision points. When present, an
            active interaction link is exempt only during its contact interval.
        pelvis_support_height_resolver: Optional callback that receives the pelvis
            intervals detected by Voronoi and returns one support-surface height per
            interval. It changes heights only; contact timing remains Voronoi's own.
        pelvis_support_height_source: Provenance label for the optional resolver.
        config: Under-specified numerical assumptions.
        input_stage: Label for the motion representation supplied in ``joints``.

    Returns:
        A ``TerrainSpec`` and a detailed, JSON-serializable reconstruction report.
    """

    config = config or VoronoiConfig()
    if not isinstance(config, VoronoiConfig):
        raise TypeError("config must be a VoronoiConfig")
    if not isinstance(input_stage, str) or not input_stage.strip():
        raise ValueError("input_stage must be a non-empty motion-stage label")
    if not isinstance(pelvis_support_height_source, str) or not pelvis_support_height_source.strip():
        raise ValueError("pelvis_support_height_source must be a non-empty label")
    joints, names, fps, links, offsets, collision, collision_names = _validate_inputs(
        joints,
        demo_joints,
        fps,
        terrain_links,
        link_surface_offsets,
        collision_points,
        collision_point_names,
    )
    foot_links = tuple(link for link in DEFAULT_CONTACT_JOINTS if link in links)
    if pelvis_link is not None:
        pelvis_link = str(pelvis_link)
        if pelvis_link not in links:
            raise ValueError("pelvis_link must be one of terrain_links or None")

    runs, _active, contact_diagnostics, warnings = _infer_contact_runs(
        joints,
        names,
        fps,
        links,
        offsets,
        config,
        DEFAULT_CONTACT_JOINTS,
        pelvis_link,
    )
    runs, resolved_pelvis_heights = _resolve_pelvis_support_heights(
        runs,
        pelvis_link,
        pelvis_support_height_resolver,
    )
    candidate_edges = _edges_from_runs(runs)
    negative_edges = [edge for edge in candidate_edges if edge.point[2] < -1e-12]
    if negative_edges:
        minimum = min(float(edge.point[2]) for edge in negative_edges)
        raise ValueError(
            "Voronoi TerrainSpec output requires a z=0 floor datum and cannot "
            f"represent negative support heights (minimum candidate {minimum:.6g} m)"
        )
    candidate_cells = _candidate_edge_cells(candidate_edges, config)
    all_collision_samples = _collision_samples_by_cell(
        candidate_cells,
        collision,
        collision_names,
        config,
    )
    edges, pruning = _prune_infeasible_edges(
        candidate_edges,
        _collision_timelines(all_collision_samples),
        config,
    )
    merge_execution: dict[str, int] = {}
    clusters = _cluster_edges(edges, config, execution_stats=merge_execution)
    proposals = _rasterize_plateaus(clusters, config)
    collision_samples = {cell: all_collision_samples[cell] for cell in proposals if cell in all_collision_samples}
    cells, carve = _select_and_carve_cells(proposals, clusters, collision_samples, config)
    boxes = _cells_to_boxes(cells, config.grid_size_m)

    zero_offset_links = [link for link in links if offsets[link] == 0.0]
    if zero_offset_links:
        warnings.append(
            "zero contact-surface offset assumed for "
            + ", ".join(zero_offset_links)
            + "; pass contact-surface trajectories or explicit link_surface_offsets for joint/link centres"
        )
    if collision_points is None:
        collision_model = "input_joints"
        warnings.append(
            "edge pruning and collision carving use sparse input joints rather than full robot collision geometry"
        )
    else:
        collision_model = "caller_supplied_points"
        if collision_names is None:
            warnings.append("custom collision proxies have no names, so active contact-link exemptions are unavailable")
    if any(cluster.height <= config.minimum_box_height_m for cluster in clusters):
        warnings.append("non-positive or floor-height plateau proposals were represented by the implicit z=0 floor")

    config_dict = asdict(config)
    contact_source = "kinematic"
    paper_specified = [
        "represent each foot and the pelvis as one paper-level interaction link",
        "detect low-velocity and low-acceleration interaction-link contacts",
        "prune whole infeasible edges that collide outside their interaction interval",
        "add a square plateau centred at each contact position and height",
        "merge nearby or isolated plateaus at similar heights",
        "carve robot-colliding terrain regions after plateau merging",
        "represent terrain as a 2.5D elevation map",
    ]
    tip_derived = [
        "grid_size_m",
        "plateau_side_m",
        "height_merge_tolerance_m",
        "spatial_merge_distance_m",
        "pelvis_foot_separation_m",
        "nearest-contact ownership where different height clusters overlap",
    ]
    algorithm_details = [
        "velocity_threshold_m_s",
        "acceleration_threshold_m_s2",
        "min_contact_duration_s",
        "max_contact_gap_s",
        (
            "velocity_threshold_m_s is applied to raw 3D proxy speed; SceneBot specifies "
            "low velocity without a number, while TIP's reported 0.25 threshold applies "
            "to its velocity-plus-temporal-regularizer objective"
        ),
        (
            "acceleration_threshold_m_s2 implements SceneBot's low-acceleration "
            "criterion, whose numerical threshold is not reported"
        ),
        (
            "min_contact_duration_s and max_contact_gap_s implement SceneBot's stated "
            "temporal consistency, whose numerical thresholds are not reported"
        ),
        "collision_xy_radius_m",
        "collision_vertical_tolerance_m",
        "world derivatives use numpy.gradient without smoothing",
        "gap-closed frames become candidate edges as temporal-consistency evidence",
        (
            "interaction-edge pruning checks point/disk proxies at the edge's "
            "interaction-location grid cell before plateau reconstruction"
        ),
        "exact-XY merge candidates are processed globally by increasing distance; 1e-12 m^2 distance ties are batched into invariant lower-first height/link bands",
        "spatial components merge Kruskal-style while total height span remains strictly below height_merge_tolerance_m",
        "exact duplicate XYZ samples are compressed only for merge-graph construction",
        "max_merge_candidate_pairs fails closed before materializing an oversized inclusive XY-neighbor graph",
        "different height clusters use nearest-contact ownership with a higher-surface deterministic tie break",
        "collision carving reveals the highest non-colliding lower proposal, else floor",
        "z=0 is an implicit floor and negative support heights are rejected",
        "row-run boxes preserve cell interiors; shared closed box boundaries resolve to the higher neighbor",
    ]
    provenance = {
        "source": "fit_voronoi_terrain_from_motion",
        "method": "Voronoi",
        "references": [TIP_PAPER, SCENEBOT_PAPER],
        "input_stage": input_stage,
        "contact_source": contact_source,
        "pelvis_support_height_source": pelvis_support_height_source,
        "terrain_links": list(links),
        "link_surface_offsets_m": offsets,
        "collision_model": collision_model,
        "config": config_dict,
        "operations": paper_specified,
        "voronoi_parameters": tip_derived,
        "algorithm_details": algorithm_details,
    }
    terrain = TerrainSpec(boxes=boxes, provenance=provenance)
    surviving_frames = {(edge.link, edge.frame) for edge in edges}
    # Freeze evaluation queries from accepted pre-pruning evidence. A failed or
    # pruned reconstruction must not make its own ground-truth query disappear, and one
    # temporal support run must not fragment into many independently weighted queries.
    support_intervals = [
        {
            "link": run.link,
            "start": run.start,
            "end": run.end,
            "kind": "pelvis" if run.link == pelvis_link else "foot" if run.link in foot_links else "terrain",
            "evidence_stage": "accepted_before_edge_pruning",
            "surface_height_m": run.height,
            "height_source": (
                pelvis_support_height_source if run.link == pelvis_link else "link_trajectory_minus_fixed_offset"
            ),
        }
        for run in runs
    ]
    report = {
        "model": "voronoi",
        "method": "Voronoi",
        "references": [TIP_PAPER, SCENEBOT_PAPER],
        "input_stage": input_stage,
        "n_frames": len(joints),
        "fps": fps,
        "terrain_links": list(links),
        "contact_joints": list(foot_links),
        "pelvis_link": pelvis_link,
        "link_surface_offsets_m": offsets,
        "contact_source": contact_source,
        "pelvis_support_height_source": pelvis_support_height_source,
        "resolved_pelvis_support_heights_m": resolved_pelvis_heights,
        "contact_detection": contact_diagnostics,
        "n_contact_events": len(runs),
        "n_foot_contact_events": sum(run.link in foot_links for run in runs),
        "n_contact_edges": len(edges),
        **pruning,
        "contact_events": [
            {
                "link": run.link,
                "start": run.start,
                "end": run.end,
                "n_frames": run.weight,
                "n_edges_after_pruning": sum((run.link, frame) in surviving_frames for frame in run.active_frames),
                "centre_xy": run.centre_xy.tolist(),
                "mean_height": run.height,
            }
            for run in runs
        ],
        "support_intervals": support_intervals,
        "n_height_clusters": len(clusters),
        "merge_execution": merge_execution,
        "height_clusters": [
            {
                "index": cluster.index,
                "height": cluster.height,
                "height_min": min(float(edge.point[2]) for edge in cluster.edges),
                "height_max": max(float(edge.point[2]) for edge in cluster.edges),
                "n_events": len({(edge.link, edge.interval_start, edge.interval_end) for edge in cluster.edges}),
                "n_edges": len(cluster.edges),
                "links": sorted({edge.link for edge in cluster.edges}),
            }
            for cluster in clusters
        ],
        "n_preprune_candidate_cells": len(candidate_cells),
        "n_candidate_cells": len(proposals),
        **carve,
        "n_cells_with_nearby_collision_samples": len(collision_samples),
        "n_cells_rejected_by_final_carve": carve["n_cells_lowered"] + carve["n_cells_carved_to_floor"],
        "n_boxes": len(boxes),
        "collision_model": collision_model,
        "config": config_dict,
        "operations": paper_specified,
        "voronoi_parameters": tip_derived,
        "algorithm_details": algorithm_details,
        "warnings": warnings,
    }
    return terrain, report


__all__ = [
    "DEFAULT_VORONOI_TERRAIN_LINKS",
    "PelvisSupportHeightResolver",
    "VoronoiConfig",
    "fit_voronoi_terrain_from_motion",
]
