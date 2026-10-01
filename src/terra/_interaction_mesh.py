"""Internal interaction-mesh graph operations used by TERRA."""

from __future__ import annotations

import numpy as np


def _calculate_laplacian_coordinates(
    vertices: np.ndarray,
    adj_list,
    epsilon: float = 1e-6,
    uniform_weight: bool = True,
) -> np.ndarray:
    """Calculate Laplacian coordinates for interaction-mesh vertices.

    Args:
        vertices: Vertex positions with shape ``(N, 3)``.
        adj_list: Neighbor indices for each vertex.
        epsilon: Numerical offset for inverse-distance weights.
        uniform_weight: Whether to weight all neighbors equally.

    Returns:
        Laplacian coordinates with the same shape as ``vertices``.
    """
    laplacian = np.zeros_like(vertices)

    if uniform_weight:
        rows_by_degree: dict[int, list[int]] = {}
        for index, neighbors in enumerate(adj_list):
            if len(neighbors) > 0:
                rows_by_degree.setdefault(len(neighbors), []).append(index)
        for degree, indices in rows_by_degree.items():
            neighbor_indices = np.asarray([adj_list[index] for index in indices], dtype=np.intp)
            weighted_sum = np.sum(vertices[neighbor_indices], axis=1)
            laplacian[indices] = vertices[indices] - weighted_sum / float(degree)
        return laplacian

    for index, neighbors in enumerate(adj_list):
        if len(neighbors) == 0:
            continue
        vertex = vertices[index]
        neighbor_positions = vertices[neighbors]
        distances = np.linalg.norm(vertex - neighbor_positions, axis=1)
        weights = 1.0 / (1.5 * distances + epsilon)
        sum_of_weights = np.sum(weights)
        weighted_sum = np.sum(weights[:, np.newaxis] * neighbor_positions, axis=0)
        laplacian[index] = vertex - weighted_sum / sum_of_weights
    return laplacian


def _get_adjacency_list(tetrahedra: np.ndarray, num_vertices: int):
    """Build a vertex adjacency list from tetrahedra.

    Args:
        tetrahedra: Vertex indices with shape ``(M, 4)``.
        num_vertices: Total number of vertices.

    Returns:
        Neighbor-index lists for every vertex.
    """
    adjacency = [set() for _ in range(num_vertices)]
    for tetrahedron in tetrahedra:
        vertex0, vertex1, vertex2, vertex3 = tetrahedron
        for left, right in (
            (vertex0, vertex1),
            (vertex0, vertex2),
            (vertex0, vertex3),
            (vertex1, vertex2),
            (vertex1, vertex3),
            (vertex2, vertex3),
        ):
            adjacency[left].add(right)
            adjacency[right].add(left)
    return [list(neighbors) for neighbors in adjacency]
