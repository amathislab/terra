"""Height-field metrics for comparing TERRA boxes with PRISM object meshes.

The reconstruction target is the upward support envelope, not the hidden bottom
or vertical side faces of an object. PRISM is Z-up, so this module intersects
vertical rays with mesh triangles and keeps their highest intersection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _as_vertices(value: Any) -> np.ndarray:
    vertices = np.asarray(value, dtype=float)
    if vertices.ndim == 3 and vertices.shape[0] == 1:
        vertices = vertices[0]
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"mesh vertices must have shape (V, 3), got {vertices.shape}")
    return vertices


def mesh_triangles(objects: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    """Collect all valid triangles from a PRISM ``objects`` mapping."""
    triangles = []
    for obj in objects.values():
        vertices = _as_vertices(obj["vertices"])
        faces = np.asarray(obj["faces"], dtype=np.int64)
        if faces.ndim == 3 and faces.shape[0] == 1:
            faces = faces[0]
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"mesh faces must have shape (F, 3), got {faces.shape}")
        if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("mesh face index is outside the vertex array")
        triangles.append(vertices[faces])
    return np.concatenate(triangles, axis=0) if triangles else np.empty((0, 3, 3))


def mesh_height_at(
    xy: np.ndarray,
    objects: Mapping[str, Mapping[str, Any]],
    *,
    floor_z: float = 0.0,
    edge_tolerance: float = 1e-9,
) -> np.ndarray:
    """Return the highest PRISM mesh surface over each XY query, or ``floor_z``.

    Vertical faces have zero projected area and are intentionally ignored. Both
    windings are accepted; if top and bottom faces overlap in XY, the maximum Z
    intersection selects the usable upper surface.
    """
    points = np.asarray(xy, dtype=float)
    if points.shape[-1] != 2:
        raise ValueError(f"xy must end in dimension 2, got {points.shape}")
    shape = points.shape[:-1]
    points = points.reshape(-1, 2)
    heights = np.full(len(points), float(floor_z), dtype=float)

    for triangle in mesh_triangles(objects):
        a, b, c = triangle
        ab = b[:2] - a[:2]
        ac = c[:2] - a[:2]
        determinant = ab[0] * ac[1] - ab[1] * ac[0]
        if abs(determinant) <= edge_tolerance:
            continue

        lo = np.minimum.reduce(triangle[:, :2]) - edge_tolerance
        hi = np.maximum.reduce(triangle[:, :2]) + edge_tolerance
        candidates = np.flatnonzero(
            (points[:, 0] >= lo[0]) & (points[:, 0] <= hi[0]) & (points[:, 1] >= lo[1]) & (points[:, 1] <= hi[1])
        )
        if not len(candidates):
            continue

        q = points[candidates] - a[:2]
        weight_b = (q[:, 0] * ac[1] - ac[0] * q[:, 1]) / determinant
        weight_c = (ab[0] * q[:, 1] - q[:, 0] * ab[1]) / determinant
        inside = (
            (weight_b >= -edge_tolerance)
            & (weight_c >= -edge_tolerance)
            & (weight_b + weight_c <= 1.0 + edge_tolerance)
        )
        selected = candidates[inside]
        if not len(selected):
            continue
        z = a[2] + weight_b[inside] * (b[2] - a[2]) + weight_c[inside] * (c[2] - a[2])
        heights[selected] = np.maximum(heights[selected], z)
    return heights.reshape(shape)


def _terrain_bounds(terrain: Any) -> np.ndarray:
    corners = []
    for box in terrain.boxes:
        local = np.array(
            [
                [u, v, w]
                for u in (-box.size[0], box.size[0])
                for v in (-box.size[1], box.size[1])
                for w in (-box.size[2], box.size[2])
            ]
        )
        corners.append(local @ box.rotation.T + np.asarray(box.pos))
    return np.concatenate(corners, axis=0) if corners else np.empty((0, 3))


def evaluation_grid(
    objects: Mapping[str, Mapping[str, Any]],
    terrain: Any,
    *,
    resolution: float = 0.02,
    margin: float = 0.04,
    domain_xy: np.ndarray | None = None,
    max_points: int = 2_000_000,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a deterministic XY grid for the requested evaluation domain.

    Without ``domain_xy``, the legacy domain covers the joint GT/prediction bounds.
    With ``domain_xy``, prediction bounds are ignored and the grid covers the GT mesh
    plus those independently supplied workspace points. This gives every method the
    same finite flat-ground denominator for a given take.
    """
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and positive")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")
    vertices = [_as_vertices(obj["vertices"]) for obj in objects.values()]
    if domain_xy is None:
        predicted = _terrain_bounds(terrain)
        all_points = [array[:, :2] for array in (*vertices, predicted) if len(array)]
    else:
        domain = np.asarray(domain_xy, dtype=float).reshape(-1, 2)
        if len(domain) and not np.all(np.isfinite(domain)):
            raise ValueError("domain_xy must contain only finite coordinates")
        all_points = [array[:, :2] for array in vertices if len(array)]
        if len(domain):
            all_points.append(domain)
    if not all_points:
        return np.empty((0, 2)), (0, 0)
    points = np.concatenate(all_points, axis=0)
    lo = points.min(axis=0) - margin
    hi = points.max(axis=0) + margin
    # Sample cell centres. Sampling exact polygon edges makes a zero-area boundary carry
    # finite grid weight and lets harmless inclusive/exclusive conventions change IoU.
    axes = [
        np.arange(
            np.floor(lo[i] / resolution) * resolution + 0.5 * resolution,
            np.ceil(hi[i] / resolution) * resolution,
            resolution,
        )
        for i in range(2)
    ]
    count = len(axes[0]) * len(axes[1])
    if count > max_points:
        raise ValueError(f"evaluation grid would contain {count:,} points; increase resolution or max_points")
    gx, gy = np.meshgrid(*axes, indexing="xy")
    return np.column_stack((gx.ravel(), gy.ravel())), gx.shape


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _percentile(values: np.ndarray, q: float) -> float | None:
    return None if not len(values) else float(np.percentile(values, q))


def coordinate_audit(
    take: Mapping[str, Any],
    objects: Mapping[str, Mapping[str, Any]],
    *,
    raised_epsilon: float = 0.02,
) -> dict[str, Any]:
    """Check PRISM CoP Z against the reference mesh without fitting an alignment.

    The audit is deliberately diagnostic: PRISM's insole CoP and object surface use
    slightly different physical conventions.  Reconstruction-to-mesh scores therefore
    use the mesh surface itself and never subtract the residual measured here.
    """
    residuals = []
    for side in ("L_Foot", "R_Foot"):
        foot = take["insole"][side]
        contacts = np.any(np.asarray(foot["contacts"], dtype=bool), axis=1)
        cop = np.asarray(foot["CoP_world"], dtype=float)
        if cop.shape != (len(contacts), 3):
            raise ValueError(f"{side} CoP_world must have shape ({len(contacts)}, 3), got {cop.shape}")
        valid = contacts & np.all(np.isfinite(cop), axis=1)
        points = cop[valid]
        gt = mesh_height_at(points[:, :2], objects)
        raised = gt > raised_epsilon
        residuals.extend((points[raised, 2] - gt[raised]).tolist())

    values = np.asarray(residuals, dtype=float)
    bias = float(np.median(values)) if len(values) else None
    centered = values - bias if bias is not None else values
    warning = None
    if not len(values):
        warning = "no raised insole contacts found; inspect seated-support scoring"
    elif np.percentile(np.abs(centered), 95) > 0.03 or abs(bias) > 0.10:
        warning = "possible coordinate-frame or CoP convention mismatch"
    return {
        "raised_contact_frames": len(values),
        "cop_to_mesh_mae_m": float(np.mean(np.abs(values))) if len(values) else None,
        "cop_to_mesh_p95_m": _percentile(np.abs(values), 95),
        "cop_to_mesh_bias_m": float(np.mean(values)) if len(values) else None,
        "cop_to_mesh_median_bias_m": bias,
        "cop_to_mesh_centered_p95_m": _percentile(np.abs(centered), 95),
        "warning": warning,
    }


def score_height_fields(
    objects: Mapping[str, Mapping[str, Any]],
    terrain: Any,
    *,
    resolution: float = 0.02,
    floor_z: float = 0.0,
    raised_epsilon: float = 0.02,
    evaluation_domain_xy: np.ndarray | None = None,
    evaluation_margin: float = 0.04,
    tolerances: Sequence[float] = (0.02, 0.05),
) -> dict[str, Any]:
    """Compare the full raised support envelopes on a regular XY grid."""
    xy, grid_shape = evaluation_grid(
        objects,
        terrain,
        resolution=resolution,
        margin=evaluation_margin,
        domain_xy=evaluation_domain_xy,
    )
    if not len(xy):
        return {"grid_shape": grid_shape, "grid_points": 0, "has_geometry": False}

    gt = mesh_height_at(xy, objects, floor_z=floor_z)
    pred = np.asarray(terrain.height_at(xy[:, 0], xy[:, 1]), dtype=float)
    gt_mask = gt > floor_z + raised_epsilon
    pred_mask = pred > floor_z + raised_epsilon
    gt_flat_mask = ~gt_mask
    pred_flat_mask = ~pred_mask
    union = gt_mask | pred_mask
    overlap = gt_mask & pred_mask
    flat_overlap = gt_flat_mask & pred_flat_mask
    error = np.abs(pred - gt)
    cell_area = resolution**2
    result: dict[str, Any] = {
        "has_geometry": bool(np.any(union)),
        "resolution_m": float(resolution),
        "grid_shape": [int(v) for v in grid_shape],
        "grid_points": len(xy),
        "evaluation_area_m2": float(len(xy) * cell_area),
        "gt_raised_area_m2": float(gt_mask.sum() * cell_area),
        "pred_raised_area_m2": float(pred_mask.sum() * cell_area),
        "raised_union_area_m2": float(union.sum() * cell_area),
        "gt_raised_cells": int(gt_mask.sum()),
        "pred_raised_cells": int(pred_mask.sum()),
        "gt_flat_cells": int(gt_flat_mask.sum()),
        "pred_flat_cells": int(pred_flat_mask.sum()),
        "raised_union_cells": int(union.sum()),
        "raised_overlap_cells": int(overlap.sum()),
        "flat_overlap_cells": int(flat_overlap.sum()),
        "raised_footprint_iou": _safe_ratio(int(overlap.sum()), int(union.sum())),
        "raised_terrain_coverage": _safe_ratio(int(overlap.sum()), int(gt_mask.sum())),
        "flat_terrain_coverage": _safe_ratio(int(flat_overlap.sum()), int(gt_flat_mask.sum())),
        "height_mae_union_m": float(error[union].mean()) if np.any(union) else None,
        "height_abs_error_sum_union_m": float(error[union].sum()) if np.any(union) else 0.0,
        "height_rmse_union_m": (float(np.sqrt(np.mean(np.square(error[union])))) if np.any(union) else None),
        "height_sq_error_sum_union_m2": (float(np.square(error[union]).sum()) if np.any(union) else 0.0),
        "height_p95_union_m": _percentile(error[union], 95),
        "height_mae_gt_footprint_m": float(error[gt_mask].mean()) if np.any(gt_mask) else None,
        "height_p95_gt_footprint_m": _percentile(error[gt_mask], 95),
        "height_bias_gt_footprint_m": (float((pred[gt_mask] - gt[gt_mask]).mean()) if np.any(gt_mask) else None),
    }
    for tolerance in tolerances:
        matched = overlap & (error <= tolerance)
        precision = _safe_ratio(int(matched.sum()), int(pred_mask.sum()))
        recall = _safe_ratio(int(matched.sum()), int(gt_mask.sum()))
        f1 = (
            None
            if precision is None or recall is None or precision + recall == 0
            else float(2 * precision * recall / (precision + recall))
        )
        label = f"{round(tolerance * 1000):d}mm"
        result[f"support_matched_cells_{label}"] = int(matched.sum())
        result[f"support_precision_{label}"] = precision
        result[f"support_recall_{label}"] = recall
        result[f"support_f1_{label}"] = f1
    return result


def score_observed_support(
    xy: np.ndarray,
    objects: Mapping[str, Mapping[str, Any]],
    terrain: Any,
    *,
    floor_z: float = 0.0,
    raised_epsilon: float = 0.02,
) -> dict[str, Any]:
    """Score predicted-vs-GT height only at independently measured support points."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if not len(xy):
        return {"points": 0, "raised_points": 0}
    gt = mesh_height_at(xy, objects, floor_z=floor_z)
    pred = np.asarray(terrain.height_at(xy[:, 0], xy[:, 1]), dtype=float)
    raised = gt > floor_z + raised_epsilon
    error = np.abs(pred[raised] - gt[raised])
    covered = pred[raised] > floor_z + raised_epsilon
    return {
        "points": len(xy),
        "raised_points": int(raised.sum()),
        "covered_raised_points": int(covered.sum()),
        "raised_coverage": (None if not np.any(raised) else float(np.mean(covered))),
        "height_mae_m": float(error.mean()) if len(error) else None,
        "height_abs_error_sum_m": float(error.sum()),
        "height_sq_error_sum_m2": float(np.square(error).sum()),
        "height_p95_m": _percentile(error, 95),
        "height_p99_m": _percentile(error, 99),
        "height_max_m": float(error.max()) if len(error) else None,
        "height_bias_m": float((pred[raised] - gt[raised]).mean()) if len(error) else None,
        "within_20mm_count": int(np.sum(error <= 0.02 + 1e-12)),
        "within_50mm_count": int(np.sum(error <= 0.05 + 1e-12)),
        "within_20mm": float(np.mean(error <= 0.02 + 1e-12)) if len(error) else None,
        "within_50mm": float(np.mean(error <= 0.05 + 1e-12)) if len(error) else None,
    }
