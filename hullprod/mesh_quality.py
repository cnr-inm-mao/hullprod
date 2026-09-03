"""Scale-aware triangulation diagnostics for curvature interpretation.

The reliability classification in this module is an advisory software
heuristic.  It is not a manufacturing threshold, a geometry acceptance
standard, or a substitute for mesh-convergence evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.csgraph import connected_components

from .mesh_ops import face_angles, vertex_areas

_HEURISTIC_THRESHOLDS = {
    "poor_triangle_quality_min_below": 1.0e-4,
    "caution_triangle_quality_p01_below": 5.0e-2,
    "poor_edge_min_to_median_below": 1.0e-5,
    "caution_edge_p01_to_median_below": 1.0e-3,
    "poor_area_min_to_median_below": 1.0e-6,
    "caution_area_p01_to_median_below": 1.0e-3,
    "poor_valence_max_above": 50,
    "caution_valence_max_above": 20,
}


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "minimum": float("nan"),
            "p01": float("nan"),
            "p05": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "maximum": float("nan"),
            "nonfinite_count": len(values),
        }
    return {
        "minimum": float(np.min(finite)),
        "p01": float(np.percentile(finite, 1.0)),
        "p05": float(np.percentile(finite, 5.0)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "p99": float(np.percentile(finite, 99.0)),
        "maximum": float(np.max(finite)),
        "nonfinite_count": int(len(values) - len(finite)),
    }


def _ratio(value: float, reference: float) -> float:
    return float(value / max(reference, 1.0e-30))


def _face_component_count(mesh: trimesh.Trimesh) -> int:
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return 0
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency) == 0:
        return n_faces
    rows = np.concatenate((adjacency[:, 0], adjacency[:, 1]))
    columns = np.concatenate((adjacency[:, 1], adjacency[:, 0]))
    graph = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(n_faces, n_faces),
    ).tocsr()
    count, _ = connected_components(graph, directed=False, return_labels=True)
    return int(count)


def _reason(
    code: str,
    severity: str,
    message: str,
    value: float | int,
    threshold: float | int | str,
) -> dict[str, str | float | int]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "value": value,
        "threshold": threshold,
    }


def analyze_mesh_quality(
    mesh: trimesh.Trimesh,
    areas: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return JSON-ready mesh-quality statistics and curvature advisories."""
    if areas is None:
        areas = vertex_areas(mesh)
    areas = np.asarray(areas, dtype=float)
    if areas.shape != (len(mesh.vertices),):
        raise ValueError("areas must contain one value per mesh vertex")

    unique_edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    edge_inverse = np.asarray(mesh.edges_unique_inverse, dtype=np.int64)
    if len(unique_edges):
        edge_vectors = mesh.vertices[unique_edges[:, 0]] - mesh.vertices[unique_edges[:, 1]]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        edge_counts = np.bincount(edge_inverse, minlength=len(unique_edges))
        valence = np.bincount(unique_edges.ravel(), minlength=len(mesh.vertices))
    else:
        edge_lengths = np.empty(0, dtype=float)
        edge_counts = np.empty(0, dtype=int)
        valence = np.zeros(len(mesh.vertices), dtype=int)

    angles_degrees = np.degrees(face_angles(mesh)) if len(mesh.faces) else np.empty((0, 3))
    minimum_face_angles = np.min(angles_degrees, axis=1) if len(angles_degrees) else np.empty(0)
    maximum_face_angles = np.max(angles_degrees, axis=1) if len(angles_degrees) else np.empty(0)

    if len(mesh.faces):
        face_edge_lengths = edge_lengths[edge_inverse].reshape((-1, 3))
        squared_sum = np.sum(face_edge_lengths**2, axis=1)
        face_areas = np.asarray(mesh.area_faces, dtype=float)
        normalized_quality = (
            4.0
            * np.sqrt(3.0)
            * face_areas
            / np.maximum(
                squared_sum,
                1.0e-30,
            )
        )
        longest_edge = np.max(face_edge_lengths, axis=1)
        aspect_ratio = longest_edge**2 / np.maximum(2.0 * face_areas, 1.0e-30)
    else:
        normalized_quality = np.empty(0)
        aspect_ratio = np.empty(0)
        face_areas = np.empty(0)

    edge_stats = _distribution(edge_lengths)
    area_stats = _distribution(areas)
    quality_stats = _distribution(normalized_quality)
    valence_stats = _distribution(valence)
    edge_median = float(edge_stats["median"])
    area_median = float(area_stats["median"])
    edge_ratios = {
        "minimum_to_median": _ratio(float(edge_stats["minimum"]), edge_median),
        "p01_to_median": _ratio(float(edge_stats["p01"]), edge_median),
        "p05_to_median": _ratio(float(edge_stats["p05"]), edge_median),
        "maximum_to_median": _ratio(float(edge_stats["maximum"]), edge_median),
    }
    area_ratios = {
        "minimum_to_median": _ratio(float(area_stats["minimum"]), area_median),
        "p01_to_median": _ratio(float(area_stats["p01"]), area_median),
        "p05_to_median": _ratio(float(area_stats["p05"]), area_median),
        "maximum_to_median": _ratio(float(area_stats["maximum"]), area_median),
    }

    boundary_edge_count = int(np.sum(edge_counts == 1))
    non_manifold_edge_count = int(np.sum(edge_counts > 2))
    boundary_vertex_mask = np.zeros(len(mesh.vertices), dtype=bool)
    if boundary_edge_count:
        boundary_vertex_mask[unique_edges[edge_counts == 1].ravel()] = True
    boundary_vertex_count = int(np.sum(boundary_vertex_mask))
    component_count = _face_component_count(mesh)

    reasons: list[dict[str, str | float | int]] = []
    if float(quality_stats["minimum"]) < _HEURISTIC_THRESHOLDS["poor_triangle_quality_min_below"]:
        reasons.append(
            _reason(
                "extreme_sliver_triangle",
                "poor",
                "At least one triangle has extremely low normalized shape quality.",
                float(quality_stats["minimum"]),
                _HEURISTIC_THRESHOLDS["poor_triangle_quality_min_below"],
            )
        )
    elif float(quality_stats["p01"]) < _HEURISTIC_THRESHOLDS["caution_triangle_quality_p01_below"]:
        reasons.append(
            _reason(
                "low_triangle_quality_tail",
                "caution",
                "The lowest one percent of triangles has low normalized shape quality.",
                float(quality_stats["p01"]),
                _HEURISTIC_THRESHOLDS["caution_triangle_quality_p01_below"],
            )
        )

    if edge_ratios["minimum_to_median"] < _HEURISTIC_THRESHOLDS["poor_edge_min_to_median_below"]:
        reasons.append(
            _reason(
                "extremely_short_edge",
                "poor",
                "The shortest edge is extreme relative to the median edge length.",
                edge_ratios["minimum_to_median"],
                _HEURISTIC_THRESHOLDS["poor_edge_min_to_median_below"],
            )
        )
    elif edge_ratios["p01_to_median"] < _HEURISTIC_THRESHOLDS["caution_edge_p01_to_median_below"]:
        reasons.append(
            _reason(
                "short_edge_tail",
                "caution",
                "The shortest one percent of edges is highly nonuniform.",
                edge_ratios["p01_to_median"],
                _HEURISTIC_THRESHOLDS["caution_edge_p01_to_median_below"],
            )
        )

    if area_ratios["minimum_to_median"] < _HEURISTIC_THRESHOLDS["poor_area_min_to_median_below"]:
        reasons.append(
            _reason(
                "extremely_small_associated_area",
                "poor",
                "A vertex-associated area is extreme relative to the median.",
                area_ratios["minimum_to_median"],
                _HEURISTIC_THRESHOLDS["poor_area_min_to_median_below"],
            )
        )
    elif area_ratios["p01_to_median"] < _HEURISTIC_THRESHOLDS["caution_area_p01_to_median_below"]:
        reasons.append(
            _reason(
                "small_associated_area_tail",
                "caution",
                "The lowest one percent of vertex-associated areas is highly nonuniform.",
                area_ratios["p01_to_median"],
                _HEURISTIC_THRESHOLDS["caution_area_p01_to_median_below"],
            )
        )

    maximum_valence = int(valence_stats["maximum"])
    if maximum_valence > _HEURISTIC_THRESHOLDS["poor_valence_max_above"]:
        reasons.append(
            _reason(
                "extreme_vertex_valence",
                "poor",
                "Maximum vertex valence is extreme for a triangular surface mesh.",
                maximum_valence,
                _HEURISTIC_THRESHOLDS["poor_valence_max_above"],
            )
        )
    elif maximum_valence > _HEURISTIC_THRESHOLDS["caution_valence_max_above"]:
        reasons.append(
            _reason(
                "high_vertex_valence",
                "caution",
                "Maximum vertex valence is unusually high.",
                maximum_valence,
                _HEURISTIC_THRESHOLDS["caution_valence_max_above"],
            )
        )

    if non_manifold_edge_count:
        reasons.append(
            _reason(
                "non_manifold_edges",
                "poor",
                "Non-manifold edges make local curvature neighborhoods ambiguous.",
                non_manifold_edge_count,
                0,
            )
        )
    if boundary_edge_count:
        reasons.append(
            _reason(
                "open_boundary",
                "caution",
                "Open boundaries require care when interpreting discrete curvature.",
                boundary_edge_count,
                "0 for a closed surface",
            )
        )
    if component_count > 1:
        reasons.append(
            _reason(
                "multiple_face_components",
                "caution",
                "The mesh contains multiple connected face components.",
                component_count,
                1,
            )
        )

    severity_rank = {"good": 0, "caution": 1, "poor": 2}
    status = "good"
    for reason in reasons:
        if severity_rank[str(reason["severity"])] > severity_rank[status]:
            status = str(reason["severity"])

    return {
        "schema_version": 1,
        "advisory_note": (
            "Software heuristics for curvature interpretation; not manufacturing "
            "thresholds or a universal mesh-acceptance standard."
        ),
        "triangle": {
            "minimum_angle_degrees": _distribution(minimum_face_angles),
            "maximum_angle_degrees": _distribution(maximum_face_angles),
            "normalized_shape_quality": {
                **quality_stats,
                "definition": "4*sqrt(3)*area/sum(edge_length^2); 1 is equilateral",
            },
            "longest_edge_to_shortest_altitude": {
                **_distribution(aspect_ratio),
                "definition": "longest_edge^2/(2*area); lower is better",
            },
            "degenerate_triangle_count": int(np.sum(face_areas <= 0.0)),
        },
        "edge": {
            "length": edge_stats,
            "dimensionless_ratios": edge_ratios,
        },
        "vertex": {
            "valence": {
                **valence_stats,
                "maximum_valence": maximum_valence,
            },
            "associated_area": area_stats,
            "associated_area_dimensionless_ratios": area_ratios,
        },
        "topology": {
            "unique_edge_count": len(unique_edges),
            "boundary_edge_count": boundary_edge_count,
            "boundary_vertex_count": boundary_vertex_count,
            "boundary_edge_fraction": _ratio(boundary_edge_count, len(unique_edges)),
            "boundary_vertex_fraction": _ratio(boundary_vertex_count, len(mesh.vertices)),
            "non_manifold_edge_count": non_manifold_edge_count,
            "non_manifold_edge_fraction": _ratio(
                non_manifold_edge_count,
                len(unique_edges),
            ),
            "connected_face_component_count": component_count,
            "is_watertight": bool(mesh.is_watertight),
        },
        "curvature_reliability": {
            "status": status,
            "reasons": reasons,
            "heuristic_thresholds": dict(_HEURISTIC_THRESHOLDS),
        },
    }
