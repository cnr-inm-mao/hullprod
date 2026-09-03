"""Experimental discrete-curvature estimators.

These functions are comparison paths for estimator research.  They do not
replace the canonical estimators in :mod:`hullprod.mesh_ops` and are not used
by :func:`hullprod.metrics.compute_metrics`.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .mesh_ops import (
    cotangent_laplacian,
    gaussian_curvature,
    vertex_areas,
)

__all__ = [
    "gaussian_curvature_boundary_interpolated",
    "gaussian_curvature_interior_only",
    "mean_curvature_legacy_v012",
    "mean_curvature_normal_vector",
    "mean_curvature_projected",
]


def _vertex_areas(mesh: trimesh.Trimesh, areas: np.ndarray | None) -> np.ndarray:
    if areas is None:
        return vertex_areas(mesh)
    values = np.asarray(areas, dtype=float)
    if values.shape != (len(mesh.vertices),):
        raise ValueError("areas must contain one value per mesh vertex")
    return values


def _boundary_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return a mask for vertices incident to a boundary edge."""
    edges = np.sort(np.asarray(mesh.edges, dtype=np.int64), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = np.zeros(len(mesh.vertices), dtype=bool)
    boundary[unique_edges[counts == 1].ravel()] = True
    return boundary


def mean_curvature_normal_vector(
    mesh: trimesh.Trimesh,
    areas: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate the vector whose magnitude is ``H=(k1+k2)/2``.

    HullProd's cotangent Laplacian stores half-cotangent edge weights.  With
    that convention, ``(L @ x) / (2 A)`` approximates the mean-curvature
    normal vector.  This is the vector formed inside the production estimator.
    """
    vertex_areas = _vertex_areas(mesh, areas)
    laplacian = cotangent_laplacian(mesh)
    denominator = np.maximum(2.0 * vertex_areas, 1.0e-30)
    return np.asarray(laplacian @ mesh.vertices) / denominator[:, None]


def mean_curvature_projected(
    mesh: trimesh.Trimesh,
    areas: np.ndarray | None = None,
    *,
    signed: bool = False,
) -> np.ndarray:
    """Project the conventional curvature vector onto the vertex normal.

    This removes tangential boundary forces, but it is an experimental
    boundary treatment rather than the production HullProd estimator.
    """
    vector = mean_curvature_normal_vector(mesh, areas)
    normal_component = np.einsum("ij,ij->i", vector, mesh.vertex_normals)
    return normal_component if signed else np.abs(normal_component)


def mean_curvature_legacy_v012(
    mesh: trimesh.Trimesh,
    areas: np.ndarray | None = None,
    *,
    signed: bool = False,
) -> np.ndarray:
    """Reproduce the factor-of-two-low v0.1.2 behavior for audits only.

    This diagnostic preserves historical benchmark provenance.  It is not a
    compatibility mode and is not used by the production metric pipeline.
    """
    vector = mean_curvature_normal_vector(mesh, areas)
    magnitude = np.linalg.norm(vector, axis=1)
    if not signed:
        return 0.5 * magnitude
    normal_component = np.einsum("ij,ij->i", vector, mesh.vertex_normals)
    orientation = np.sign(normal_component)
    orientation[orientation == 0.0] = 1.0
    return 0.5 * magnitude * orientation


def gaussian_curvature_interior_only(
    mesh: trimesh.Trimesh,
    areas: np.ndarray | None = None,
) -> np.ndarray:
    """Return angle-defect Gaussian curvature only at interior vertices.

    Boundary vertices are marked ``NaN``.  The production boundary defect
    ``pi-sum(theta)`` contains the geodesic turning term from Gauss--Bonnet and
    therefore is not interpreted here as local intrinsic surface curvature.
    """
    values = gaussian_curvature(mesh, _vertex_areas(mesh, areas))
    values[_boundary_vertices(mesh)] = np.nan
    return values


def gaussian_curvature_boundary_interpolated(
    mesh: trimesh.Trimesh,
    areas: np.ndarray | None = None,
) -> np.ndarray:
    """Extrapolate interior angle-defect curvature to boundary vertices.

    Boundary values are inverse-edge-length averages propagated from finite
    interior estimates.  This avoids treating boundary turning as surface
    Gaussian curvature.  It is a diagnostic extrapolation; meshes with no
    interior vertices retain ``NaN`` boundary values.
    """
    values = gaussian_curvature_interior_only(mesh, areas)
    boundary = ~np.isfinite(values)
    if not np.any(boundary):
        return values

    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    edge_vectors = mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    weights = 1.0 / np.maximum(edge_lengths, 1.0e-30)

    while np.any(boundary):
        finite = np.isfinite(values)
        sums = np.zeros(len(values), dtype=float)
        weight_sums = np.zeros(len(values), dtype=float)

        forward = boundary[edges[:, 0]] & finite[edges[:, 1]]
        np.add.at(
            sums,
            edges[forward, 0],
            values[edges[forward, 1]] * weights[forward],
        )
        np.add.at(weight_sums, edges[forward, 0], weights[forward])

        reverse = boundary[edges[:, 1]] & finite[edges[:, 0]]
        np.add.at(
            sums,
            edges[reverse, 1],
            values[edges[reverse, 0]] * weights[reverse],
        )
        np.add.at(weight_sums, edges[reverse, 1], weights[reverse])

        fill = boundary & (weight_sums > 0.0)
        if not np.any(fill):
            break
        values[fill] = sums[fill] / weight_sums[fill]
        boundary[fill] = False

    return values
