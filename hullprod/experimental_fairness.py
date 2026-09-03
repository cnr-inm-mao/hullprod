"""Fairness discretizations retained for scientific comparison.

F2 is the validated P1 face-gradient realization used by the paper release;
F0 and F1 remain here to preserve the audit paths and equivalence evidence.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .mesh_ops import cotangent_laplacian, vertex_areas

__all__ = [
    "dirichlet_energy_cotangent",
    "dirichlet_energy_edge_average",
    "dirichlet_energy_face_gradient",
    "edge_average_vertex_contributions",
    "face_gradient_contributions",
    "normalized_fairness",
]


def _field(mesh: trimesh.Trimesh, values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (len(mesh.vertices),):
        raise ValueError("values must contain one scalar per mesh vertex")
    if not np.all(np.isfinite(result)):
        raise ValueError("values must be finite")
    return result


def edge_average_vertex_contributions(
    mesh: trimesh.Trimesh,
    values: np.ndarray,
    *,
    areas: np.ndarray | None = None,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Return F0 vertex contributions before reference-length normalization.

    For vertex ``i``, the current HullProd discretization is

    ``A_i / degree(i) * sum_j ((f_i-f_j)/|x_i-x_j|)^2``.

    The equal directional average is dimensionally correct but is not the P1
    surface finite-element Dirichlet energy.
    """
    field = _field(mesh, values)
    vertex_area = vertex_areas(mesh) if areas is None else np.asarray(areas, dtype=float)
    if vertex_area.shape != field.shape:
        raise ValueError("areas must contain one value per mesh vertex")
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if len(edges) == 0:
        return np.full(len(field), np.nan)
    lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
    valid = np.isfinite(lengths) & (lengths > eps)
    if not np.any(valid):
        return np.full(len(field), np.nan)
    edges = edges[valid]
    directional2 = ((field[edges[:, 0]] - field[edges[:, 1]]) / lengths[valid]) ** 2
    sums = np.zeros(len(field), dtype=float)
    counts = np.zeros(len(field), dtype=float)
    np.add.at(sums, edges[:, 0], directional2)
    np.add.at(sums, edges[:, 1], directional2)
    np.add.at(counts, edges[:, 0], 1.0)
    np.add.at(counts, edges[:, 1], 1.0)
    return vertex_area * sums / np.maximum(counts, 1.0)


def dirichlet_energy_edge_average(
    mesh: trimesh.Trimesh,
    values: np.ndarray,
    *,
    areas: np.ndarray | None = None,
    eps: float = 1.0e-12,
) -> float:
    """Return the raw F0 edge-average energy used by production fairness."""
    return float(
        np.nansum(
            edge_average_vertex_contributions(
                mesh,
                values,
                areas=areas,
                eps=eps,
            )
        )
    )


def dirichlet_energy_cotangent(
    mesh: trimesh.Trimesh,
    values: np.ndarray,
) -> float:
    """Return F1, the P1 FEM/cotangent scalar Dirichlet energy.

    HullProd's cotangent matrix has per-face edge weights ``cot(theta)/2``.
    It is therefore the FEM stiffness matrix itself and ``f.T @ L @ f``
    equals ``integral |grad_s f_h|^2 dA`` with no additional factor.
    """
    field = _field(mesh, values)
    laplacian = cotangent_laplacian(mesh)
    value = float(field @ (laplacian @ field))
    # Sparse summation roundoff can produce a tiny negative value for a
    # constant field. Preserve material negative values as defect evidence.
    if value < 0.0 and abs(value) <= 1.0e-12 * max(float(field @ field), 1.0):
        return 0.0
    return value


def face_gradient_contributions(
    mesh: trimesh.Trimesh,
    values: np.ndarray,
) -> np.ndarray:
    """Return F2 per-face P1 energies ``A_f * |grad_f f_h|^2``."""
    field = _field(mesh, values)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return np.empty(0, dtype=float)
    points = mesh.vertices[faces]
    e01 = points[:, 1] - points[:, 0]
    e02 = points[:, 2] - points[:, 0]
    normal2 = np.cross(e01, e02)
    norm2_squared = np.einsum("ij,ij->i", normal2, normal2)
    if np.any(norm2_squared <= 1.0e-30):
        raise ValueError("face-gradient energy is undefined on degenerate triangles")

    grad_lambda0 = np.cross(normal2, points[:, 2] - points[:, 1])
    grad_lambda1 = np.cross(normal2, points[:, 0] - points[:, 2])
    grad_lambda2 = np.cross(normal2, points[:, 1] - points[:, 0])
    gradients = (
        field[faces[:, 0], None] * grad_lambda0
        + field[faces[:, 1], None] * grad_lambda1
        + field[faces[:, 2], None] * grad_lambda2
    ) / norm2_squared[:, None]
    face_area = 0.5 * np.sqrt(norm2_squared)
    return face_area * np.einsum("ij,ij->i", gradients, gradients)


def dirichlet_energy_face_gradient(
    mesh: trimesh.Trimesh,
    values: np.ndarray,
    *,
    face_mask: np.ndarray | None = None,
) -> float:
    """Return F2 by explicitly integrating the P1 gradient on each face."""
    contributions = face_gradient_contributions(mesh, values)
    if face_mask is not None:
        mask = np.asarray(face_mask, dtype=bool)
        if mask.shape != (len(mesh.faces),):
            raise ValueError("face_mask must contain one value per mesh face")
        contributions = contributions[mask]
    return float(np.sum(contributions))


def normalized_fairness(
    dirichlet_energy: float,
    *,
    length_ref: float,
    area_norm: float,
) -> float:
    """Apply the unchanged canonical normalization ``L_ref^4 / A``."""
    if length_ref <= 0.0 or area_norm <= 0.0:
        raise ValueError("length_ref and area_norm must be positive")
    return float(length_ref**4 * dirichlet_energy / area_norm)
