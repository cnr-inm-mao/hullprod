"""Experimental mass realizations of the cotangent mean-curvature vector.

These operators are isolated scientific comparison paths. They are not used by
the production metric pipeline and deliberately perform no smoothing,
filtering, remeshing, or winding repair.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh

from .mesh_ops import cotangent_laplacian, face_angles, vertex_areas

MassMethod = Literal[
    "barycentric",
    "mixed_voronoi",
    "fem_lumped",
    "fem_consistent",
]

__all__ = [
    "MassMethod",
    "consistent_p1_mass_matrix",
    "fem_lumped_areas",
    "mean_curvature_mass",
    "mean_curvature_normal_mass",
    "mixed_voronoi_areas",
]


def consistent_p1_mass_matrix(mesh: trimesh.Trimesh) -> sp.csr_matrix:
    """Assemble the scalar P1 surface mass matrix.

    Each triangle of area ``A`` contributes ``A/12 * [[2,1,1], [1,2,1],
    [1,1,2]]``. The same matrix applies independently to all three vector
    coordinates.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    areas = np.asarray(mesh.area_faces, dtype=float)
    n_vertices = len(mesh.vertices)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for local_i in range(3):
        for local_j in range(3):
            rows.append(faces[:, local_i])
            columns.append(faces[:, local_j])
            coefficient = 2.0 if local_i == local_j else 1.0
            values.append(coefficient * areas / 12.0)
    return sp.coo_matrix(
        (
            np.concatenate(values),
            (np.concatenate(rows), np.concatenate(columns)),
        ),
        shape=(n_vertices, n_vertices),
    ).tocsr()


def fem_lumped_areas(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return row-sum lumping of the consistent P1 mass matrix.

    This is identically the barycentric associated area: each incident
    triangle contributes one third of its area.
    """
    mass = consistent_p1_mass_matrix(mesh)
    return np.asarray(mass.sum(axis=1)).ravel()


def mixed_voronoi_areas(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return Meyer et al.'s mixed Voronoi areas, including obtuse handling.

    A non-obtuse triangle uses the circumcentric Voronoi subdivision. In an
    obtuse triangle, half its area is assigned to the obtuse vertex and one
    quarter to each other vertex, yielding a positive surface tiling.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    points = np.asarray(mesh.vertices, dtype=float)[faces]
    angles = face_angles(mesh)
    face_area = np.asarray(mesh.area_faces, dtype=float)
    result = np.zeros(len(mesh.vertices), dtype=float)
    obtuse_local = np.argmax(angles, axis=1)
    is_obtuse = np.max(angles, axis=1) > 0.5 * np.pi
    cotangents = 1.0 / np.tan(np.clip(angles, 1.0e-12, np.pi - 1.0e-12))

    for local_i in range(3):
        local_j = (local_i + 1) % 3
        local_k = (local_i + 2) % 3
        edge_ij2 = np.einsum(
            "ij,ij->i",
            points[:, local_i] - points[:, local_j],
            points[:, local_i] - points[:, local_j],
        )
        edge_ik2 = np.einsum(
            "ij,ij->i",
            points[:, local_i] - points[:, local_k],
            points[:, local_i] - points[:, local_k],
        )
        contribution = (cotangents[:, local_j] * edge_ik2 + cotangents[:, local_k] * edge_ij2) / 8.0
        contribution[is_obtuse & (obtuse_local == local_i)] = (
            face_area[is_obtuse & (obtuse_local == local_i)] / 2.0
        )
        contribution[is_obtuse & (obtuse_local != local_i)] = (
            face_area[is_obtuse & (obtuse_local != local_i)] / 4.0
        )
        np.add.at(result, faces[:, local_i], contribution)
    return result


def mean_curvature_normal_mass(
    mesh: trimesh.Trimesh,
    *,
    method: MassMethod,
) -> np.ndarray:
    """Return experimental conventional mean-curvature normal ``H n``.

    HullProd's positive cotangent stiffness matrix ``K`` represents the weak
    operator ``-Delta_s``. Since ``-Delta_s X = 2 H n`` under the convention
    ``H=(k1+k2)/2``, the discrete equation is ``M q = K X`` for
    ``q=2 H n``. This function always returns ``q/2``.
    """
    stiffness_load = np.asarray(cotangent_laplacian(mesh) @ mesh.vertices)
    if method == "barycentric":
        areas = vertex_areas(mesh)
        return stiffness_load / np.maximum(2.0 * areas[:, None], 1.0e-30)
    if method == "mixed_voronoi":
        areas = mixed_voronoi_areas(mesh)
        return stiffness_load / np.maximum(2.0 * areas[:, None], 1.0e-30)
    mass = consistent_p1_mass_matrix(mesh)
    if method == "fem_lumped":
        areas = np.asarray(mass.sum(axis=1)).ravel()
        return stiffness_load / np.maximum(2.0 * areas[:, None], 1.0e-30)
    if method == "fem_consistent":
        factor = spla.splu(mass.tocsc())
        return np.asarray(factor.solve(stiffness_load)) / 2.0
    raise ValueError(f"unknown mass method: {method}")


def mean_curvature_mass(
    mesh: trimesh.Trimesh,
    *,
    method: MassMethod,
    signed: bool = True,
) -> np.ndarray:
    """Project an experimental curvature-normal vector onto vertex normals.

    Trimesh's corner-angle-weighted vertex normals define the scalar sign.
    Consistent global orientation reversal therefore reverses signed H while
    preserving unsigned magnitude. Local winding is not repaired.
    """
    vector = mean_curvature_normal_mass(mesh, method=method)
    signed_values = np.einsum("ij,ij->i", vector, mesh.vertex_normals)
    return signed_values if signed else np.abs(signed_values)
