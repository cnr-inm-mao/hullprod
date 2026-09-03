"""Literature-standard curvature estimators for isolated benchmark studies.

Nothing in this module is called by the production metric pipeline.  The
implementations deliberately do not smooth geometry, repair winding, clip
fields, or choose parameters from HullProd benchmark outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh

from .experimental_mass_curvature import consistent_p1_mass_matrix
from .mesh_ops import cotangent_laplacian

NeighborhoodPolicy = Literal["minimum", "two_ring", "three_ring"]


@dataclass(frozen=True)
class DifferentialEstimate:
    """Pointwise differential quantities in the mesh orientation convention."""

    mean: np.ndarray
    gaussian: np.ndarray
    gradient_mean: np.ndarray | None
    valid: np.ndarray
    condition_number: np.ndarray | None = None


def max_weighted_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return Max (1999) normals as used by Rusinkiewicz's ``trimesh2``."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    points = np.asarray(mesh.vertices, dtype=float)[faces]
    a = points[:, 0] - points[:, 1]
    b = points[:, 1] - points[:, 2]
    c = points[:, 2] - points[:, 0]
    cross = np.cross(a, b)
    lengths2 = np.stack(
        (
            np.einsum("ij,ij->i", a, a),
            np.einsum("ij,ij->i", b, b),
            np.einsum("ij,ij->i", c, c),
        ),
        axis=1,
    )
    normals = np.zeros_like(mesh.vertices, dtype=float)
    denominators = (
        lengths2[:, 0] * lengths2[:, 2],
        lengths2[:, 1] * lengths2[:, 0],
        lengths2[:, 2] * lengths2[:, 1],
    )
    for local, denominator in enumerate(denominators):
        weight = 1.0 / np.maximum(denominator, 1.0e-30)
        np.add.at(normals, faces[:, local], cross * weight[:, None])
    norms = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(norms[:, None], 1.0e-30)
    return normals


def _restricted_voronoi_corner_areas(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Port the restricted Voronoi corner areas used by ``trimesh2``."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    points = np.asarray(mesh.vertices, dtype=float)[faces]
    edges = np.stack(
        (
            points[:, 2] - points[:, 1],
            points[:, 0] - points[:, 2],
            points[:, 1] - points[:, 0],
        ),
        axis=1,
    )
    area = 0.5 * np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1)
    length2 = np.einsum("fij,fij->fi", edges, edges)
    bcw = np.column_stack(
        (
            length2[:, 0] * (length2[:, 1] + length2[:, 2] - length2[:, 0]),
            length2[:, 1] * (length2[:, 2] + length2[:, 0] - length2[:, 1]),
            length2[:, 2] * (length2[:, 0] + length2[:, 1] - length2[:, 2]),
        )
    )
    corner = np.zeros((len(faces), 3), dtype=float)
    non_obtuse = np.all(bcw > 0.0, axis=1)
    scale = 0.5 * area / np.maximum(np.sum(bcw, axis=1), 1.0e-30)
    for local in range(3):
        corner[non_obtuse, local] = scale[non_obtuse] * (
            bcw[non_obtuse, (local + 1) % 3] + bcw[non_obtuse, (local + 2) % 3]
        )
    for obtuse in range(3):
        selected = (~non_obtuse) & (bcw[:, obtuse] <= 0.0)
        if not np.any(selected):
            continue
        first = (obtuse + 1) % 3
        second = (obtuse + 2) % 3
        corner[selected, first] = (
            -0.25
            * length2[selected, second]
            * area[selected]
            / np.minimum(
                np.einsum("ij,ij->i", edges[selected, obtuse], edges[selected, second]),
                -1.0e-30,
            )
        )
        corner[selected, second] = (
            -0.25
            * length2[selected, first]
            * area[selected]
            / np.minimum(
                np.einsum("ij,ij->i", edges[selected, obtuse], edges[selected, first]),
                -1.0e-30,
            )
        )
        corner[selected, obtuse] = area[selected] - (
            corner[selected, first] + corner[selected, second]
        )
    point = np.zeros(len(mesh.vertices), dtype=float)
    for local in range(3):
        np.add.at(point, faces[:, local], corner[:, local])
    return corner, point


def _rotate_basis(
    old_u: np.ndarray,
    old_v: np.ndarray,
    new_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    old_normal = np.cross(old_u, old_v)
    dot = float(np.dot(old_normal, new_normal))
    if dot <= -1.0 + 1.0e-14:
        return -old_u, -old_v
    perpendicular = new_normal - dot * old_normal
    delta = (old_normal + new_normal) / max(1.0 + dot, 1.0e-30)
    return (
        old_u - delta * np.dot(old_u, perpendicular),
        old_v - delta * np.dot(old_v, perpendicular),
    )


def _project_curvature(
    old_u: np.ndarray,
    old_v: np.ndarray,
    tensor: np.ndarray,
    new_u: np.ndarray,
    new_v: np.ndarray,
) -> np.ndarray:
    rotated_u, rotated_v = _rotate_basis(new_u, new_v, np.cross(old_u, old_v))
    transform = np.array(
        [
            [np.dot(rotated_u, old_u), np.dot(rotated_u, old_v)],
            [np.dot(rotated_v, old_u), np.dot(rotated_v, old_v)],
        ]
    )
    matrix = np.array([[tensor[0], tensor[1]], [tensor[1], tensor[2]]])
    projected = transform @ matrix @ transform.T
    return np.array([projected[0, 0], projected[0, 1], projected[1, 1]])


def _project_cubic(
    old_u: np.ndarray,
    old_v: np.ndarray,
    tensor: np.ndarray,
    new_u: np.ndarray,
    new_v: np.ndarray,
) -> np.ndarray:
    rotated_u, rotated_v = _rotate_basis(new_u, new_v, np.cross(old_u, old_v))
    u1, v1 = np.dot(rotated_u, old_u), np.dot(rotated_u, old_v)
    u2, v2 = np.dot(rotated_v, old_u), np.dot(rotated_v, old_v)
    a, b, c, d = tensor
    return np.array(
        [
            a * u1**3 + 3 * b * u1**2 * v1 + 3 * c * u1 * v1**2 + d * v1**3,
            a * u1**2 * u2
            + b * (u1**2 * v2 + 2 * u2 * u1 * v1)
            + c * (u2 * v1**2 + 2 * u1 * v1 * v2)
            + d * v1**2 * v2,
            a * u1 * u2**2
            + b * (u2**2 * v1 + 2 * u1 * u2 * v2)
            + c * (u1 * v2**2 + 2 * u2 * v2 * v1)
            + d * v1 * v2**2,
            a * u2**3 + 3 * b * u2**2 * v2 + 3 * c * u2 * v2**2 + d * v2**3,
        ]
    )


def _rusinkiewicz_tensor_vectorized(mesh: trimesh.Trimesh) -> DifferentialEstimate:
    """Vectorized curvature-only form of the official tensor-recovery algebra."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)
    normals = max_weighted_vertex_normals(mesh)
    corner_area, point_area = _restricted_voronoi_corner_areas(mesh)
    vertex_count = len(vertices)

    candidates = np.stack(
        (
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 1]],
            vertices[faces[:, 0]] - vertices[faces[:, 2]],
        ),
        axis=1,
    )
    vertex_ids = faces.ravel()
    candidate_flat = candidates.reshape(-1, 3)
    reversed_ids = vertex_ids[::-1]
    unique_ids, reversed_first = np.unique(reversed_ids, return_index=True)
    direction_u = np.zeros_like(vertices)
    direction_u[unique_ids] = candidate_flat[::-1][reversed_first]
    direction_u = np.cross(direction_u, normals)
    direction_u /= np.maximum(np.linalg.norm(direction_u, axis=1)[:, None], 1.0e-30)
    direction_v = np.cross(normals, direction_u)

    points = vertices[faces]
    edge = np.stack(
        (points[:, 2] - points[:, 1], points[:, 0] - points[:, 2], points[:, 1] - points[:, 0]),
        axis=1,
    )
    tangent = edge[:, 0] / np.maximum(np.linalg.norm(edge[:, 0], axis=1)[:, None], 1.0e-30)
    binormal = np.cross(np.cross(edge[:, 0], edge[:, 1]), tangent)
    binormal /= np.maximum(np.linalg.norm(binormal, axis=1)[:, None], 1.0e-30)
    u = np.einsum("fli,fi->fl", edge, tangent)
    v = np.einsum("fli,fi->fl", edge, binormal)

    matrix = np.zeros((len(faces), 3, 3), dtype=float)
    matrix[:, 0, 0] = np.sum(u * u, axis=1)
    matrix[:, 0, 1] = np.sum(u * v, axis=1)
    matrix[:, 2, 2] = np.sum(v * v, axis=1)
    matrix[:, 1, 0] = matrix[:, 0, 1]
    matrix[:, 1, 1] = matrix[:, 0, 0] + matrix[:, 2, 2]
    matrix[:, 1, 2] = matrix[:, 0, 1]
    matrix[:, 2, 1] = matrix[:, 1, 2]

    right = np.zeros((len(faces), 3), dtype=float)
    for local in range(3):
        difference = normals[faces[:, (local - 1) % 3]] - normals[faces[:, (local + 1) % 3]]
        dnu = np.einsum("fi,fi->f", difference, tangent)
        dnv = np.einsum("fi,fi->f", difference, binormal)
        right[:, 0] += dnu * u[:, local]
        right[:, 1] += dnu * v[:, local] + dnv * u[:, local]
        right[:, 2] += dnv * v[:, local]

    determinant = np.linalg.det(matrix)
    regular = np.isfinite(determinant) & (np.abs(determinant) > np.finfo(float).tiny)
    fitted = np.zeros((len(faces), 3), dtype=float)
    fitted[regular] = np.linalg.solve(matrix[regular], right[regular, :, None]).squeeze(-1)
    valid = np.ones(vertex_count, dtype=bool)
    if np.any(~regular):
        valid[np.unique(faces[~regular])] = False

    ku = np.zeros(vertex_count)
    kuv = np.zeros(vertex_count)
    kv = np.zeros(vertex_count)
    face_normal = np.cross(tangent, binormal)
    a, b, c = fitted.T
    for local in range(3):
        vertex = faces[:, local]
        old_u = direction_u[vertex]
        old_v = direction_v[vertex]
        old_normal = normals[vertex]
        dot = np.einsum("fi,fi->f", old_normal, face_normal)
        antipodal = dot <= -1.0 + 1.0e-14
        perpendicular = face_normal - dot[:, None] * old_normal
        delta = (old_normal + face_normal) / np.maximum((1.0 + dot)[:, None], 1.0e-30)
        rotated_u = old_u - delta * np.einsum("fi,fi->f", old_u, perpendicular)[:, None]
        rotated_v = old_v - delta * np.einsum("fi,fi->f", old_v, perpendicular)[:, None]
        rotated_u[antipodal] = -old_u[antipodal]
        rotated_v[antipodal] = -old_v[antipodal]
        t00 = np.einsum("fi,fi->f", rotated_u, tangent)
        t01 = np.einsum("fi,fi->f", rotated_u, binormal)
        t10 = np.einsum("fi,fi->f", rotated_v, tangent)
        t11 = np.einsum("fi,fi->f", rotated_v, binormal)
        projected_0 = t00 * t00 * a + 2.0 * t00 * t01 * b + t01 * t01 * c
        projected_1 = t00 * t10 * a + (t00 * t11 + t01 * t10) * b + t01 * t11 * c
        projected_2 = t10 * t10 * a + 2.0 * t10 * t11 * b + t11 * t11 * c
        weight = corner_area[:, local] / np.maximum(point_area[vertex], 1.0e-30)
        np.add.at(ku, vertex, weight * projected_0)
        np.add.at(kuv, vertex, weight * projected_1)
        np.add.at(kv, vertex, weight * projected_2)

    tensor = np.empty((vertex_count, 2, 2), dtype=float)
    tensor[:, 0, 0], tensor[:, 0, 1] = ku, kuv
    tensor[:, 1, 0], tensor[:, 1, 1] = kuv, kv
    eigenvalues = np.linalg.eigvalsh(tensor)
    first_index = np.argmax(np.abs(eigenvalues), axis=1)
    first = eigenvalues[np.arange(vertex_count), first_index]
    second = eigenvalues[np.arange(vertex_count), 1 - first_index]
    mean = 0.5 * (first + second)
    gaussian = first * second
    valid &= np.isfinite(mean) & np.isfinite(gaussian)
    return DifferentialEstimate(mean, gaussian, None, valid)


def rusinkiewicz_curvature(
    mesh: trimesh.Trimesh,
    *,
    derivatives: bool = True,
) -> DifferentialEstimate:
    """Estimate curvature and its derivative using Rusinkiewicz (2004).

    This is a double-precision Python port of the author's official
    ``trimesh2`` reference implementation: Max normals, per-face normal-
    variation tensor fits, restricted Voronoi averaging, and the analogous
    per-face cubic derivative-tensor fit.  Set ``derivatives=False`` when only
    the curvature tensor is required; this skips the mathematically independent
    derivative recovery without changing the returned H or K fields.
    """
    if not derivatives:
        return _rusinkiewicz_tensor_vectorized(mesh)

    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)
    normals = max_weighted_vertex_normals(mesh)
    corner_area, point_area = _restricted_voronoi_corner_areas(mesh)
    vertex_count = len(vertices)

    direction_u = np.zeros_like(vertices)
    for face in faces:
        direction_u[face[0]] = vertices[face[1]] - vertices[face[0]]
        direction_u[face[1]] = vertices[face[2]] - vertices[face[1]]
        direction_u[face[2]] = vertices[face[0]] - vertices[face[2]]
    direction_u = np.cross(direction_u, normals)
    direction_u /= np.maximum(np.linalg.norm(direction_u, axis=1)[:, None], 1.0e-30)
    direction_v = np.cross(normals, direction_u)

    ku = np.zeros(vertex_count)
    kuv = np.zeros(vertex_count)
    kv = np.zeros(vertex_count)
    valid = np.ones(vertex_count, dtype=bool)
    for face_index, face in enumerate(faces):
        edge = np.array(
            [
                vertices[face[2]] - vertices[face[1]],
                vertices[face[0]] - vertices[face[2]],
                vertices[face[1]] - vertices[face[0]],
            ]
        )
        tangent = edge[0] / max(np.linalg.norm(edge[0]), 1.0e-30)
        binormal = np.cross(np.cross(edge[0], edge[1]), tangent)
        binormal /= max(np.linalg.norm(binormal), 1.0e-30)
        matrix = np.zeros((3, 3))
        right = np.zeros(3)
        for local in range(3):
            u = np.dot(edge[local], tangent)
            v = np.dot(edge[local], binormal)
            matrix[0, 0] += u * u
            matrix[0, 1] += u * v
            matrix[2, 2] += v * v
            dn = normals[face[(local - 1) % 3]] - normals[face[(local + 1) % 3]]
            dnu, dnv = np.dot(dn, tangent), np.dot(dn, binormal)
            right += (dnu * u, dnu * v + dnv * u, dnv * v)
        matrix[1, 0] = matrix[0, 1]
        matrix[1, 1] = matrix[0, 0] + matrix[2, 2]
        matrix[1, 2] = matrix[0, 1]
        matrix[2, 1] = matrix[1, 2]
        try:
            fitted = np.linalg.solve(matrix, right)
        except np.linalg.LinAlgError:
            valid[face] = False
            continue
        for local, vertex in enumerate(face):
            projected = _project_curvature(
                tangent,
                binormal,
                fitted,
                direction_u[vertex],
                direction_v[vertex],
            )
            weight = corner_area[face_index, local] / max(point_area[vertex], 1.0e-30)
            ku[vertex] += weight * projected[0]
            kuv[vertex] += weight * projected[1]
            kv[vertex] += weight * projected[2]

    principal_1 = np.zeros(vertex_count)
    principal_2 = np.zeros(vertex_count)
    for vertex in range(vertex_count):
        old_u, old_v = _rotate_basis(direction_u[vertex], direction_v[vertex], normals[vertex])
        eigenvalues, eigenvectors = np.linalg.eigh(
            np.array([[ku[vertex], kuv[vertex]], [kuv[vertex], kv[vertex]]])
        )
        order = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[order]
        vector = eigenvectors[:, order[0]]
        principal_1[vertex], principal_2[vertex] = eigenvalues
        direction_u[vertex] = vector[0] * old_u + vector[1] * old_v
        direction_u[vertex] /= max(np.linalg.norm(direction_u[vertex]), 1.0e-30)
        direction_v[vertex] = np.cross(normals[vertex], direction_u[vertex])

    mean = 0.5 * (principal_1 + principal_2)
    gaussian = principal_1 * principal_2
    curvature_valid = valid & np.isfinite(mean) & np.isfinite(gaussian)
    derivative = np.zeros((vertex_count, 4))
    for face_index, face in enumerate(faces):
        edge = np.array(
            [
                vertices[face[2]] - vertices[face[1]],
                vertices[face[0]] - vertices[face[2]],
                vertices[face[1]] - vertices[face[0]],
            ]
        )
        tangent = edge[0] / max(np.linalg.norm(edge[0]), 1.0e-30)
        binormal = np.cross(np.cross(edge[0], edge[1]), tangent)
        binormal /= max(np.linalg.norm(binormal), 1.0e-30)
        face_curvature = np.array(
            [
                _project_curvature(
                    direction_u[vertex],
                    direction_v[vertex],
                    np.array([principal_1[vertex], 0.0, principal_2[vertex]]),
                    tangent,
                    binormal,
                )
                for vertex in face
            ]
        )
        matrix = np.zeros((4, 4))
        right = np.zeros(4)
        for local in range(3):
            difference = face_curvature[(local - 1) % 3] - face_curvature[(local + 1) % 3]
            u = np.dot(edge[local], tangent)
            v = np.dot(edge[local], binormal)
            u2, v2, uv = u * u, v * v, u * v
            matrix[0, 0] += u2
            matrix[0, 1] += uv
            matrix[3, 3] += v2
            right += (
                u * difference[0],
                v * difference[0] + 2 * u * difference[1],
                2 * v * difference[1] + u * difference[2],
                v * difference[2],
            )
        matrix[1, 0] = matrix[0, 1]
        matrix[1, 1] = 2 * matrix[0, 0] + matrix[3, 3]
        matrix[1, 2] = 2 * matrix[0, 1]
        matrix[2, 1] = matrix[1, 2]
        matrix[2, 2] = matrix[0, 0] + 2 * matrix[3, 3]
        matrix[2, 3] = matrix[0, 1]
        matrix[3, 2] = matrix[2, 3]
        try:
            fitted = np.linalg.solve(matrix, right)
        except np.linalg.LinAlgError:
            valid[face] = False
            continue
        for local, vertex in enumerate(face):
            projected = _project_cubic(
                tangent,
                binormal,
                fitted,
                direction_u[vertex],
                direction_v[vertex],
            )
            weight = corner_area[face_index, local] / max(point_area[vertex], 1.0e-30)
            derivative[vertex] += weight * projected

    gradient = (
        0.5 * (derivative[:, 0] + derivative[:, 2])[:, None] * direction_u
        + 0.5 * (derivative[:, 1] + derivative[:, 3])[:, None] * direction_v
    )
    return DifferentialEstimate(
        mean=mean,
        gaussian=gaussian,
        gradient_mean=gradient,
        valid=curvature_valid & np.all(np.isfinite(gradient), axis=1),
    )


def _vertex_adjacency(mesh: trimesh.Trimesh) -> list[set[int]]:
    result = [set() for _ in range(len(mesh.vertices))]
    for first, second in np.asarray(mesh.edges_unique, dtype=np.int64):
        result[first].add(int(second))
        result[second].add(int(first))
    return result


def _ring(mesh_adjacency: list[set[int]], vertex: int, rings: int) -> np.ndarray:
    selected = {vertex}
    frontier = {vertex}
    for _ in range(rings):
        frontier = set().union(*(mesh_adjacency[item] for item in frontier)) - selected
        selected.update(frontier)
        if not frontier:
            break
    return np.asarray(sorted(selected), dtype=np.int64)


def _jet_monomials(degree: int) -> list[tuple[int, int]]:
    return [(total - j, j) for total in range(degree + 1) for j in range(total + 1)]


def osculating_jet_curvature(
    mesh: trimesh.Trimesh,
    *,
    degree: Literal[2, 3],
    neighborhood: NeighborhoodPolicy = "minimum",
) -> DifferentialEstimate:
    """Fit Cazals--Pouget PCA/SVD osculating jets at mesh vertices.

    ``minimum`` collects the smallest complete k-ring containing at least
    ``(degree+1)(degree+2)/2`` samples, matching CGAL's documented default
    mesh-neighborhood policy. Fixed two- and three-ring alternatives are
    exposed only for the predeclared sensitivity study.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    reference_normals = max_weighted_vertex_normals(mesh)
    adjacency = _vertex_adjacency(mesh)
    monomials = _jet_monomials(degree)
    required = len(monomials)
    mean = np.full(len(vertices), np.nan)
    gaussian = np.full(len(vertices), np.nan)
    gradient = np.full((len(vertices), 3), np.nan) if degree >= 3 else None
    condition = np.full(len(vertices), np.inf)
    valid = np.zeros(len(vertices), dtype=bool)

    for vertex, origin in enumerate(vertices):
        if neighborhood == "minimum":
            rings = 0
            sample = _ring(adjacency, vertex, rings)
            while len(sample) < required and rings < 12:
                rings += 1
                sample = _ring(adjacency, vertex, rings)
        else:
            rings = 2 if neighborhood == "two_ring" else 3
            sample = _ring(adjacency, vertex, rings)
        if len(sample) < required:
            continue
        offsets = vertices[sample] - origin
        centered = offsets - np.mean(offsets, axis=0)
        covariance = centered.T @ centered
        _, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, 0]
        tangent_x = eigenvectors[:, 2]
        if np.dot(normal, reference_normals[vertex]) < 0.0:
            normal = -normal
        tangent_y = np.cross(normal, tangent_x)
        tangent_y /= max(np.linalg.norm(tangent_y), 1.0e-30)
        coordinates = offsets @ np.column_stack((tangent_x, tangent_y, normal))
        x, y, z = coordinates.T
        scale = float(np.mean(np.sqrt(x * x + y * y)))
        if not scale > 1.0e-15:
            continue
        design = np.column_stack(
            [
                (x / scale) ** i * (y / scale) ** j / (factorial(i) * factorial(j))
                for i, j in monomials
            ]
        )
        coefficients_scaled, _, rank, singular = np.linalg.lstsq(design, z, rcond=None)
        if rank < required or singular[-1] <= 0.0:
            continue
        coefficients = {
            power: coefficients_scaled[index] / scale ** sum(power)
            for index, power in enumerate(monomials)
        }
        condition[vertex] = singular[0] / singular[-1]
        p = coefficients[(1, 0)]
        q = coefficients[(0, 1)]
        r = coefficients[(2, 0)]
        s = coefficients[(1, 1)]
        t = coefficients[(0, 2)]
        metric = 1.0 + p * p + q * q
        numerator = (1.0 + q * q) * r - 2.0 * p * q * s + (1.0 + p * p) * t
        # Cazals--Pouget use W=-I^-1 II.  With the fitting z-axis aligned to
        # the supplied outward normal, the signed conventional H is -N/(2W^3/2).
        mean[vertex] = -numerator / (2.0 * metric**1.5)
        gaussian[vertex] = (r * t - s * s) / metric**2

        if degree >= 3 and gradient is not None:
            u = coefficients[(3, 0)]
            v = coefficients[(2, 1)]
            w = coefficients[(1, 2)]
            z3 = coefficients[(0, 3)]
            numerator_x = (
                2 * q * s * r
                + (1 + q * q) * u
                - 2 * ((r * q + p * s) * s + p * q * v)
                + 2 * p * r * t
                + (1 + p * p) * w
            )
            numerator_y = (
                2 * q * t * r
                + (1 + q * q) * v
                - 2 * ((s * q + p * t) * s + p * q * w)
                + 2 * p * s * t
                + (1 + p * p) * z3
            )
            metric_x = 2 * p * r + 2 * q * s
            metric_y = 2 * p * s + 2 * q * t
            hx = -0.5 * (numerator_x / metric**1.5 - 1.5 * numerator * metric_x / metric**2.5)
            hy = -0.5 * (numerator_y / metric**1.5 - 1.5 * numerator * metric_y / metric**2.5)
            inverse_metric = np.array([[1 + q * q, -p * q], [-p * q, 1 + p * p]]) / metric
            contravariant = inverse_metric @ np.array([hx, hy])
            basis_x = tangent_x + p * normal
            basis_y = tangent_y + q * normal
            gradient[vertex] = contravariant[0] * basis_x + contravariant[1] * basis_y
        valid[vertex] = True

    return DifferentialEstimate(
        mean=mean,
        gaussian=gaussian,
        gradient_mean=gradient,
        valid=valid,
        condition_number=condition,
    )


def edge_jump_stabilization(mesh: trimesh.Trimesh) -> sp.csr_matrix:
    """Assemble Hansbo--Larson--Zahedi's meshed-surface edge-jump form.

    The mesh parameter is exactly the paper's ``h=N^-1/2``.  The published
    formulation is nondimensional; dimensional scaling is handled explicitly
    by :func:`stabilized_fem_mean_curvature`.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    points = np.asarray(mesh.vertices, dtype=float)[faces]
    e01 = points[:, 1] - points[:, 0]
    e02 = points[:, 2] - points[:, 0]
    normal2 = np.cross(e01, e02)
    normal2_squared = np.einsum("ij,ij->i", normal2, normal2)
    gradients = np.stack(
        (
            np.cross(normal2, points[:, 2] - points[:, 1]),
            np.cross(normal2, points[:, 0] - points[:, 2]),
            np.cross(normal2, points[:, 1] - points[:, 0]),
        ),
        axis=1,
    ) / np.maximum(normal2_squared[:, None, None], 1.0e-30)
    edge_faces: dict[tuple[int, int], list[tuple[int, int]]] = {}
    local_edges = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    for face_index, face in enumerate(faces):
        for first, second, opposite in local_edges:
            edge = tuple(sorted((int(face[first]), int(face[second]))))
            edge_faces.setdefault(edge, []).append((face_index, opposite))

    h = 1.0 / np.sqrt(max(len(mesh.vertices), 1))
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    vertices = np.asarray(mesh.vertices, dtype=float)
    for edge, adjacent in edge_faces.items():
        if len(adjacent) != 2:
            continue
        edge_vector = vertices[edge[1]] - vertices[edge[0]]
        edge_length = np.linalg.norm(edge_vector)
        if edge_length <= 1.0e-30:
            continue
        jump: dict[int, float] = {}
        midpoint = 0.5 * (vertices[edge[0]] + vertices[edge[1]])
        for face_index, opposite_local in adjacent:
            third = points[face_index, opposite_local]
            edge_unit = edge_vector / edge_length
            inward = third - midpoint
            inward -= edge_unit * np.dot(inward, edge_unit)
            conormal = -inward / max(np.linalg.norm(inward), 1.0e-30)
            for local, vertex in enumerate(faces[face_index]):
                jump[int(vertex)] = jump.get(int(vertex), 0.0) + float(
                    np.dot(conormal, gradients[face_index, local])
                )
        keys = list(jump)
        multiplier = h * edge_length
        for first in keys:
            for second in keys:
                rows.append(first)
                columns.append(second)
                values.append(multiplier * jump[first] * jump[second])
    return sp.coo_matrix(
        (values, (rows, columns)),
        shape=(len(vertices), len(vertices)),
    ).tocsr()


def stabilized_fem_mean_curvature(
    mesh: trimesh.Trimesh,
    *,
    tau: float = 0.1,
    coordinate_scale: float = 1.0,
    signed: bool = True,
) -> np.ndarray:
    """Return the published edge-jump stabilized P1 mean curvature.

    ``tau=0.1`` is the value used for the triangulated-torus experiments in
    Hansbo, Larson, and Zahedi (2015).  Their coordinates are nondimensional.
    ``coordinate_scale`` declares the physical length represented by one unit
    of that nondimensional system; the factor ``coordinate_scale**3`` follows
    by mapping their edge-jump bilinear form back to physical coordinates.
    Their solved vector represents ``(k1+k2)n``; division by two returns
    HullProd's conventional H normal.
    """
    if tau < 0.0:
        raise ValueError("tau must be nonnegative")
    if coordinate_scale <= 0.0:
        raise ValueError("coordinate_scale must be positive")
    mass = consistent_p1_mass_matrix(mesh)
    jump = edge_jump_stabilization(mesh)
    system = (mass + tau * coordinate_scale**3 * jump).tocsc()
    load = np.asarray(cotangent_laplacian(mesh) @ mesh.vertices)
    vector = np.asarray(spla.splu(system).solve(load)) / 2.0
    normal_component = np.einsum("ij,ij->i", vector, max_weighted_vertex_normals(mesh))
    return normal_component if signed else np.abs(normal_component)


def direct_gradient_energy(
    mesh: trimesh.Trimesh,
    gradient: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    face_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Integrate a direct vertex gradient with the consistent P1 mass matrix.

    Returns the energy and represented valid-area fraction.  Faces touching an
    invalid vertex are excluded; no missing value is filled or clipped.
    """
    values = np.asarray(gradient, dtype=float)
    if values.shape != (len(mesh.vertices), 3):
        raise ValueError("gradient must contain one vector per vertex")
    valid_vertices = np.all(np.isfinite(values), axis=1)
    if valid is not None:
        valid_vertices &= np.asarray(valid, dtype=bool)
    face_valid = np.all(valid_vertices[np.asarray(mesh.faces)], axis=1)
    if face_mask is not None:
        requested = np.asarray(face_mask, dtype=bool)
        if requested.shape != (len(mesh.faces),):
            raise ValueError("face_mask must contain one value per mesh face")
        face_valid &= requested
    if not np.any(face_valid):
        return float("nan"), 0.0
    submesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces)[face_valid],
        process=False,
    )
    mass = consistent_p1_mass_matrix(submesh)
    clean = np.where(valid_vertices[:, None], values, 0.0)
    energy = sum(float(clean[:, axis] @ (mass @ clean[:, axis])) for axis in range(3))
    return energy, float(np.sum(mesh.area_faces[face_valid]) / mesh.area)
