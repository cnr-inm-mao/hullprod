from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import trimesh


def vertex_areas(mesh: trimesh.Trimesh) -> np.ndarray:
    """Compute barycentric vertex-associated areas."""
    n = len(mesh.vertices)
    areas = np.zeros(n, dtype=float)
    face_areas = mesh.area_faces
    faces = mesh.faces
    for local in range(3):
        np.add.at(areas, faces[:, local], face_areas / 3.0)
    return areas


def face_angles(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return internal angles for each triangular face."""
    v = mesh.vertices
    f = mesh.faces
    p0 = v[f[:, 0]]
    p1 = v[f[:, 1]]
    p2 = v[f[:, 2]]

    def angle(a, b, c):
        u = b - a
        w = c - a
        nu = np.linalg.norm(u, axis=1)
        nw = np.linalg.norm(w, axis=1)
        cosang = np.einsum("ij,ij->i", u, w) / np.maximum(nu * nw, 1e-30)
        return np.arccos(np.clip(cosang, -1.0, 1.0))

    return np.column_stack(
        [
            angle(p0, p1, p2),
            angle(p1, p2, p0),
            angle(p2, p0, p1),
        ]
    )


def boundary_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    """Boolean mask of boundary vertices."""
    n = len(mesh.vertices)
    mask = np.zeros(n, dtype=bool)
    edges = mesh.edges_sorted
    edges_unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = edges_unique[counts == 1]
    if len(boundary_edges):
        mask[np.unique(boundary_edges.ravel())] = True
    return mask


def gaussian_curvature(mesh: trimesh.Trimesh, areas: np.ndarray | None = None) -> np.ndarray:
    """Discrete Gaussian curvature from angle defect.

    Interior vertices use 2*pi minus incident angles. Boundary vertices use
    pi minus incident angles. The result is divided by vertex-associated area.
    """
    if areas is None:
        areas = vertex_areas(mesh)

    n = len(mesh.vertices)
    angles = face_angles(mesh)
    angle_sum = np.zeros(n, dtype=float)
    for local in range(3):
        np.add.at(angle_sum, mesh.faces[:, local], angles[:, local])

    bmask = boundary_vertices(mesh)
    defect = np.where(bmask, np.pi - angle_sum, 2.0 * np.pi - angle_sum)
    return defect / np.maximum(areas, 1e-30)


def cotangent_laplacian(mesh: trimesh.Trimesh) -> sp.csr_matrix:
    """Symmetric cotangent Laplacian matrix."""
    f = mesh.faces
    angles = face_angles(mesh)
    n = len(mesh.vertices)
    cot = 1.0 / np.tan(np.clip(angles, 1e-12, np.pi - 1e-12))

    row_idx = []
    col_idx = []
    values = []

    edge_pairs = [(1, 2, 0), (2, 0, 1), (0, 1, 2)]
    for a, b, opp in edge_pairs:
        ia = f[:, a]
        ib = f[:, b]
        w = 0.5 * cot[:, opp]

        row_idx.extend(ia)
        col_idx.extend(ib)
        values.extend(-w)

        row_idx.extend(ib)
        col_idx.extend(ia)
        values.extend(-w)

        row_idx.extend(ia)
        col_idx.extend(ia)
        values.extend(w)

        row_idx.extend(ib)
        col_idx.extend(ib)
        values.extend(w)

    return sp.coo_matrix((values, (row_idx, col_idx)), shape=(n, n)).tocsr()


def mean_curvature(
    mesh: trimesh.Trimesh, areas: np.ndarray | None = None, signed: bool = False
) -> np.ndarray:
    """Estimate conventional mean curvature ``H = (k1 + k2) / 2``.

    HullProd's cotangent Laplacian uses half-cotangent edge weights, so
    ``(L @ x) / (2 A)`` approximates the mean-curvature normal vector.  The
    unsigned result is its magnitude.  ``signed=True`` preserves the existing
    orientation convention based on the dot product with vertex normals.
    """
    if areas is None:
        areas = vertex_areas(mesh)
    L = cotangent_laplacian(mesh)
    hn = (L @ mesh.vertices) / np.maximum(2.0 * areas[:, None], 1e-30)
    H = np.linalg.norm(hn, axis=1)
    if signed:
        normals = mesh.vertex_normals
        s = np.sign(np.einsum("ij,ij->i", hn, normals))
        H = H * np.where(s == 0, 1.0, s)
    return H


def edge_neighbor_data(mesh: trimesh.Trimesh):
    """Return unique internal edges and adjacent face ids."""
    edges = mesh.edges_sorted
    face_ids = np.repeat(np.arange(len(mesh.faces)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges_o = edges[order]
    faces_o = face_ids[order]
    unique_edges, adjacent = [], []
    i = 0
    while i < len(edges_o):
        j = i + 1
        while j < len(edges_o) and np.all(edges_o[j] == edges_o[i]):
            j += 1
        if j - i == 2:
            unique_edges.append(edges_o[i])
            adjacent.append(faces_o[i:j])
        i = j
    if not unique_edges:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)
    return np.asarray(unique_edges, dtype=int), np.asarray(adjacent, dtype=int)


def largest_connected_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return the largest face-connected component."""
    comps = mesh.split(only_watertight=False)
    if not comps:
        return mesh
    return max(comps, key=lambda m: m.area)
