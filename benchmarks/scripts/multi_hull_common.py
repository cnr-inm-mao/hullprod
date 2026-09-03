"""Reusable geometry utilities for the multi-hull validation campaign.

This module is benchmark tooling, not part of HullProd's production estimator
API.  Authoritative CAD masters are read without modification; every scale,
axis change, component selection, and derived half/full operation is recorded.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from hullprod.mesh_quality import analyze_mesh_quality

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "benchmarks/config/multi_hull_validation.json"


def load_campaign_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the versioned multi-hull campaign configuration."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of *path* without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(case: dict[str, Any]) -> Path:
    """Resolve and checksum a configured authoritative master."""
    source = REPO_ROOT / case["source"]
    if not source.is_file():
        raise FileNotFoundError(f"Missing configured source: {source}")
    observed = sha256_file(source)
    expected = case["source_sha256"]
    if observed != expected:
        raise ValueError(f"Source checksum mismatch for {source}: {observed} != {expected}")
    return source


def tessellation_deflections(
    case: dict[str, Any], linear_deflection_over_lref: float
) -> tuple[float, float]:
    """Return physical-metre and OpenCascade-coordinate deflections."""
    physical = linear_deflection_over_lref * float(case["lref_m"])
    working = physical / float(case["ocp_to_physical_m"])
    return physical, working


def apply_canonical_transform(mesh: trimesh.Trimesh, case: dict[str, Any]) -> None:
    """Apply the configured non-mutating source-to-canonical transform."""
    scale = float(case["ocp_to_physical_m"])
    matrix = np.asarray(case["source_to_canonical_matrix"], dtype=float)
    translation = np.asarray(case["canonical_translation_m"], dtype=float)
    mesh.vertices = (mesh.vertices * scale) @ matrix.T + translation


def clean_mesh(mesh: trimesh.Trimesh, *, merge_digits: int = 9) -> trimesh.Trimesh:
    """Merge coincident CAD-seam vertices and remove topological duplicates."""
    result = mesh.copy()
    result.merge_vertices(digits_vertex=merge_digits)
    if len(result.faces):
        result.update_faces(result.nondegenerate_faces())
        result.update_faces(result.unique_faces())
    result.remove_unreferenced_vertices()
    return result


def remove_transfer_debris(
    mesh: trimesh.Trimesh,
    *,
    lref: float,
    minimum_area_over_lref2: float,
) -> tuple[trimesh.Trimesh, dict[str, float | int]]:
    """Remove only numerically negligible disconnected transfer fragments."""
    components = mesh.split(only_watertight=False)
    threshold = minimum_area_over_lref2 * lref * lref
    retained = [component for component in components if component.area >= threshold]
    discarded = [component for component in components if component.area < threshold]
    if not retained:
        raise ValueError("Transfer-debris filter would remove every mesh component")
    result = trimesh.util.concatenate(retained)
    result = clean_mesh(result)
    audit: dict[str, float | int] = {
        "minimum_component_area_m2": float(threshold),
        "minimum_component_area_over_lref2": float(minimum_area_over_lref2),
        "discarded_component_count": len(discarded),
        "discarded_face_count": int(sum(len(component.faces) for component in discarded)),
        "discarded_area_m2": float(sum(component.area for component in discarded)),
    }
    return result, audit


def mirror_half_mesh(
    mesh: trimesh.Trimesh,
    *,
    seam_tolerance: float,
    merge_digits: int = 9,
) -> tuple[trimesh.Trimesh, dict[str, float]]:
    """Mirror a half hull about ``y=0`` and weld its symmetry seam.

    Vertices already within ``seam_tolerance`` of the centerplane are projected
    onto it before reflection.  This tolerance is recorded by the caller and is
    never inferred from a dimensional manufacturing rule.
    """
    half = mesh.copy()
    near = np.abs(half.vertices[:, 1]) <= seam_tolerance
    maximum_projection = float(np.max(np.abs(half.vertices[near, 1]), initial=0.0))
    half.vertices[near, 1] = 0.0

    reflected_vertices = half.vertices.copy()
    reflected_vertices[:, 1] *= -1.0
    offset = len(half.vertices)
    reflected_faces = half.faces[:, ::-1] + offset
    full = trimesh.Trimesh(
        vertices=np.vstack((half.vertices, reflected_vertices)),
        faces=np.vstack((half.faces, reflected_faces)),
        process=False,
    )
    full = clean_mesh(full, merge_digits=merge_digits)
    audit = {
        "seam_tolerance_m": float(seam_tolerance),
        "projected_vertex_count": int(np.count_nonzero(near)),
        "maximum_projection_m": maximum_projection,
        "maximum_projection_over_lref": 0.0,
    }
    return full, audit


def cut_full_mesh(
    mesh: trimesh.Trimesh,
    *,
    seam_tolerance: float,
    merge_digits: int = 9,
) -> tuple[trimesh.Trimesh, dict[str, float]]:
    """Derive the ``y >= 0`` half of a full symmetric triangulation."""
    # Clip directly instead of using trimesh.slice_plane, whose polygon path
    # requires Shapely.  Benchmark operation must not add a mandatory package.
    vertices: list[np.ndarray] = []
    faces: list[list[int]] = []
    for source_face in mesh.faces:
        polygon = [mesh.vertices[index].copy() for index in source_face]
        clipped: list[np.ndarray] = []
        for start, end in zip(polygon, [*polygon[1:], polygon[0]], strict=True):
            start_inside = start[1] >= 0.0
            end_inside = end[1] >= 0.0
            if start_inside:
                clipped.append(start)
            if start_inside != end_inside:
                fraction = -start[1] / (end[1] - start[1])
                intersection = start + fraction * (end - start)
                intersection[1] = 0.0
                clipped.append(intersection)
        if len(clipped) < 3:
            continue
        offset = len(vertices)
        vertices.extend(clipped)
        for index in range(1, len(clipped) - 1):
            faces.append([offset, offset + index, offset + index + 1])
    if not faces:
        raise ValueError("Centerplane cut produced no faces")
    half = trimesh.Trimesh(
        vertices=np.asarray(vertices),
        faces=np.asarray(faces),
        process=False,
    )
    near = np.abs(half.vertices[:, 1]) <= seam_tolerance
    maximum_projection = float(np.max(np.abs(half.vertices[near, 1]), initial=0.0))
    half.vertices[near, 1] = 0.0
    half = clean_mesh(half, merge_digits=merge_digits)
    audit = {
        "seam_tolerance_m": float(seam_tolerance),
        "projected_vertex_count": int(np.count_nonzero(near)),
        "maximum_projection_m": maximum_projection,
        "maximum_projection_over_lref": 0.0,
    }
    return half, audit


def _edge_topology(mesh: trimesh.Trimesh) -> tuple[int, int, int]:
    inverse = mesh.edges_unique_inverse
    counts = np.bincount(inverse, minlength=len(mesh.edges_unique))
    boundary_edges = int(np.count_nonzero(counts == 1))
    non_manifold_edges = int(np.count_nonzero(counts > 2))
    boundary_vertices = int(len(np.unique(mesh.edges_unique[counts == 1])) if boundary_edges else 0)
    return boundary_edges, boundary_vertices, non_manifold_edges


def mesh_metadata(
    mesh: trimesh.Trimesh,
    *,
    lref: float,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Return geometry, topology, and quality provenance for one mesh."""
    boundary_edges, boundary_vertices, non_manifold_edges = _edge_topology(mesh)
    components = mesh.split(only_watertight=False)
    metadata: dict[str, Any] = {
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "surface_area_m2": float(mesh.area),
        "surface_area_over_lref2": float(mesh.area / (lref * lref)),
        "bounds_m": {
            "minimum": mesh.bounds[0].tolist(),
            "maximum": mesh.bounds[1].tolist(),
            "span": np.ptp(mesh.bounds, axis=0).tolist(),
        },
        "boundary_edge_count": boundary_edges,
        "boundary_vertex_count": boundary_vertices,
        "non_manifold_edge_count": non_manifold_edges,
        "connected_component_count": len(components),
        "mesh_quality": analyze_mesh_quality(mesh),
    }
    if output_path is not None:
        metadata["output_mesh"] = str(output_path.relative_to(REPO_ROOT))
        metadata["output_mesh_sha256"] = sha256_file(output_path)
        metadata["output_mesh_size_bytes"] = output_path.stat().st_size
    return metadata


def _selected_root_indices(reader: Any, case: dict[str, Any]) -> list[int]:
    if "selected_root_indices" in case:
        return [int(value) for value in case["selected_root_indices"]]
    levels = {int(value) for value in case["selected_iges_levels"]}
    return [
        index
        for index in range(1, reader.NbRootsForTransfer() + 1)
        if reader.RootForTransfer(index).Level() in levels
    ]


def tessellate_iges(
    case: dict[str, Any],
    *,
    linear_deflection_over_lref: float,
    angular_deflection: float,
    output_path: Path,
    heal: bool = True,
    sewing_tolerance_over_lref: float = 1e-6,
    minimum_component_area_over_lref2: float = 1e-12,
    merge_digits: int = 9,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Tessellate selected IGES roots and write a canonical-metre STL."""
    try:
        from OCP.BRep import BRep_Builder
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.IGESControl import IGESControl_Reader
        from OCP.ShapeFix import ShapeFix_Shape
        from OCP.StlAPI import StlAPI_Writer
        from OCP.TopoDS import TopoDS_Compound
    except ImportError as error:  # pragma: no cover - optional benchmark dependency
        raise RuntimeError("cadquery-ocp is required for IGES tessellation") from error

    source = verify_source(case)
    reader = IGESControl_Reader()
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise RuntimeError(f"OpenCascade could not read {source}")

    selected = _selected_root_indices(reader, case)
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    transferred: list[int] = []
    for root_index in selected:
        before = reader.NbShapes()
        if not reader.TransferOneRoot(root_index):
            continue
        after = reader.NbShapes()
        if after <= before:
            continue
        builder.Add(compound, reader.Shape(after))
        transferred.append(root_index)
    if not transferred:
        raise RuntimeError(f"No configured surface roots transferred from {source}")

    lref = float(case["lref_m"])
    ocp_scale = float(case["ocp_to_physical_m"])
    sewing_tolerance_m = sewing_tolerance_over_lref * lref
    sewing = BRepBuilderAPI_Sewing(sewing_tolerance_m / ocp_scale)
    sewing.Add(compound)
    sewing.Perform()
    shape = sewing.SewedShape()
    if heal:
        fixer = ShapeFix_Shape(shape)
        fixer.Perform()
        shape = fixer.Shape()

    physical_deflection, ocp_deflection = tessellation_deflections(
        case, linear_deflection_over_lref
    )
    mesher = BRepMesh_IncrementalMesh(
        shape,
        ocp_deflection,
        False,
        angular_deflection,
        True,
    )
    mesher.Perform()
    if not mesher.IsDone():
        raise RuntimeError(f"OpenCascade tessellation failed for {source}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hullprod-cad-") as temp_directory:
        raw_stl = Path(temp_directory) / "source_coordinates.stl"
        writer = StlAPI_Writer()
        writer.ASCIIMode = False
        if not writer.Write(shape, str(raw_stl)):
            raise RuntimeError(f"OpenCascade STL export failed for {source}")
        loaded = trimesh.load_mesh(raw_stl, process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        loaded = loaded.dump(concatenate=True)
    apply_canonical_transform(loaded, case)
    mesh = clean_mesh(loaded, merge_digits=merge_digits)
    mesh, debris_audit = remove_transfer_debris(
        mesh,
        lref=lref,
        minimum_area_over_lref2=minimum_component_area_over_lref2,
    )
    mesh.export(output_path, file_type="stl")
    # STL is the campaign artifact.  Reload its float32 coordinates so the
    # recorded counts and every derived representation match what downstream
    # HullProd runs actually consume.
    exported = trimesh.load_mesh(output_path, process=True)
    if not isinstance(exported, trimesh.Trimesh):
        exported = exported.dump(concatenate=True)
    mesh = clean_mesh(exported, merge_digits=merge_digits)
    mesh.export(output_path, file_type="stl")

    metadata = {
        "source": str(source.relative_to(REPO_ROOT)),
        "source_sha256": case["source_sha256"],
        "selected_root_indices": selected,
        "transferred_root_indices": transferred,
        "canonical_transform": {
            "ocp_to_physical_m": ocp_scale,
            "matrix": case["source_to_canonical_matrix"],
            "translation_m": case["canonical_translation_m"],
        },
        "lref_m": lref,
        "linear_deflection_m": physical_deflection,
        "linear_deflection_over_lref": linear_deflection_over_lref,
        "open_cascade_linear_deflection": ocp_deflection,
        "angular_deflection_rad": angular_deflection,
        "relative_deflection": False,
        "shape_healing": heal,
        "cad_sewing_tolerance_m": sewing_tolerance_m,
        "cad_sewing_tolerance_over_lref": sewing_tolerance_over_lref,
        "transfer_debris_filter": debris_audit,
    }
    metadata.update(mesh_metadata(mesh, lref=lref, output_path=output_path))
    return mesh, metadata
