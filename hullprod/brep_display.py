"""Visualization-only tessellation and native BRep field sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from .brep_geometry import BRepModel, evaluate_surface_differential
from .schema import CurvatureClassID
from .types import ProducibilityConfig

DISPLAY_QUALITY_PRESETS: dict[str, dict[str, float]] = {
    "draft": {
        "linear_deflection_span_fraction": 2.0e-3,
        "angular_deflection": 0.25,
    },
    "standard": {
        "linear_deflection_span_fraction": 7.5e-4,
        "angular_deflection": 0.12,
    },
    "fine": {
        "linear_deflection_span_fraction": 3.5e-4,
        "angular_deflection": 0.075,
    },
}


@dataclass(frozen=True)
class BRepDisplay:
    """A non-canonical drawing mesh with fields sampled from the BRep."""

    mesh: trimesh.Trimesh
    local_fields: dict[str, np.ndarray]
    metadata: dict[str, Any]


def sample_brep_reference_points(model: BRepModel) -> tuple[np.ndarray, dict[str, Any]]:
    """Return controlled BRep tessellation points used only for automatic ``L_ref``.

    Deflection is scaled by the square root of exact BRep area, which is
    translation/rotation invariant and scales with model length.  The returned
    samples do not participate in the canonical native-BRep integrations.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopLoc import TopLoc_Location

    area = float(model.metadata.get("independent_surface_area", 0.0))
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("Positive BRep surface area is required for automatic reference length")
    area_length_scale = float(np.sqrt(area))
    area_scale_fraction = 2.0e-3
    linear_deflection = max(area_scale_fraction * area_length_scale, 1.0e-12)
    angular_deflection = 0.25

    BRepTools.Clean_s(model.shape)
    mesher = BRepMesh_IncrementalMesh(
        model.shape,
        linear_deflection,
        False,
        angular_deflection,
        True,
    )
    mesher.Perform()
    if not mesher.IsDone():
        raise RuntimeError("OpenCascade automatic-reference sampling failed")

    vertices: list[list[float]] = []
    for face in model.faces:
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None:
            continue
        transformation = location.Transformation()
        for node_index in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(node_index).Transformed(transformation)
            vertices.append([point.X(), point.Y(), point.Z()])
    if len(vertices) < 2:
        raise RuntimeError("OpenCascade automatic-reference sampling produced too few points")
    return np.asarray(vertices, dtype=float), {
        "representation": "controlled_BRep_tessellation",
        "purpose": "automatic_reference_length_only",
        "canonical_metrics_depend_on_sampling": False,
        "linear_deflection": linear_deflection,
        "linear_deflection_area_length_fraction": area_scale_fraction,
        "area_length_scale": area_length_scale,
        "angular_deflection": angular_deflection,
    }


def resolve_display_settings(
    config: ProducibilityConfig,
    *,
    geometric_max_span: float,
) -> dict[str, Any]:
    """Resolve visualization-only preset values and explicit overrides."""
    quality = str(config.brep_display_quality).strip().lower()
    if quality not in DISPLAY_QUALITY_PRESETS:
        choices = ", ".join(DISPLAY_QUALITY_PRESETS)
        raise ValueError(f"Unknown display quality {quality!r}; expected one of: {choices}")
    if not np.isfinite(geometric_max_span) or geometric_max_span <= 0.0:
        raise ValueError("Geometric maximum span must be positive for display tessellation")
    preset = DISPLAY_QUALITY_PRESETS[quality]
    span_fraction = float(preset["linear_deflection_span_fraction"])
    linear = (
        float(config.brep_display_linear_deflection)
        if config.brep_display_linear_deflection is not None
        else max(span_fraction * geometric_max_span, 1.0e-8)
    )
    angular = (
        float(config.brep_display_angular_deflection)
        if config.brep_display_angular_deflection is not None
        else float(preset["angular_deflection"])
    )
    if not np.isfinite(linear) or linear <= 0.0:
        raise ValueError("Display linear deflection must be positive")
    if not np.isfinite(angular) or angular <= 0.0:
        raise ValueError("Display angular deflection must be positive")
    return {
        "display_quality": quality,
        "linear_deflection": linear,
        "angular_deflection": angular,
        "linear_deflection_span_fraction": linear / geometric_max_span,
        "preset_linear_deflection_span_fraction": span_fraction,
        "preset_angular_deflection": float(preset["angular_deflection"]),
        "linear_deflection_source": (
            "explicit_override"
            if config.brep_display_linear_deflection is not None
            else "display_quality_preset"
        ),
        "angular_deflection_source": (
            "explicit_override"
            if config.brep_display_angular_deflection is not None
            else "display_quality_preset"
        ),
        "relative_deflection": False,
    }


def generate_brep_display(
    model: BRepModel,
    config: ProducibilityConfig,
    *,
    length_ref: float,
) -> BRepDisplay:
    """Tessellate for drawing, then sample fields from exact face derivatives."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location

    bounds = np.asarray(model.metadata["bounds"], dtype=float)
    span = float(np.max(np.ptp(bounds, axis=0)))
    settings = resolve_display_settings(config, geometric_max_span=span)
    deflection = float(settings["linear_deflection"])
    # OpenCascade caches polygonal representations on the shape.  Clear only
    # that non-geometric cache so a requested display resolution is honored;
    # this does not alter the source BRep surfaces or topology.
    BRepTools.Clean_s(model.shape)
    mesher = BRepMesh_IncrementalMesh(
        model.shape,
        deflection,
        False,
        float(settings["angular_deflection"]),
        True,
    )
    mesher.Perform()
    if not mesher.IsDone():
        raise RuntimeError("OpenCascade visualization tessellation failed")

    vertices: list[list[float]] = []
    uv_values: list[tuple[float, float]] = []
    face_indices: list[int] = []
    triangles: list[list[int]] = []
    for face_index, face in enumerate(model.faces):
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None or triangulation.NbTriangles() == 0:
            continue
        if not triangulation.HasUVNodes():
            continue
        offset = len(vertices)
        transformation = location.Transformation()
        for node_index in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(node_index).Transformed(transformation)
            uv = triangulation.UVNode(node_index)
            vertices.append([point.X(), point.Y(), point.Z()])
            uv_values.append((uv.X(), uv.Y()))
            face_indices.append(face_index)
        reversed_face = face.Orientation() == TopAbs_REVERSED
        for triangle_index in range(1, triangulation.NbTriangles() + 1):
            n1, n2, n3 = triangulation.Triangle(triangle_index).Get()
            values = [offset + n1 - 1, offset + n2 - 1, offset + n3 - 1]
            if reversed_face:
                values[1], values[2] = values[2], values[1]
            triangles.append(values)
    if not triangles:
        raise RuntimeError("Visualization tessellation produced no UV-mapped triangles")
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(triangles, dtype=np.int64),
        process=False,
    )

    h_threshold = (
        config.h_threshold
        if config.h_threshold is not None
        else config.h_threshold_factor / length_ref
    )
    k_threshold = (
        config.k_threshold
        if config.k_threshold is not None
        else config.k_threshold_factor / length_ref**2
    )
    count = len(vertices)
    mean = np.full(count, np.nan)
    gaussian = np.full(count, np.nan)
    gradient_squared = np.full(count, np.nan)
    normal = np.full((count, 3), np.nan)
    valid = np.zeros(count, dtype=float)
    fairness_valid = np.zeros(count, dtype=float)
    classes = np.full(count, int(CurvatureClassID.INVALID), dtype=np.int8)
    surfaces = [BRepAdaptor_Surface(face, True) for face in model.faces]
    for index, ((u, v), face_index) in enumerate(zip(uv_values, face_indices, strict=True)):
        try:
            value = evaluate_surface_differential(
                model.faces[face_index],
                u,
                v,
                need_third=config.brep_compute_fairness,
                surface=surfaces[face_index],
            )
            mean[index] = value.mean_curvature
            gaussian[index] = value.gaussian_curvature
            normal[index] = value.normal
            valid[index] = 1.0
            if config.brep_compute_fairness and value.gradient_mean_squared is not None:
                gradient_squared[index] = value.gradient_mean_squared
                fairness_valid[index] = 1.0
            if abs(gaussian[index]) <= k_threshold and abs(mean[index]) <= h_threshold:
                classes[index] = int(CurvatureClassID.FLAT)
            elif abs(gaussian[index]) <= k_threshold:
                classes[index] = int(CurvatureClassID.SINGLE_CURVATURE)
            elif gaussian[index] > k_threshold:
                classes[index] = int(CurvatureClassID.ELLIPTIC_DOUBLE_CURVATURE)
            else:
                classes[index] = int(CurvatureClassID.SADDLE_REVERSE_DOUBLE_CURVATURE)
        except Exception:
            try:
                value = evaluate_surface_differential(
                    model.faces[face_index],
                    u,
                    v,
                    need_third=False,
                    surface=surfaces[face_index],
                )
                mean[index] = value.mean_curvature
                gaussian[index] = value.gaussian_curvature
                normal[index] = value.normal
                valid[index] = 1.0
            except Exception:
                pass
    sample_valid = valid.astype(bool)
    classes[sample_valid & (np.abs(gaussian) <= k_threshold) & (np.abs(mean) <= h_threshold)] = int(
        CurvatureClassID.FLAT
    )
    classes[sample_valid & (np.abs(gaussian) <= k_threshold) & (np.abs(mean) > h_threshold)] = int(
        CurvatureClassID.SINGLE_CURVATURE
    )
    classes[sample_valid & (gaussian > k_threshold)] = int(
        CurvatureClassID.ELLIPTIC_DOUBLE_CURVATURE
    )
    classes[sample_valid & (gaussian < -k_threshold)] = int(
        CurvatureClassID.SADDLE_REVERSE_DOUBLE_CURVATURE
    )
    principal_offset = np.sqrt(np.maximum(mean**2 - gaussian, 0.0))
    threshold_mask = np.full(count, -1, dtype=np.int8)
    threshold_mask[sample_valid] = (np.abs(gaussian[sample_valid]) > k_threshold).astype(np.int8)
    local_fields = {
        "H": mean,
        "K": gaussian,
        "principal_curvature_k1": mean + principal_offset,
        "principal_curvature_k2": mean - principal_offset,
        "H_valid": valid.astype(np.uint8),
        "K_valid": valid.astype(np.uint8),
        "surface_normal_x": normal[:, 0],
        "surface_normal_y": normal[:, 1],
        "surface_normal_z": normal[:, 2],
        "developability_density": np.abs(gaussian) * length_ref**2,
        "developability_positive_density": np.maximum(gaussian, 0.0) * length_ref**2,
        "developability_negative_density": np.maximum(-gaussian, 0.0) * length_ref**2,
        "developability_threshold_mask": threshold_mask,
        "curvature_class_id": classes,
        "brep_face_index": np.asarray(face_indices, dtype=np.int32),
    }
    if config.experimental_metrics or config.n_stations > 0 or config.brep_compute_fairness:
        local_fields["curvature_energy_density"] = (mean * length_ref) ** 2
    if config.brep_compute_fairness:
        local_fields["curvature_fairness_density"] = gradient_squared * length_ref**4
        local_fields["fairness_valid"] = fairness_valid.astype(np.uint8)
    metadata = {
        **settings,
        "generated": True,
        "purpose": "visualization_export_and_mesh_vs_brep_verification_only",
        "canonical_metrics_depend_on_display_mesh": False,
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "triangle_count": len(mesh.faces),
        "field_source": "direct_BRep_face_UV_differential_evaluation",
    }
    return BRepDisplay(mesh=mesh, local_fields=local_fields, metadata=metadata)
