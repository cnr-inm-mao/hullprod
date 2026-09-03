"""Experimental native-BRep realization of HullProd's continuous metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import trimesh

from .brep_display import generate_brep_display, sample_brep_reference_points
from .brep_geometry import BRepModel
from .brep_quadrature import integrate_brep_metrics
from .brep_sections import brep_section_waviness
from .brep_validity import classify_brep_validity
from .reference_length import reference_length_preflight, resolve_reference_length
from .schema import schema_metadata, signature_metadata
from .types import MetricResult, ProducibilityConfig
from .units import brep_unit_metadata
from .validity import metric_validity_record


@dataclass(frozen=True)
class BRepAssessment:
    """Common metric result plus an optional non-canonical drawing mesh."""

    result: MetricResult
    display_mesh: trimesh.Trimesh | None


def _topology_continuity(model: BRepModel) -> dict[str, Any]:
    """Report free edges and declared continuity across shared CAD edges."""
    from OCP.BRep import BRep_Tool
    from OCP.GeomAbs import (
        GeomAbs_C0,
        GeomAbs_C1,
        GeomAbs_C2,
        GeomAbs_C3,
        GeomAbs_CN,
        GeomAbs_G1,
        GeomAbs_G2,
    )
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    names = {
        GeomAbs_C0: "C0",
        GeomAbs_C1: "C1",
        GeomAbs_C2: "C2",
        GeomAbs_C3: "C3",
        GeomAbs_CN: "CN",
        GeomAbs_G1: "G1",
        GeomAbs_G2: "G2",
    }
    mapping = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(model.shape, TopAbs_EDGE, TopAbs_FACE, mapping)
    free = non_manifold = 0
    continuity: dict[str, int] = {}
    unknown = 0
    for index in range(1, mapping.Extent() + 1):
        edge = TopoDS.Edge_s(mapping.FindKey(index))
        ancestors = [TopoDS.Face_s(value) for value in mapping.FindFromIndex(index)]
        if len(ancestors) == 1:
            free += 1
            continue
        if len(ancestors) != 2:
            non_manifold += 1
            continue
        try:
            label = names.get(
                BRep_Tool.Continuity_s(edge, ancestors[0], ancestors[1]),
                "other",
            )
            continuity[label] = continuity.get(label, 0) + 1
        except Exception:
            unknown += 1
    sharp = sum(count for label, count in continuity.items() if label in {"C0", "other"})
    return {
        "edge_count": int(mapping.Extent()),
        "free_edge_count": free,
        "non_manifold_edge_count": non_manifold,
        "shared_edge_continuity_counts": continuity,
        "sharp_or_C0_shared_edge_count": sharp,
        "unknown_continuity_edge_count": unknown,
        "patch_boundaries_are_not_automatically_physical_creases": True,
    }


def compute_brep_metrics(
    model: BRepModel,
    config: ProducibilityConfig | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> BRepAssessment:
    """Evaluate canonical metrics directly from trimmed parametric geometry."""
    if config is None:
        config = ProducibilityConfig()
    emit = progress or (lambda _message: None)
    bounds = np.asarray(model.metadata["bounds"], dtype=float)
    emit("Estimating automatic geometric span for reference-length provenance...")
    reference_points, reference_sampling = sample_brep_reference_points(model)
    length_ref, reference_length = resolve_reference_length(
        reference_points,
        config.length_ref,
        source="controlled_BRep_reference_tessellation",
        sampling=reference_sampling,
    )

    phase_timings: dict[str, float] = {}
    units = brep_unit_metadata(model.metadata)
    preflight_messages, reference_warning = reference_length_preflight(reference_length, units)
    for message in preflight_messages:
        emit(message)
    if reference_warning:
        emit(f"WARNING: {reference_warning}")
    emit("Computing native BRep metric integrals...")
    core_started = perf_counter()
    integral = integrate_brep_metrics(model, config, length_ref=length_ref)
    phase_timings["canonical_core_metrics_seconds"] = perf_counter() - core_started
    emit(f"BRep integrals completed in {phase_timings['canonical_core_metrics_seconds']:.1f} s.")
    if config.n_stations > 0:
        emit(f"Computing {config.n_stations} exact transverse sections...")
        sections_started = perf_counter()
        section_metrics, section_metadata = brep_section_waviness(
            model,
            config,
            length_ref=length_ref,
        )
        phase_timings["exact_sections_seconds"] = perf_counter() - sections_started
        emit(f"Sections completed in {phase_timings['exact_sections_seconds']:.1f} s.")
    else:
        section_metrics = {
            "section_waviness": None,
            "section_waviness_fft": None,
            "valid_sections": 0,
        }
        section_metadata = {
            "requested_stations": 0,
            "stations": [],
            "status": "not_evaluated",
            "reason": "Section metrics require explicit experimental opt-in.",
            "section_side_policy_actual": config.section_side_policy,
        }
    curvature_area = integral.values["curvature_valid_area"]
    fairness_area = integral.values["fairness_valid_area"]
    total_area = integral.values["surface_area"]
    class_total = sum(integral.class_areas.values())
    classes = {
        "area_fraction_flat": integral.class_areas["flat"] / max(class_total, config.eps),
        "area_fraction_cylindrical_single_curvature": (
            integral.class_areas["single"] / max(class_total, config.eps)
        ),
        "area_fraction_elliptic_double_curvature": (
            integral.class_areas["elliptic"] / max(class_total, config.eps)
        ),
        "area_fraction_saddle_reverse_double_curvature": (
            integral.class_areas["saddle"] / max(class_total, config.eps)
        ),
    }
    validity, public_values, regularity_classification = classify_brep_validity(
        integral,
        section_metrics,
        config,
    )
    experimental_enabled = bool(
        config.experimental_metrics or config.n_stations > 0 or config.brep_compute_fairness
    )
    if not experimental_enabled:
        public_values["curvature_energy"] = None
        validity["curvature_energy"] = metric_validity_record(
            value=None,
            validity="not_evaluated",
            reason=(
                "Screened metric not exposed by the default assessment; internal "
                "regularity evidence remains available to qualify the retained integrals."
            ),
            representation="brep",
            backend="brep_native",
        )
    metrics = {
        "surface_area": public_values["surface_area"],
        "length_ref": float(length_ref),
        "h_threshold": public_values["h_threshold"],
        "k_threshold": public_values["k_threshold"],
        "curvature_energy": public_values["curvature_energy"],
        "curvature_fairness": public_values["curvature_fairness"],
        "developability_deviation": public_values["developability_deviation"],
        "developability_deviation_positive": public_values["developability_deviation_positive"],
        "developability_deviation_negative": public_values["developability_deviation_negative"],
        "developability_area_ratio": public_values["developability_area_ratio"],
        "local_plate_twist": public_values["local_plate_twist"],
        "section_waviness": public_values["section_waviness"],
        "section_waviness_fft": public_values["section_waviness_fft"],
        "valid_sections": int(public_values["valid_sections"] or 0),
    }

    display_mesh = None
    local_fields: dict[str, Any] = {}
    display_metadata: dict[str, Any] = {
        "generated": False,
        "display_quality": config.brep_display_quality,
        "canonical_metrics_depend_on_display_mesh": False,
    }
    if config.brep_display_mesh:
        quality = config.brep_display_quality.upper()
        emit(f"Generating {quality} display tessellation...")
        display_started = perf_counter()
        display = generate_brep_display(model, config, length_ref=length_ref)
        phase_timings["display_tessellation_seconds"] = perf_counter() - display_started
        display_mesh = display.mesh
        local_fields = display.local_fields
        display_metadata = display.metadata
        emit(
            "Display tessellation completed: "
            f"{len(display_mesh.faces):,} triangles in "
            f"{phase_timings['display_tessellation_seconds']:.1f} s."
        )

    topology = _topology_continuity(model)
    quadrature_converged = integral.metadata["convergence_status"] == "converged"
    warnings = list(model.metadata.get("import_warnings", []))
    if reference_warning:
        warnings.append(reference_warning)
    if not quadrature_converged:
        warnings.append(
            "Native BRep quadrature reached its configured safeguard or did not "
            "match the independent OpenCascade area tolerance; inspect quadrature metadata."
        )
    for name, record in validity.items():
        if record["status"] not in {"valid", "valid_improper_integral_convergent"}:
            warnings.append(f"{name}: {record['reason']}")
    metadata = {
        **schema_metadata(),
        **signature_metadata(),
        "representation": "brep",
        "backend": "brep_native",
        "backend_readiness": "stable_metric_specific_validity",
        "source_format": model.metadata["source_format"],
        "input_geometry": {
            "path": model.metadata["source_path"],
            "sha256": model.metadata["source_sha256"],
        },
        "brep_import": model.metadata,
        "bounds_min": bounds[0].tolist(),
        "bounds_max": bounds[1].tolist(),
        "reference_length": reference_length,
        "experimental_metrics_enabled": experimental_enabled,
        "n_faces": len(model.faces),
        "metric_validity": validity,
        "field_associations": {name: "point" for name in local_fields},
        "validity_vocabulary_version": 1,
        "regularity_classification": regularity_classification,
        "estimator_metadata": {
            "curvature_method": "OpenCascade_BRepAdaptor_D2_direct",
            "mean_curvature_convention": "signed_H=(k1+k2)/2",
            "mean_curvature_orientation": "sign_follows_TopoDS_face_orientation",
            "gaussian_boundary_policy": "regular_trimmed_BRep_points_are_valid",
            "gaussian_area_normalization": "represented_valid_BRep_area",
            "fairness_integrator": "direct_trimmed_BRep_quadrature",
            "fairness_input": "analytic_D3_surface_gradient_of_signed_H",
            "section_method": "BRep_plane_intersection_exact_curve_D2",
            "local_plate_twist_interpretation": "not_applicable_in_BRep_mode",
        },
        "curvature_thresholds": {
            "h_threshold": integral.values["h_threshold"],
            "k_threshold": integral.values["k_threshold"],
            "h_threshold_factor": float(config.h_threshold_factor),
            "k_threshold_factor": float(config.k_threshold_factor),
            "h_threshold_formula": "factor/L_ref unless absolute override",
            "k_threshold_formula": "factor/L_ref^2 unless absolute override",
        },
        "curvature_valid_area": curvature_area,
        "curvature_valid_area_fraction": curvature_area / max(total_area, config.eps),
        "fairness_valid_area": fairness_area,
        "fairness_valid_area_fraction": fairness_area / max(total_area, config.eps),
        "quadrature": integral.metadata,
        "warnings": warnings,
        "continuity": topology,
        "section_settings": section_metadata,
        "display_mesh": display_metadata,
        "canonical_metrics_depend_on_display_mesh": False,
        "phase_timings": phase_timings,
        "units": units,
    }
    result = MetricResult(
        metrics=metrics,
        curvature_classes=classes,
        metadata=metadata,
        local_fields=local_fields,
    )
    return BRepAssessment(result=result, display_mesh=display_mesh)
