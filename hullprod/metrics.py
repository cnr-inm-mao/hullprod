from __future__ import annotations

from collections.abc import Callable

import numpy as np
import trimesh

from .curvature import estimate_mesh_curvature
from .fairness import face_gradient_contributions
from .mesh_ops import (
    boundary_vertices,
    edge_neighbor_data,
    vertex_areas,
)
from .mesh_quality import analyze_mesh_quality
from .reference_length import reference_length_preflight, resolve_reference_length
from .schema import CurvatureClassID, schema_metadata, signature_metadata
from .sections import section_waviness
from .types import MetricResult, ProducibilityConfig
from .units import mesh_unit_metadata
from .validity import metric_validity_record


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def _curvature_fairness(
    mesh: trimesh.Trimesh,
    H: np.ndarray,
    vertex_area: np.ndarray,
    area_total: float,
    L_ref: float,
    eps: float,
) -> float:
    """Dimensionless area-weighted curvature fairness index."""
    edges = mesh.edges_unique

    if len(edges) == 0:
        return float("nan")

    vertices = mesh.vertices
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    valid = lengths > eps

    if not np.any(valid):
        return float("nan")

    valid_edges = edges[valid]
    valid_lengths = lengths[valid]

    g2_edge = ((H[valid_edges[:, 0]] - H[valid_edges[:, 1]]) / np.maximum(valid_lengths, eps)) ** 2

    grad2_sum = np.zeros(len(mesh.vertices), dtype=float)
    counts = np.zeros(len(mesh.vertices), dtype=float)

    np.add.at(grad2_sum, valid_edges[:, 0], g2_edge)
    np.add.at(grad2_sum, valid_edges[:, 1], g2_edge)
    np.add.at(counts, valid_edges[:, 0], 1.0)
    np.add.at(counts, valid_edges[:, 1], 1.0)

    grad2_vertex = grad2_sum / np.maximum(counts, 1.0)

    return float((L_ref**4) * np.sum(vertex_area * grad2_vertex) / max(area_total, eps))


def _edge_length_threshold(
    mesh: trimesh.Trimesh,
    percentile: float,
    eps: float,
) -> float:
    edges = mesh.edges_unique
    if len(edges) == 0:
        return float("nan")

    lengths = np.linalg.norm(
        mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]],
        axis=1,
    )
    threshold = _finite_percentile(lengths, percentile)
    return max(threshold, eps)


def _curvature_fairness_filtered(
    mesh: trimesh.Trimesh,
    H: np.ndarray,
    vertex_area: np.ndarray,
    vertex_mask: np.ndarray,
    L_ref: float,
    edge_length_min: float,
    eps: float,
) -> float:
    """Experimental robust fairness using vertex and edge-length masks."""
    edges = mesh.edges_unique

    if len(edges) == 0:
        return float("nan")

    vertices = mesh.vertices
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)

    valid = np.isfinite(lengths) & (lengths >= edge_length_min) & (lengths > eps)
    if not np.any(valid):
        return float("nan")

    valid_edges = edges[valid]
    valid_lengths = lengths[valid]

    g2_edge = ((H[valid_edges[:, 0]] - H[valid_edges[:, 1]]) / np.maximum(valid_lengths, eps)) ** 2

    grad2_sum = np.zeros(len(mesh.vertices), dtype=float)
    counts = np.zeros(len(mesh.vertices), dtype=float)

    np.add.at(grad2_sum, valid_edges[:, 0], g2_edge)
    np.add.at(grad2_sum, valid_edges[:, 1], g2_edge)
    np.add.at(counts, valid_edges[:, 0], 1.0)
    np.add.at(counts, valid_edges[:, 1], 1.0)

    grad2_vertex = grad2_sum / np.maximum(counts, 1.0)

    mask = vertex_mask & np.isfinite(grad2_vertex) & np.isfinite(vertex_area)
    area_masked = float(np.sum(vertex_area[mask]))

    if area_masked <= eps:
        return float("nan")

    return float((L_ref**4) * np.sum(vertex_area[mask] * grad2_vertex[mask]) / area_masked)


def _curvature_energy_robust(
    H: np.ndarray,
    areas: np.ndarray,
    mask: np.ndarray,
    L_ref: float,
    h_clip_percentile: float,
    eps: float,
) -> tuple[float, float]:
    """Experimental robust curvature energy with area mask and H winsorization."""
    mask = mask & np.isfinite(H) & np.isfinite(areas)
    area_masked = float(np.sum(areas[mask]))

    if area_masked <= eps:
        return float("nan"), float("nan")

    h_clip = _finite_percentile(np.abs(H[mask]), h_clip_percentile)
    H_work = np.clip(H, -h_clip, h_clip)

    H_ref = 1.0 / L_ref
    value = float(np.sum(areas[mask] * (H_work[mask] / H_ref) ** 2) / area_masked)

    return value, h_clip


def _developability_components(
    K: np.ndarray,
    areas: np.ndarray,
    mask: np.ndarray,
    area_norm: float,
    L_ref: float,
    eps: float,
) -> tuple[float, float, float]:
    K_ref = 1.0 / (L_ref * L_ref)
    mask = mask & np.isfinite(K) & np.isfinite(areas)

    positive = float(np.sum(areas[mask] * np.maximum(K[mask], 0.0) / K_ref) / max(area_norm, eps))
    negative = float(np.sum(areas[mask] * np.maximum(-K[mask], 0.0) / K_ref) / max(area_norm, eps))

    return positive + negative, positive, negative


def _twist_index(mesh: trimesh.Trimesh) -> tuple[float, np.ndarray]:
    edges, adjacent = edge_neighbor_data(mesh)

    if len(edges) == 0:
        return float("nan"), np.zeros(len(mesh.faces))

    normals = mesh.face_normals
    vertices = mesh.vertices

    n1 = normals[adjacent[:, 0]]
    n2 = normals[adjacent[:, 1]]

    dot = np.einsum("ij,ij->i", n1, n2)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))

    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    idx = float(np.sum(lengths * theta**2) / np.maximum(np.sum(lengths), 1e-30))

    face_twist = np.zeros(len(mesh.faces))
    counts = np.zeros(len(mesh.faces))

    for k, (f1, f2) in enumerate(adjacent):
        face_twist[f1] += theta[k]
        face_twist[f2] += theta[k]
        counts[f1] += 1
        counts[f2] += 1

    face_twist = face_twist / np.maximum(counts, 1.0)

    return idx, face_twist


def _edge_length_metadata(mesh: trimesh.Trimesh) -> dict[str, float]:
    """Return internal-edge length metadata used to interpret twist metrics."""
    edges, _ = edge_neighbor_data(mesh)

    if len(edges) == 0:
        return {
            "edge_length_mean": float("nan"),
            "edge_length_median": float("nan"),
            "edge_length_min": float("nan"),
            "edge_length_p01": float("nan"),
            "edge_length_p05": float("nan"),
            "edge_length_p95": float("nan"),
            "edge_length_p99": float("nan"),
            "edge_length_max": float("nan"),
            "edge_length_min_to_median": float("nan"),
            "edge_length_max_to_median": float("nan"),
        }

    edge_lengths = np.linalg.norm(
        mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]],
        axis=1,
    )
    median = float(np.median(edge_lengths))

    return {
        "edge_length_mean": float(np.mean(edge_lengths)),
        "edge_length_median": median,
        "edge_length_min": float(np.min(edge_lengths)),
        "edge_length_p01": float(np.percentile(edge_lengths, 1.0)),
        "edge_length_p05": float(np.percentile(edge_lengths, 5.0)),
        "edge_length_p95": float(np.percentile(edge_lengths, 95.0)),
        "edge_length_p99": float(np.percentile(edge_lengths, 99.0)),
        "edge_length_max": float(np.max(edge_lengths)),
        "edge_length_min_to_median": float(np.min(edge_lengths) / max(median, 1e-30)),
        "edge_length_max_to_median": float(np.max(edge_lengths) / max(median, 1e-30)),
    }


def _curvature_classes(
    H: np.ndarray,
    K: np.ndarray,
    areas: np.ndarray,
    h_thr: float,
    k_thr: float,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    valid = np.isfinite(H) & np.isfinite(K) & np.isfinite(areas)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    area = float(np.sum(areas[valid]))

    if area <= 0:
        return {}

    flat = valid & (np.abs(K) <= k_thr) & (np.abs(H) <= h_thr)
    cylindrical = valid & (np.abs(K) <= k_thr) & (np.abs(H) > h_thr)
    elliptic = valid & (K > k_thr)
    saddle = valid & (K < -k_thr)

    return {
        "area_fraction_flat": float(np.sum(areas[flat]) / area),
        "area_fraction_cylindrical_single_curvature": float(np.sum(areas[cylindrical]) / area),
        "area_fraction_elliptic_double_curvature": float(np.sum(areas[elliptic]) / area),
        "area_fraction_saddle_reverse_double_curvature": float(np.sum(areas[saddle]) / area),
    }


def _curvature_class_ids(
    H: np.ndarray,
    K: np.ndarray,
    h_thr: float,
    k_thr: float,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Return the stable categorical field, keeping invalid distinct from flat."""
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(H) & np.isfinite(K)
    class_ids = np.full(len(H), int(CurvatureClassID.INVALID), dtype=np.int8)
    flat = valid & (np.abs(K) <= k_thr) & (np.abs(H) <= h_thr)
    single = valid & (np.abs(K) <= k_thr) & (np.abs(H) > h_thr)
    elliptic = valid & (K > k_thr)
    saddle = valid & (K < -k_thr)
    class_ids[flat] = int(CurvatureClassID.FLAT)
    class_ids[single] = int(CurvatureClassID.SINGLE_CURVATURE)
    class_ids[elliptic] = int(CurvatureClassID.ELLIPTIC_DOUBLE_CURVATURE)
    class_ids[saddle] = int(CurvatureClassID.SADDLE_REVERSE_DOUBLE_CURVATURE)
    return class_ids


def _robust_metrics(
    mesh: trimesh.Trimesh,
    H: np.ndarray,
    K: np.ndarray,
    areas: np.ndarray,
    L_ref: float,
    config: ProducibilityConfig,
) -> tuple[dict[str, float], dict[str, float | bool | str], dict[str, np.ndarray]]:
    finite = np.isfinite(areas) & np.isfinite(H) & np.isfinite(K)
    area_threshold = _finite_percentile(
        areas[finite],
        config.robust_vertex_area_percentile,
    )
    vertex_mask = finite & (areas >= area_threshold)
    area_masked = float(np.sum(areas[vertex_mask]))

    edge_threshold = _edge_length_threshold(
        mesh,
        config.robust_edge_length_percentile,
        config.eps,
    )

    curvature_energy, h_clip = _curvature_energy_robust(
        H,
        areas,
        vertex_mask,
        L_ref,
        config.robust_h_clip_percentile,
        config.eps,
    )

    curvature_fairness = _curvature_fairness_filtered(
        mesh,
        H,
        areas,
        vertex_mask,
        L_ref,
        edge_threshold,
        config.eps,
    )

    dev, dev_pos, dev_neg = _developability_components(
        K,
        areas,
        vertex_mask,
        area_masked,
        L_ref,
        config.eps,
    )

    metrics = {
        "curvature_energy_robust": curvature_energy,
        "curvature_fairness_robust": curvature_fairness,
        "developability_deviation_robust": dev,
        "developability_deviation_positive_robust": dev_pos,
        "developability_deviation_negative_robust": dev_neg,
    }

    metadata = {
        "robust_metrics_enabled": True,
        "robust_metrics_note": (
            "Experimental diagnostic metrics using vertex-area masking, "
            "edge-length masking for fairness, and H winsorization for "
            "curvature energy. Not a replacement for default v5 metrics."
        ),
        "robust_vertex_area_percentile": float(config.robust_vertex_area_percentile),
        "robust_edge_length_percentile": float(config.robust_edge_length_percentile),
        "robust_h_clip_percentile": float(config.robust_h_clip_percentile),
        "robust_vertex_area_threshold": float(area_threshold),
        "robust_edge_length_threshold": float(edge_threshold),
        "robust_h_clip_value": float(h_clip),
        "robust_vertex_fraction": float(np.sum(vertex_mask) / len(vertex_mask)),
        "robust_area_fraction": float(area_masked / max(float(np.sum(areas)), config.eps)),
    }

    local_fields = {
        "robust_vertex_mask": vertex_mask.astype(float),
    }

    return metrics, metadata, local_fields


def compute_metrics(
    mesh: trimesh.Trimesh,
    config: ProducibilityConfig | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> MetricResult:
    """Compute geometry-based producibility metrics for a triangulated hull surface."""
    if config is None:
        config = ProducibilityConfig()

    area_total = float(mesh.area)
    bounds = mesh.bounds
    L_ref, reference_length = resolve_reference_length(
        mesh.vertices,
        config.length_ref,
        source="represented_mesh_vertices",
        sampling={
            "representation": "native_mesh_vertices",
            "purpose": "automatic_reference_length_and_assessed_representation",
        },
    )
    units = mesh_unit_metadata()
    emit = progress or (lambda _message: None)
    preflight_messages, reference_warning = reference_length_preflight(reference_length, units)
    for message in preflight_messages:
        emit(message)
    if reference_warning:
        emit(f"WARNING: {reference_warning}")
    emit("Computing triangle-mesh curvature and retained signature...")
    experimental_enabled = bool(
        config.experimental_metrics or config.n_stations > 0 or config.robust_metrics
    )

    H_ref = 1.0 / L_ref

    h_thr = (
        config.h_threshold if config.h_threshold is not None else config.h_threshold_factor / L_ref
    )
    k_thr = (
        config.k_threshold
        if config.k_threshold is not None
        else config.k_threshold_factor / (L_ref * L_ref)
    )

    areas = vertex_areas(mesh)
    mesh_quality = analyze_mesh_quality(mesh, areas=areas)
    estimate = estimate_mesh_curvature(mesh)
    boundary = boundary_vertices(mesh)
    H = np.asarray(estimate.mean, dtype=float)
    h_valid = estimate.valid & np.isfinite(H) & np.isfinite(areas)
    K_raw = np.asarray(estimate.gaussian, dtype=float)
    k_valid = estimate.valid & np.isfinite(K_raw) & np.isfinite(areas) & ~boundary
    K = np.where(k_valid, K_raw, np.nan)

    h_valid_area = float(np.sum(areas[h_valid]))
    if experimental_enabled:
        energy_contribution = np.full(len(H), np.nan)
        energy_contribution[h_valid] = (
            areas[h_valid] * (H[h_valid] / H_ref) ** 2 / max(h_valid_area, config.eps)
        )
        curvature_energy = float(np.nansum(energy_contribution))
        fairness_face_valid = np.all(h_valid[mesh.faces], axis=1)
        h_work = np.where(h_valid, H, 0.0)
        fairness_raw = face_gradient_contributions(mesh, h_work)
        fairness_valid_area = float(np.sum(mesh.area_faces[fairness_face_valid]))
        fairness_contribution = np.full(len(mesh.faces), np.nan)
        fairness_contribution[fairness_face_valid] = (
            L_ref**4 * fairness_raw[fairness_face_valid] / max(fairness_valid_area, config.eps)
        )
        curvature_fairness = float(np.nansum(fairness_contribution))
        twist, face_twist = _twist_index(mesh)
    else:
        energy_contribution = None
        curvature_energy = None
        fairness_face_valid = np.zeros(len(mesh.faces), dtype=bool)
        fairness_raw = None
        fairness_valid_area = 0.0
        fairness_contribution = None
        curvature_fairness = None
        twist = None
        face_twist = None

    developability_valid_area = float(np.sum(areas[k_valid]))
    developability_deviation, developability_positive, developability_negative = (
        _developability_components(
            K,
            areas,
            k_valid,
            developability_valid_area,
            L_ref,
            config.eps,
        )
    )

    developability_area_ratio = float(
        np.sum(areas[k_valid & (np.abs(K_raw) > k_thr)])
        / max(developability_valid_area, config.eps)
    )

    if config.n_stations > 0:
        wav = section_waviness(
            mesh,
            n_stations=config.n_stations,
            margin=config.station_margin,
            min_points=config.min_section_points,
            fft_cutoff_fraction=config.fft_cutoff_fraction,
            resample_points=config.section_resample_points,
            length_ref=L_ref,
            section_side_policy=config.section_side_policy,
            section_centerline_tolerance=config.section_centerline_tolerance,
        )
        section_metric_names = {"section_waviness", "section_waviness_fft", "valid_sections"}
        section_metrics = {key: value for key, value in wav.items() if key in section_metric_names}
        section_metadata = {key: value for key, value in wav.items() if key not in section_metric_names}
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

    classes = _curvature_classes(
        H,
        K,
        areas,
        h_thr=h_thr,
        k_thr=k_thr,
        valid_mask=k_valid,
    )
    class_ids = _curvature_class_ids(H, K, h_thr, k_thr, k_valid)

    metrics = {
        "surface_area": area_total,
        "length_ref": L_ref,
        "h_threshold": float(h_thr),
        "k_threshold": float(k_thr),
        "curvature_energy": curvature_energy,
        "curvature_fairness": curvature_fairness,
        "developability_deviation": developability_deviation,
        "developability_deviation_positive": developability_positive,
        "developability_deviation_negative": developability_negative,
        "developability_area_ratio": developability_area_ratio,
        "local_plate_twist": twist,
        **section_metrics,
    }

    mesh_backend = "mesh_rusinkiewicz"
    mesh_reason = (
        "Finite value for the supplied triangulated representation; refinement "
        "evidence is required for representation-independent interpretation."
    )
    mesh_advisory = mesh_quality["curvature_reliability"]["status"]

    def mesh_sensitive_record(value, valid_area, *, reason=mesh_reason, **extra):
        return metric_validity_record(
            value=value,
            validity="mesh_representation_sensitive",
            reason=reason,
            represented_area=area_total,
            valid_area=valid_area,
            numerical_convergence={
                "mesh_quality_advisory": mesh_advisory,
                "mesh_refinement_evidence_supplied": False,
            },
            representation="mesh",
            backend=mesh_backend,
            **extra,
        )

    developability_validity = mesh_sensitive_record(
        developability_deviation,
        developability_valid_area,
        boundary_values="invalid_not_zero_filled",
    )
    metric_validity = {
        "surface_area": metric_validity_record(
            value=area_total,
            validity="valid",
            reason="Surface area is integrated over the supplied triangles.",
            represented_area=area_total,
            valid_area=area_total,
            representation="mesh",
            backend=mesh_backend,
        ),
        "curvature_energy": (
            mesh_sensitive_record(curvature_energy, h_valid_area)
            if curvature_energy is not None
            else metric_validity_record(
                value=None,
                validity="not_evaluated",
                reason="Screened metric; request experimental metrics to evaluate it.",
                representation="mesh",
                backend=mesh_backend,
            )
        ),
        "developability": developability_validity,
        "developability_deviation": dict(developability_validity),
        "developability_deviation_positive": mesh_sensitive_record(
            developability_positive,
            developability_valid_area,
            boundary_values="invalid_not_zero_filled",
        ),
        "developability_deviation_negative": mesh_sensitive_record(
            developability_negative,
            developability_valid_area,
            boundary_values="invalid_not_zero_filled",
        ),
        "developability_area_ratio": mesh_sensitive_record(
            developability_area_ratio,
            developability_valid_area,
            boundary_values="invalid_not_zero_filled",
        ),
        "curvature_classes": mesh_sensitive_record(1.0, developability_valid_area),
        "curvature_fairness": (
            mesh_sensitive_record(
                curvature_fairness,
                fairness_valid_area,
                reason=(
                    "High-sensitivity signed-H P1-F2 diagnostic; mesh-representation "
                    "convergence evidence is required."
                ),
            )
            if curvature_fairness is not None
            else metric_validity_record(
                value=None,
                validity="not_evaluated",
                reason="Screened metric; request experimental metrics to evaluate it.",
                representation="mesh",
                backend=mesh_backend,
            )
        ),
        "section_waviness": metric_validity_record(
            value=section_metrics["section_waviness"],
            validity=(
                "mesh_representation_sensitive"
                if section_metrics["valid_sections"] > 0
                else "not_evaluated"
            ),
            reason=(
                "Finite value for mesh-plane intersection polylines; exact-CAD and "
                "mesh section realizations are representation sensitive."
                if section_metrics["valid_sections"] > 0
                else "Section metrics require explicit experimental opt-in."
            ),
            representation="mesh",
            backend=mesh_backend,
            valid_sections=int(section_metrics["valid_sections"]),
            requested_sections=int(config.n_stations),
        ),
        "section_waviness_fft": metric_validity_record(
            value=section_metrics["section_waviness_fft"],
            validity=(
                "mesh_representation_sensitive"
                if section_metrics["valid_sections"] > 0
                else "not_evaluated"
            ),
            reason=(
                "Finite value for mesh-plane intersection polylines; exact-CAD and "
                "mesh section realizations are representation sensitive."
                if section_metrics["valid_sections"] > 0
                else "Section metrics require explicit experimental opt-in."
            ),
            representation="mesh",
            backend=mesh_backend,
            valid_sections=int(section_metrics["valid_sections"]),
            requested_sections=int(config.n_stations),
        ),
        "local_plate_twist": (
            mesh_sensitive_record(
                twist,
                area_total,
                reason="Mesh-dependent edge-dihedral diagnostic; not a BRep invariant.",
            )
            if twist is not None
            else metric_validity_record(
                value=None,
                validity="not_evaluated",
                reason="Mesh diagnostic; request experimental metrics to evaluate it.",
                representation="mesh",
                backend=mesh_backend,
            )
        ),
    }

    metadata = {
        **schema_metadata(),
        **signature_metadata(),
        "representation": "mesh",
        "backend": "mesh_rusinkiewicz",
        "n_vertices": len(mesh.vertices),
        "n_faces": len(mesh.faces),
        "is_watertight": bool(mesh.is_watertight),
        "bounds_min": bounds[0].tolist(),
        "bounds_max": bounds[1].tolist(),
        "reference_length": reference_length,
        "warnings": [reference_warning] if reference_warning else [],
        "experimental_metrics_enabled": experimental_enabled,
        "robust_metrics_enabled": False,
        "metric_validity": metric_validity,
        "field_associations": {
            "curvature_fairness_density": "cell",
            "curvature_fairness_contribution": "cell",
            "face_twist": "cell",
        },
        "validity_vocabulary_version": 1,
        "estimator_metadata": {
            "curvature_method": "rusinkiewicz_tensor_3dpvt2004",
            "mean_curvature_convention": "signed_H=(k1+k2)/2",
            "mean_curvature_orientation": "sign_follows_consistent_mesh_winding",
            "mean_curvature_validity": "estimator_valid_vertices",
            "gaussian_boundary_policy": "boundary_invalid",
            "gaussian_area_normalization": "represented_valid_vertex_area",
            "fairness_integrator": "p1_face_gradient_dirichlet_F2",
            "fairness_input": "signed_rusinkiewicz_mean_curvature",
            "local_plate_twist_interpretation": "mesh_dependent_edge_dihedral_diagnostic",
        },
        "curvature_thresholds": {
            "h_threshold": float(h_thr),
            "k_threshold": float(k_thr),
            "h_threshold_factor": float(config.h_threshold_factor),
            "k_threshold_factor": float(config.k_threshold_factor),
            "h_threshold_formula": "factor/L_ref unless absolute override",
            "k_threshold_formula": "factor/L_ref^2 unless absolute override",
        },
        "curvature_valid_area": h_valid_area,
        "curvature_valid_area_fraction": h_valid_area / max(area_total, config.eps),
        "curvature_invalid_vertex_count": int(np.sum(~h_valid)),
        "developability_valid_area": developability_valid_area,
        "developability_valid_area_fraction": developability_valid_area
        / max(area_total, config.eps),
        "developability_invalid_vertex_count": int(np.sum(~k_valid)),
        "gaussian_boundary_vertex_count": int(np.sum(boundary)),
        "fairness_valid_area": fairness_valid_area,
        "fairness_valid_area_fraction": fairness_valid_area / max(area_total, config.eps),
        "section_settings": section_metadata,
        "curvature_reliability_status": mesh_quality["curvature_reliability"]["status"],
        "mesh_quality": mesh_quality,
        "units": units,
        **_edge_length_metadata(mesh),
    }

    discriminant = np.maximum(H**2 - K_raw, 0.0)
    principal_offset = np.sqrt(discriminant)
    developability_density = np.where(k_valid, np.abs(K_raw) * L_ref**2, np.nan)
    developability_positive_density = np.where(k_valid, np.maximum(K_raw, 0.0) * L_ref**2, np.nan)
    developability_negative_density = np.where(k_valid, np.maximum(-K_raw, 0.0) * L_ref**2, np.nan)
    developability_threshold_mask = np.full(len(K_raw), -1, dtype=np.int8)
    developability_threshold_mask[k_valid] = (np.abs(K_raw[k_valid]) > k_thr).astype(np.int8)
    normals = np.asarray(mesh.vertex_normals, dtype=float)

    local_fields = {
        "H": H,
        "K": K,
        "principal_curvature_k1": np.where(h_valid, H + principal_offset, np.nan),
        "principal_curvature_k2": np.where(h_valid, H - principal_offset, np.nan),
        "surface_normal_x": normals[:, 0],
        "surface_normal_y": normals[:, 1],
        "surface_normal_z": normals[:, 2],
        "H_valid": h_valid.astype(np.uint8),
        "K_valid": k_valid.astype(np.uint8),
        "K_positive": np.where(k_valid, np.maximum(K_raw, 0.0), np.nan),
        "K_negative": np.where(k_valid, np.maximum(-K_raw, 0.0), np.nan),
        "developability_density": developability_density,
        "developability_positive_density": developability_positive_density,
        "developability_negative_density": developability_negative_density,
        "developability_threshold_mask": developability_threshold_mask,
        "curvature_class_id": class_ids,
        "vertex_area": areas,
    }

    if experimental_enabled:
        fairness_density = np.full(len(mesh.faces), np.nan)
        fairness_density[fairness_face_valid] = (
            L_ref**4
            * fairness_raw[fairness_face_valid]
            / np.maximum(mesh.area_faces[fairness_face_valid], config.eps)
        )
        local_fields.update(
            {
                "curvature_energy_density": np.where(h_valid, (H * L_ref) ** 2, np.nan),
                "curvature_fairness_density": fairness_density,
                "curvature_energy_contribution": energy_contribution,
                "curvature_fairness_contribution": fairness_contribution,
                "face_twist": face_twist,
            }
        )

    if config.robust_metrics:
        robust_m, robust_meta, robust_fields = _robust_metrics(
            mesh,
            H,
            K,
            areas,
            L_ref,
            config,
        )
        metrics.update(robust_m)
        metadata.update(robust_meta)
        local_fields.update(robust_fields)

    return MetricResult(
        metrics=metrics,
        curvature_classes=classes,
        metadata=metadata,
        local_fields=local_fields,
    )
