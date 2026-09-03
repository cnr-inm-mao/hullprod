"""Stable field and output manifest construction for HullProd 1.0."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .schema import CURVATURE_CLASS_MAPPING, FIELD_SCHEMA_VERSION, OUTPUT_LAYOUT_VERSION

_FIELD_CATALOG: dict[str, dict[str, Any]] = {
    "H": {
        "formula": "H=(k1+k2)/2",
        "interpretation": "signed conventional mean curvature",
        "units": "inverse length",
        "normalization": "none",
        "global": "curvature_energy",
    },
    "K": {
        "formula": "K=k1*k2",
        "interpretation": "signed Gaussian curvature; invalid samples are not zero-filled",
        "units": "inverse length squared",
        "normalization": "none",
        "global": "developability_deviation",
    },
    "principal_curvature_k1": {
        "formula": "k1=H+sqrt(max(H^2-K,0))",
        "interpretation": "first principal curvature",
        "units": "inverse length",
        "normalization": "none",
        "global": None,
    },
    "principal_curvature_k2": {
        "formula": "k2=H-sqrt(max(H^2-K,0))",
        "interpretation": "second principal curvature",
        "units": "inverse length",
        "normalization": "none",
        "global": None,
    },
    "curvature_energy_density": {
        "formula": "(H*L_ref)^2",
        "interpretation": (
            "local nondimensional mean-curvature content; the global curvature "
            "energy is its valid-domain area average"
        ),
        "units": "dimensionless",
        "normalization": "valid-domain area average",
        "global": "curvature_energy",
    },
    "developability_density": {
        "formula": "abs(K)*L_ref^2",
        "interpretation": "local intensity of double curvature / departure from developability",
        "units": "dimensionless",
        "normalization": "represented-valid-area average",
        "global": "developability_deviation",
    },
    "developability_positive_density": {
        "formula": "max(K,0)*L_ref^2",
        "interpretation": "local elliptic positive developability contribution",
        "units": "dimensionless",
        "normalization": "represented-valid-area average",
        "global": "developability_deviation_positive",
    },
    "developability_positive": {
        "formula": "max(K,0)*L_ref^2",
        "interpretation": "local elliptic positive developability contribution",
        "units": "dimensionless",
        "normalization": "represented-valid-area average",
        "global": "developability_deviation_positive",
    },
    "developability_negative_density": {
        "formula": "max(-K,0)*L_ref^2",
        "interpretation": "local saddle/reverse negative developability contribution",
        "units": "dimensionless",
        "normalization": "represented-valid-area average",
        "global": "developability_deviation_negative",
    },
    "developability_negative": {
        "formula": "max(-K,0)*L_ref^2",
        "interpretation": "local saddle/reverse negative developability contribution",
        "units": "dimensionless",
        "normalization": "represented-valid-area average",
        "global": "developability_deviation_negative",
    },
    "developability_threshold_mask": {
        "formula": "1 if abs(K)>K_threshold, 0 otherwise, -1 invalid",
        "interpretation": "thresholded departure-from-developability mask",
        "units": "categorical dimensionless",
        "normalization": "represented-valid-area average",
        "global": "developability_area_ratio",
        "mapping": {-1: "invalid", 0: "below/equal threshold", 1: "above threshold"},
    },
    "curvature_class_id": {
        "formula": "scale-aware H/K threshold classification",
        "interpretation": "forming-relevant curvature composition class",
        "units": "categorical dimensionless",
        "normalization": "represented-valid-area fractions",
        "global": "curvature_classes",
        "mapping": CURVATURE_CLASS_MAPPING,
    },
    "curvature_fairness_density": {
        "formula": "L_ref^4*|grad_s H|^2",
        "interpretation": "HIGH-SENSITIVITY GEOMETRIC DIAGNOSTIC",
        "units": "dimensionless",
        "normalization": "valid-domain area integral divided by valid area",
        "global": "curvature_fairness",
    },
    "face_twist": {
        "formula": "mean incident face-normal dihedral angle",
        "interpretation": "mesh-dependent local plate-twist diagnostic",
        "units": "radians",
        "normalization": "none",
        "global": "local_plate_twist",
    },
    "H_valid": {
        "formula": "1 where H is represented-valid, else 0",
        "interpretation": "mean-curvature validity mask",
        "units": "boolean dimensionless",
        "normalization": "none",
        "global": "curvature_energy",
    },
    "K_valid": {
        "formula": "1 where K is represented-valid, else 0",
        "interpretation": "Gaussian-curvature validity mask",
        "units": "boolean dimensionless",
        "normalization": "none",
        "global": "developability_deviation",
    },
    "valid_mask": {
        "formula": "1 where K is represented-valid, else 0",
        "interpretation": "Gaussian-curvature validity mask",
        "units": "boolean dimensionless",
        "normalization": "none",
        "global": "developability_deviation",
    },
    "K_positive": {
        "formula": "max(K,0)",
        "interpretation": "positive/elliptic Gaussian-curvature component",
        "units": "inverse length squared",
        "normalization": "none",
        "global": "developability_deviation_positive",
    },
    "K_negative": {
        "formula": "max(-K,0)",
        "interpretation": "negative/saddle Gaussian-curvature magnitude",
        "units": "inverse length squared",
        "normalization": "none",
        "global": "developability_deviation_negative",
    },
    "surface_normal_x": {
        "formula": "x component of unit surface normal",
        "interpretation": "surface orientation support field",
        "units": "dimensionless",
        "normalization": "unit vector component",
        "global": None,
    },
    "surface_normal_y": {
        "formula": "y component of unit surface normal",
        "interpretation": "surface orientation support field",
        "units": "dimensionless",
        "normalization": "unit vector component",
        "global": None,
    },
    "surface_normal_z": {
        "formula": "z component of unit surface normal",
        "interpretation": "surface orientation support field",
        "units": "dimensionless",
        "normalization": "unit vector component",
        "global": None,
    },
    "curvature_energy_contribution": {
        "formula": "A_i*(H_i*L_ref)^2/A_valid",
        "interpretation": "derived per-vertex contribution to the global mesh sum",
        "units": "dimensionless",
        "normalization": "already includes valid-area normalization",
        "global": "curvature_energy",
    },
    "curvature_fairness_contribution": {
        "formula": "L_ref^4*A_f*|grad_f H|^2/A_valid",
        "interpretation": "derived per-cell contribution to the global mesh sum",
        "units": "dimensionless",
        "normalization": "already includes valid-area normalization",
        "global": "curvature_fairness",
    },
    "vertex_area": {
        "formula": "one third of each incident triangle area",
        "interpretation": "associated vertex area used by mesh global averages",
        "units": "length squared",
        "normalization": "none",
        "global": "surface_area",
    },
    "robust_vertex_mask": {
        "formula": "experimental configured vertex-area validity mask",
        "interpretation": "opt-in robust diagnostic mask; never canonical",
        "units": "boolean dimensionless",
        "normalization": "none",
        "global": None,
    },
    "fairness_valid": {
        "formula": "1 where native D3 fairness sampling succeeded, else 0",
        "interpretation": "display-sample validity for BRep fairness density",
        "units": "boolean dimensionless",
        "normalization": "none",
        "global": "curvature_fairness",
    },
    "brep_face_index": {
        "formula": "zero-based source BRep face index",
        "interpretation": "display sample provenance back to the native CAD face",
        "units": "categorical index",
        "normalization": "none",
        "global": None,
    },
}


def _metric_validity(result, global_name: str | None) -> dict[str, Any] | None:
    if global_name is None:
        return None
    records = result.metadata.get("metric_validity", {})
    return records.get(global_name) or (
        records.get("developability") if global_name.startswith("developability_") else None
    )


def build_field_manifest(
    result,
    exported_files: list[str] | None = None,
) -> dict[str, Any]:
    """Describe every exported local field and its scientific role."""
    representation = result.metadata.get("representation", "mesh")
    backend = result.metadata.get("backend", "unknown")
    from .export import public_surface_fields

    local_fields = public_surface_fields(result)
    associations = {name: "point" for name in local_fields}
    exported = list(exported_files or [])
    fields = []
    for name, values in local_fields.items():
        array = np.asarray(values)
        spec = _FIELD_CATALOG.get(name, {})
        global_name = spec.get("global")
        validity = _metric_validity(result, global_name)
        role = (
            "derived_visualization_sample"
            if representation == "brep"
            else (
                "canonical_mesh_diagnostic_field"
                if name in {"face_twist"}
                else "canonical_or_interpretive_mesh_field"
            )
        )
        record = {
            "name": name,
            "formula": spec.get("formula", "implementation diagnostic; see estimator metadata"),
            "plain_language_interpretation": spec.get(
                "interpretation", "supporting numerical or provenance field"
            ),
            "association": associations.get(name, "point"),
            "numeric_type": str(array.dtype),
            "units_or_dimensionless_status": spec.get("units", "implementation-defined"),
            "representation": representation,
            "backend": backend,
            "validity_status": validity.get("status") if validity else "supporting_field",
            "field_role": role,
            "normalization": spec.get("normalization", "none"),
            "corresponding_global_metric": global_name,
            "exported_files_containing_field": exported,
        }
        mapping = spec.get("mapping")
        if mapping is not None:
            record["categorical_mapping"] = dict(mapping)
        fields.append(record)
    return {
        "field_schema_version": FIELD_SCHEMA_VERSION,
        "representation": representation,
        "backend": backend,
        "coordinate_units": result.metadata.get("units"),
        "canonical_metrics_depend_on_display_mesh": result.metadata.get(
            "canonical_metrics_depend_on_display_mesh", False
        ),
        "fields": fields,
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_output_manifest(out_dir: Path, relative_paths: list[str]) -> dict[str, Any]:
    """List every file in the stable result directory with deterministic metadata."""
    files = []
    for relative in sorted(set(relative_paths)):
        path = out_dir / relative
        files.append(
            {
                "path": relative,
                "exists": path.is_file() or relative == "output_manifest.json",
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256_path(path) if path.is_file() else None,
            }
        )
    return {"output_layout_version": OUTPUT_LAYOUT_VERSION, "files": files}
