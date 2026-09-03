"""Stable HullProd 1.0 public schema and field conventions."""

from __future__ import annotations

from enum import IntEnum

SCHEMA_VERSION = "1.0.0"
FIELD_SCHEMA_VERSION = "1.0.0"
OUTPUT_LAYOUT_VERSION = "1.0.0"


class CurvatureClassID(IntEnum):
    """Stable categorical identifiers for the represented valid K domain."""

    INVALID = -1
    FLAT = 0
    SINGLE_CURVATURE = 1
    ELLIPTIC_DOUBLE_CURVATURE = 2
    SADDLE_REVERSE_DOUBLE_CURVATURE = 3


CURVATURE_CLASS_MAPPING = {
    int(CurvatureClassID.INVALID): "invalid",
    int(CurvatureClassID.FLAT): "flat",
    int(CurvatureClassID.SINGLE_CURVATURE): "single curvature",
    int(CurvatureClassID.ELLIPTIC_DOUBLE_CURVATURE): "elliptic double curvature",
    int(CurvatureClassID.SADDLE_REVERSE_DOUBLE_CURVATURE): ("saddle/reverse double curvature"),
}

RECOMMENDED_SIGNATURE_METRICS = (
    "developability_deviation",
    "developability_deviation_positive",
    "developability_deviation_negative",
    "curvature_classes",
)

RECOMMENDED_SIGNATURE_KEYS = (
    "I_D",
    "I_D_plus",
    "I_D_minus",
    "a_C_flat",
    "a_C_single",
    "a_C_elliptic",
    "a_C_saddle",
)

METRIC_STATUS = {
    "curvature_energy": "screened_experimental_nonrecommended",
    "developability_deviation": "recommended_signature",
    "developability_deviation_positive": "recommended_signature",
    "developability_deviation_negative": "recommended_signature",
    "developability_area_ratio": "auxiliary_redundant",
    "curvature_classes": "recommended_signature",
    "section_waviness": "screened_experimental_nonrecommended",
    "section_waviness_fft": "screened_experimental_nonrecommended",
    "curvature_fairness": "screened_experimental_nonrecommended",
    "local_plate_twist": "diagnostic_mesh_dependent",
}


def signature_metadata() -> dict[str, object]:
    """Return the explicit paper-aligned metric-role contract."""
    return {
        "recommended_signature": {
            "metric_keys": list(RECOMMENDED_SIGNATURE_METRICS),
            "output_keys": list(RECOMMENDED_SIGNATURE_KEYS),
            "vector": "[I_D, I_D_plus, I_D_minus, a_C^T]^T",
            "universal_weighted_score": False,
        },
        "metric_status": dict(METRIC_STATUS),
    }


def schema_metadata() -> dict[str, str | dict[int, str]]:
    """Return the version and categorical metadata embedded in every result."""
    return {
        "schema_version": SCHEMA_VERSION,
        "field_schema_version": FIELD_SCHEMA_VERSION,
        "output_layout_version": OUTPUT_LAYOUT_VERSION,
        "curvature_class_mapping": dict(CURVATURE_CLASS_MAPPING),
    }
