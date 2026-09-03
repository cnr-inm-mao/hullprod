"""Representation-aware, metric-specific scientific validity records.

Validity is deliberately separate from the metric value.  A numerical method
can complete successfully while the continuous metric is not finite on the
represented geometry, and one invalid metric does not invalidate the others.
"""

from __future__ import annotations

from typing import Any

import numpy as np

VALIDITY_VOCABULARY = (
    "valid",
    "valid_improper_integral_convergent",
    "caution_singular_measure_zero",
    "quadrature_unconverged",
    "parameterization_degenerate",
    "geometric_singularity_nonintegrable",
    "insufficient_C2",
    "insufficient_C3_for_fairness",
    "mesh_representation_sensitive",
    "not_evaluated",
    "not_applicable",
)

_STATUS_LABELS = {
    "valid": "VALID",
    "valid_improper_integral_convergent": "VALID*",
    "caution_singular_measure_zero": "CAUTION",
    "quadrature_unconverged": "UNCONVERGED",
    "parameterization_degenerate": "CAUTION",
    "geometric_singularity_nonintegrable": "SINGULAR",
    "insufficient_C2": "UNAVAILABLE",
    "insufficient_C3_for_fairness": "UNAVAILABLE",
    "mesh_representation_sensitive": "MESH-SENSITIVE",
    "not_evaluated": "NOT EVALUATED",
    "not_applicable": "N/A",
}


def metric_validity_record(
    *,
    value: float | int | None,
    validity: str,
    reason: str,
    representation: str,
    backend: str,
    represented_area: float | None = None,
    valid_area: float | None = None,
    numerical_convergence: dict[str, Any] | None = None,
    additional_validity: tuple[str, ...] = (),
    **extra: Any,
) -> dict[str, Any]:
    """Build one stable machine-readable validity record."""
    codes = (validity, *additional_validity)
    unknown = [code for code in codes if code not in VALIDITY_VOCABULARY]
    if unknown:
        raise ValueError(f"Unknown metric-validity code(s): {unknown}")
    finite_value = float(value) if value is not None and np.isfinite(float(value)) else None
    represented = float(represented_area) if represented_area is not None else None
    valid = float(valid_area) if valid_area is not None else None
    record: dict[str, Any] = {
        "value": finite_value,
        "status": validity,
        "display_status": _STATUS_LABELS[validity],
        "validity": validity,
        "validity_codes": list(codes),
        "reason": reason,
        "computable": finite_value is not None,
        "represented_area": represented,
        "valid_area": valid,
        "valid_area_fraction": (
            valid / represented
            if valid is not None and represented is not None and represented > 0.0
            else None
        ),
        "numerical_convergence": numerical_convergence or {},
        "representation": representation,
        "backend": backend,
    }
    record.update(extra)
    return record


def public_status_label(validity: dict[str, Any] | None) -> str:
    """Return the concise terminal/report label for a validity record."""
    if not validity:
        return "UNKNOWN"
    return str(validity.get("display_status", validity.get("status", "UNKNOWN")))


def brep_quadrature_note(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Return a compact secondary note without changing scientific validity."""
    quadrature = metadata.get("quadrature") or {}
    if quadrature.get("convergence_status") != "caution":
        return None
    by_pass = quadrature.get("unconverged_cells_at_maximum_depth_by_pass") or {}
    cell_count = int(by_pass.get("core", quadrature.get("unconverged_cells_at_maximum_depth", 0)))
    changes = quadrature.get("one_level_relative_change") or {}
    relative_change = changes.get("developability_deviation")
    record = (metadata.get("metric_validity") or {}).get("developability_deviation") or {}
    scientifically_valid = record.get("status") in {
        "valid",
        "valid_improper_integral_convergent",
    }
    if scientifically_valid:
        summary = (
            "The global retained developability integral satisfies the current "
            "scientific stability criterion, but some local quadrature cells "
            "reached the bounded maximum refinement depth."
        )
    else:
        summary = (
            "Some local quadrature cells reached the bounded maximum refinement "
            "depth; consult the metric-specific validity record before using the result."
        )
    return {
        "status": "caution",
        "display_status": "CAUTION",
        "scientific_status_unchanged": True,
        "scientifically_valid": scientifically_valid,
        "unconverged_core_cells_at_maximum_depth": cell_count,
        "developability_one_level_relative_change": relative_change,
        "summary": summary,
    }
