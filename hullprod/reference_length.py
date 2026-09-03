"""Reference-length resolution shared by native-BRep and mesh backends."""

from __future__ import annotations

from typing import Any

import numpy as np


def _automatic_reference_length(
    points: np.ndarray,
    *,
    source: str,
    sampling: dict[str, Any] | None,
) -> tuple[float, dict[str, Any]]:
    """Compute the automatic principal span and its complete provenance."""
    samples = np.asarray(points, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) < 2:
        raise ValueError("Automatic reference length requires at least two 3D points.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Automatic reference-length samples must be finite.")

    centered = samples - np.mean(samples, axis=0)
    covariance = centered.T @ centered / float(len(samples))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    major_variance = float(eigenvalues[0])
    if not np.isfinite(major_variance) or major_variance <= 0.0:
        raise ValueError("Geometry is degenerate; no positive automatic reference length exists.")

    # A relative eigengap makes the decision dimensionless and scale invariant.
    relative_eigengap = float((eigenvalues[0] - eigenvalues[1]) / eigenvalues[0])
    # This margin is deliberately larger than binary STL coordinate noise so a
    # reserialized sphere does not acquire a physically meaningless major axis.
    degeneracy_tolerance = 1.0e-5
    fallback = relative_eigengap <= degeneracy_tolerance
    if fallback:
        value = float(2.0 * np.max(np.linalg.norm(centered, axis=1)))
        method = "centroid_radial_diameter_isotropic_fallback"
    else:
        major_axis = eigenvectors[:, order[0]]
        value = float(np.ptp(centered @ major_axis))
        method = "principal_axis_projected_span"
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Geometry is degenerate; no positive automatic reference length exists.")

    metadata: dict[str, Any] = {
        "value": value,
        "mode": "auto_principal_span",
        "method": method,
        "source": source,
        "is_lpp": False,
        "sample_count": len(samples),
        "principal_variances": [float(item) for item in eigenvalues],
        "relative_leading_eigengap": relative_eigengap,
        "isotropic_fallback_used": fallback,
        "isotropic_fallback_eigengap_tolerance": degeneracy_tolerance,
    }
    if sampling:
        metadata["sampling"] = sampling
    return value, metadata


def resolve_reference_length(
    points: np.ndarray | None,
    explicit: float | None,
    *,
    source: str,
    sampling: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Resolve an explicit length or a rigid-motion-invariant geometric default.

    The automatic value is the extent along the dominant principal axis of the
    supplied geometry samples.  When that axis is not unique, a centroid-radial
    diameter is used so isotropic geometries remain deterministic and invariant
    under rigid rotation.  Neither automatic value is an inferred ship length
    between perpendiculars.
    """
    automatic: tuple[float, dict[str, Any]] | None = None
    if points is not None:
        automatic = _automatic_reference_length(points, source=source, sampling=sampling)

    if explicit is not None:
        value = float(explicit)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Reference length must be finite and positive.")
        metadata: dict[str, Any] = {
            "value": value,
            "mode": "explicit_user",
            "method": "explicit_user_value",
            "source": "ProducibilityConfig.length_ref_or_cli_lref",
            "is_lpp": False,
        }
        if automatic is not None:
            automatic_value, automatic_metadata = automatic
            ratio = value / automatic_value
            warning_triggered = ratio < 0.1 or ratio > 10.0
            metadata["automatic_comparison"] = automatic_metadata
            metadata["plausibility"] = {
                "user_to_automatic_ratio": ratio,
                "warning_triggered": warning_triggered,
                "criterion": "warn_if_ratio_below_0.1_or_above_10",
                "lower_ratio": 0.1,
                "upper_ratio": 10.0,
            }
        return value, metadata

    if automatic is None:
        raise ValueError("Automatic reference length requires geometry samples.")
    return automatic


def reference_length_preflight(
    reference: dict[str, Any],
    units: dict[str, Any],
) -> tuple[list[str], str | None]:
    """Build concise pre-integration unit/reference messages and any warning."""
    source = units.get("source_declared_unit")
    source_text = str(source) if source else "not declared"
    working = str(units.get("working_length_unit_symbol", "input-length units"))
    value = float(reference["value"])
    mode = (
        "auto principal-axis span"
        if reference.get("mode") == "auto_principal_span"
        else "user supplied"
    )
    messages = [
        f"Source unit: {source_text}",
        f"Working unit: {working}",
        f"Reference length: {value:.8g} {working} ({mode})",
    ]
    plausibility = reference.get("plausibility", {})
    if not plausibility.get("warning_triggered"):
        return messages, None
    automatic = reference["automatic_comparison"]
    warning = (
        f"User-supplied reference length is {value:.8g} {working}, while the "
        f"automatic geometric span is approximately {float(automatic['value']):.8g} "
        f"{working}. --lref is interpreted in the geometry working unit. "
        "Please verify the intended value."
    )
    plausibility["message"] = warning
    return messages, warning
