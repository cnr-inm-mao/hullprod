"""Generic bounded convergence classification for native BRep metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .brep_quadrature import BRepIntegralResult
from .types import ProducibilityConfig
from .validity import metric_validity_record


def _finite(value: Any) -> bool:
    return value is not None and np.isfinite(float(value))


def _refinement_evidence(
    metadata: dict[str, Any],
    metric: str,
    *,
    pass_name: str,
) -> dict[str, Any]:
    return {
        "quadrature_status": metadata["convergence_status"],
        "one_level_relative_change": metadata["one_level_relative_change"].get(metric),
        "prior_level_relative_change": metadata["prior_level_relative_change"].get(metric),
        "one_level_shallower_value": metadata["one_level_shallower_values"].get(metric),
        "two_levels_shallower_value": metadata["two_levels_shallower_values"].get(metric),
        "maximum_local_relative_estimate": metadata["maximum_local_relative_estimate"].get(
            pass_name
        ),
        "unconverged_cells_at_maximum_depth": metadata[
            "unconverged_cells_at_maximum_depth_by_pass"
        ].get(pass_name, 0),
        "maximum_depth": metadata["maximum_depth"],
    }


def _persistent_growth(metadata: dict[str, Any], metric: str, threshold: float) -> bool:
    current = float(metadata["one_level_relative_change"].get(metric, 0.0))
    prior = float(metadata["prior_level_relative_change"].get(metric, 0.0))
    return current >= threshold and prior >= threshold


def classify_brep_validity(
    integral: BRepIntegralResult,
    section_metrics: dict[str, float],
    config: ProducibilityConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, float | None], dict[str, bool]]:
    """Classify every native metric independently from bounded evidence.

    The escalation is deliberately bounded: the adaptive tree supplies its
    final value and the same tree truncated by one and two terminal levels.
    A non-integrable singularity is reported only when invariant integral
    growth persists across both steps, the terminal cells dominate the
    contribution, and the differential evaluation shows degenerating local
    geometry.  Otherwise an unresolved calculation remains *unconverged*.
    """
    values = dict(integral.values)
    metadata = integral.metadata
    total_area = float(values["surface_area"])
    curvature_area = float(values["curvature_valid_area"])
    fairness_area = float(values["fairness_valid_area"])
    by_pass = metadata["unconverged_cells_at_maximum_depth_by_pass"]
    concentration = metadata["contribution_concentration"]
    regularity = metadata["regularity_evidence"]

    area_valid = bool(metadata["area_converged_against_independent_brep_measure"])
    core_unconverged = int(by_pass.get("core", 0)) > 0
    fairness_unconverged = int(by_pass.get("fairness", 0)) > 0
    core_jacobian_ratio = float(regularity["core"]["minimum_jacobian"] or 0.0) / max(
        float(regularity["core"]["maximum_jacobian"] or 0.0), config.eps
    )
    fairness_jacobian_ratio = float(regularity["fairness"]["minimum_jacobian"] or 0.0) / max(
        float(regularity["fairness"]["maximum_jacobian"] or 0.0), config.eps
    )
    core_geometric_warning = (
        float(regularity["core"]["maximum_first_fundamental_condition"] or 0.0) >= 1.0e3
        or core_jacobian_ratio <= 1.0e-3
    )
    fairness_geometric_warning = (
        float(regularity["fairness"]["maximum_first_fundamental_condition"] or 0.0) >= 1.0e5
        or fairness_jacobian_ratio <= 1.0e-4
    )
    ce_singular = (
        core_unconverged
        and core_geometric_warning
        and _persistent_growth(metadata, "curvature_energy", 0.08)
        and float(concentration["curvature_energy_unconverged_cell_fraction"]) >= 0.10
    )
    fairness_singular = (
        config.brep_compute_fairness
        and fairness_unconverged
        and fairness_geometric_warning
        and _persistent_growth(metadata, "curvature_fairness", 0.50)
        and float(concentration["fairness_unconverged_cell_fraction"]) >= 0.50
    )
    ce_stable = (
        _finite(values["curvature_energy"])
        and float(metadata["one_level_relative_change"]["curvature_energy"])
        <= max(20.0 * config.brep_quadrature_tolerance, 2.0e-3)
        and float(metadata["prior_level_relative_change"]["curvature_energy"])
        <= max(50.0 * config.brep_quadrature_tolerance, 5.0e-3)
    )
    fairness_stable = (
        _finite(values["curvature_fairness"])
        and float(metadata["one_level_relative_change"]["curvature_fairness"])
        <= max(50.0 * config.brep_quadrature_tolerance, 5.0e-3)
        and float(metadata["prior_level_relative_change"]["curvature_fairness"])
        <= max(50.0 * config.brep_quadrature_tolerance, 5.0e-3)
    )

    records: dict[str, dict[str, Any]] = {}
    records["surface_area"] = metric_validity_record(
        value=total_area if area_valid else None,
        validity="valid" if area_valid else "quadrature_unconverged",
        reason=(
            "Direct quadrature agrees with the independent OpenCascade BRep area."
            if area_valid
            else "Direct quadrature did not meet the independent BRep-area tolerance."
        ),
        represented_area=total_area,
        valid_area=total_area if area_valid else None,
        numerical_convergence={
            "area_relative_discrepancy": metadata["area_relative_discrepancy"],
            "area_relative_acceptance": metadata["area_relative_acceptance"],
        },
        representation="brep",
        backend="brep_native",
    )

    ce_evidence = _refinement_evidence(metadata, "curvature_energy", pass_name="core")
    ce_evidence["unconverged_contribution_fraction"] = concentration[
        "curvature_energy_unconverged_cell_fraction"
    ]
    if ce_singular:
        ce_validity = "geometric_singularity_nonintegrable"
        ce_value = None
        ce_reason = (
            "Persistent refinement growth and concentrated invariant H^2 density "
            "identify a non-integrable geometric curvature singularity."
        )
    elif area_valid and (not core_unconverged or ce_stable):
        ce_validity = "valid"
        ce_value = values["curvature_energy"]
        ce_reason = "The native signed-mean-curvature energy integral converged."
    else:
        ce_validity = "quadrature_unconverged"
        ce_value = None
        ce_reason = (
            "The bounded native quadrature did not establish a finite converged "
            "curvature-energy value."
        )
    records["curvature_energy"] = metric_validity_record(
        value=ce_value,
        validity=ce_validity,
        reason=ce_reason,
        represented_area=total_area,
        valid_area=curvature_area if ce_value is not None else None,
        numerical_convergence=ce_evidence,
        representation="brep",
        backend="brep_native",
    )

    k_evidence = _refinement_evidence(metadata, "developability_deviation", pass_name="core")
    k_evidence["unconverged_contribution_fraction"] = concentration[
        "developability_unconverged_cell_fraction"
    ]
    k_stable = (
        _finite(values["developability_deviation"])
        and float(metadata["one_level_relative_change"]["developability_deviation"]) <= 0.02
        and float(metadata["prior_level_relative_change"]["developability_deviation"]) <= 0.03
    )
    k_singular = (
        core_unconverged
        and core_geometric_warning
        and _persistent_growth(metadata, "developability_deviation", 0.08)
        and float(concentration["developability_unconverged_cell_fraction"]) >= 0.10
    )
    if k_singular:
        k_validity = "geometric_singularity_nonintegrable"
        k_reason = (
            "Persistent refinement growth and concentrated invariant |K| density "
            "identify a non-integrable Gaussian-curvature singularity."
        )
    elif (ce_singular or fairness_singular) and area_valid and k_stable:
        k_validity = "valid_improper_integral_convergent"
        k_reason = (
            "Absolute Gaussian curvature approaches a finite improper integral "
            "although a measure-zero geometric singularity is present."
        )
    elif area_valid and not core_unconverged:
        k_validity = "valid"
        k_reason = "The native absolute-Gaussian-curvature integral converged."
    elif area_valid and k_stable:
        k_validity = "valid"
        k_reason = (
            "The developability integral is stable across the bounded refinement "
            "sequence; unresolved cells do not materially affect its value."
        )
    else:
        k_validity = "quadrature_unconverged"
        k_reason = "Native quadrature did not establish a converged developability value."
    k_value = values["developability_deviation"] if k_validity.startswith("valid") else None
    records["developability_deviation"] = metric_validity_record(
        value=k_value,
        validity=k_validity,
        reason=k_reason,
        represented_area=total_area,
        valid_area=curvature_area if k_value is not None else None,
        numerical_convergence=k_evidence,
        representation="brep",
        backend="brep_native",
    )
    for name in (
        "developability_deviation_positive",
        "developability_deviation_negative",
    ):
        evidence = _refinement_evidence(metadata, name, pass_name="core")
        component_singular = k_singular and _persistent_growth(metadata, name, 0.08)
        component_stable = (
            _finite(values[name])
            and float(metadata["one_level_relative_change"][name]) <= 0.02
            and float(metadata["prior_level_relative_change"][name]) <= 0.03
        )
        if component_singular:
            component_validity = "geometric_singularity_nonintegrable"
            component_value = None
            component_reason = (
                "The signed Gaussian-curvature contribution has persistent "
                "non-integrable refinement growth."
            )
        elif k_value is not None or (area_valid and component_stable):
            component_validity = k_validity
            component_value = values[name]
            component_reason = k_reason
        else:
            component_validity = "quadrature_unconverged"
            component_value = None
            component_reason = (
                "Bounded quadrature did not establish this signed developability contribution."
            )
        records[name] = metric_validity_record(
            value=component_value,
            validity=component_validity,
            reason=component_reason,
            represented_area=total_area,
            valid_area=curvature_area if component_value is not None else None,
            numerical_convergence=evidence,
            representation="brep",
            backend="brep_native",
        )

    ratio_name = "developability_area_ratio"
    ratio_stable = (
        _finite(values[ratio_name])
        and float(metadata["one_level_relative_change"][ratio_name]) <= 0.01
        and float(metadata["prior_level_relative_change"][ratio_name]) <= 0.01
    )
    ratio_value = values[ratio_name] if area_valid and ratio_stable else None
    ratio_validity = (
        "caution_singular_measure_zero"
        if ratio_value is not None and (ce_singular or k_singular)
        else ("valid" if ratio_value is not None else "quadrature_unconverged")
    )
    records[ratio_name] = metric_validity_record(
        value=ratio_value,
        validity=ratio_validity,
        reason=(
            "The bounded threshold-area indicator converges; the singular set has "
            "zero represented area."
            if ratio_validity == "caution_singular_measure_zero"
            else (
                "The thresholded developability area ratio converged."
                if ratio_validity == "valid"
                else "The thresholded area ratio did not meet bounded convergence."
            )
        ),
        represented_area=total_area,
        valid_area=curvature_area if ratio_value is not None else None,
        numerical_convergence=_refinement_evidence(metadata, ratio_name, pass_name="core"),
        representation="brep",
        backend="brep_native",
    )

    class_validity = (
        "caution_singular_measure_zero"
        if ce_singular or fairness_singular or k_singular
        else ("valid" if area_valid and k_value is not None else "quadrature_unconverged")
    )
    records["curvature_classes"] = metric_validity_record(
        value=1.0 if class_validity != "quadrature_unconverged" else None,
        validity=class_validity,
        reason=(
            "Area fractions are finite; the detected singular feature has zero "
            "surface measure and is reported as a caution."
            if class_validity == "caution_singular_measure_zero"
            else "Curvature-class area fractions were integrated over valid CAD area."
        ),
        represented_area=total_area,
        valid_area=curvature_area if class_validity != "quadrature_unconverged" else None,
        numerical_convergence=k_evidence,
        representation="brep",
        backend="brep_native",
    )

    fair_evidence = _refinement_evidence(metadata, "curvature_fairness", pass_name="fairness")
    fair_evidence["unconverged_contribution_fraction"] = concentration[
        "fairness_unconverged_cell_fraction"
    ]
    if not config.brep_compute_fairness and ce_singular:
        fair_validity = "geometric_singularity_nonintegrable"
        fair_value = None
        fair_reason = (
            "The mean-curvature field is not square-integrable at the detected "
            "geometric singularity and therefore cannot satisfy the H1 regularity "
            "required by strict curvature-gradient fairness."
        )
        extra_codes = ("insufficient_C3_for_fairness",)
    elif not config.brep_compute_fairness:
        fair_validity = "not_evaluated"
        fair_value = None
        fair_reason = (
            "High-order native fairness is optional and was not requested for this assessment."
        )
        extra_codes = ()
    elif fairness_singular:
        fair_validity = "geometric_singularity_nonintegrable"
        fair_value = None
        fair_reason = (
            "The strict curvature-gradient integral is non-integrable at a genuine "
            "geometric singularity; this is not a software failure."
        )
        extra_codes = ("insufficient_C3_for_fairness",)
    elif (
        area_valid and fairness_area > config.eps and (not fairness_unconverged or fairness_stable)
    ):
        fair_validity = "valid"
        fair_value = values["curvature_fairness"]
        fair_reason = "The direct piecewise-D3 curvature-gradient integral converged."
        extra_codes = ()
    else:
        fair_validity = "quadrature_unconverged"
        fair_value = None
        fair_reason = (
            "The bounded D3 quadrature did not establish a finite converged fairness value."
        )
        extra_codes = ()
    records["curvature_fairness"] = metric_validity_record(
        value=fair_value,
        validity=fair_validity,
        additional_validity=extra_codes,
        reason=fair_reason,
        represented_area=total_area,
        valid_area=fairness_area if fair_value is not None else None,
        numerical_convergence=fair_evidence,
        representation="brep",
        backend="brep_native",
    )

    section_valid = int(section_metrics.get("valid_sections", 0)) > 0
    section_joins = int(section_metrics.get("sections_with_edge_joins", 0))
    for name in ("section_waviness", "section_waviness_fft"):
        records[name] = metric_validity_record(
            value=section_metrics.get(name) if section_valid else None,
            validity=(
                "caution_singular_measure_zero"
                if section_valid and section_joins > 0
                else ("valid" if section_valid else "not_evaluated")
            ),
            reason=(
                "Exact piecewise-BRep D2 sections produced finite station values, but "
                "cross-edge curvature regularity and resampling convergence require "
                "explicit evidence before canonical interpretation."
                if section_valid and section_joins > 0
                else "Exact single-edge BRep-plane section curves produced valid station values."
                if section_valid
                else "Section metrics require explicit experimental opt-in."
            ),
            representation="brep",
            backend="brep_native",
            valid_sections=int(section_metrics.get("valid_sections", 0)),
            requested_sections=int(config.n_stations),
            sections_with_edge_joins=section_joins,
            sampling_convergence_evidence_supplied=False,
        )
    records["local_plate_twist"] = metric_validity_record(
        value=None,
        validity="not_applicable",
        reason=(
            "mesh-dependent dihedral diagnostic; no canonical BRep-native "
            "equivalent is defined in HullProd 1.0"
        ),
        representation="brep",
        backend="brep_native",
    )

    public_values: dict[str, float | None] = {
        name: record["value"] for name, record in records.items() if name != "curvature_classes"
    }
    public_values.update(
        {
            "h_threshold": values["h_threshold"],
            "k_threshold": values["k_threshold"],
            "valid_sections": float(section_metrics.get("valid_sections", 0)),
        }
    )
    return (
        records,
        public_values,
        {
            "curvature_energy_singularity_detected": ce_singular,
            "developability_singularity_detected": k_singular,
            "fairness_singularity_detected": fairness_singular,
        },
    )
