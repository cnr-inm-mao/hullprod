"""Deterministic direct quadrature over trimmed OpenCascade BRep faces."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappush, heapreplace
from itertools import pairwise
from time import perf_counter
from typing import Any

import numpy as np

from .brep_geometry import (
    BRepModel,
    evaluate_surface_differential,
    evaluate_surface_jacobian,
    face_continuity_intervals,
)
from .types import ProducibilityConfig

_N_QUANTITIES = 13
_AREA = 0
_CURVATURE_AREA = 1
_H2 = 2
_K_ABS = 3
_K_POSITIVE = 4
_K_NEGATIVE = 5
_K_THRESHOLD_AREA = 6
_CLASS_FLAT = 7
_CLASS_SINGLE = 8
_CLASS_ELLIPTIC = 9
_CLASS_SADDLE = 10
_FAIRNESS_AREA = 11
_GRADIENT_H2 = 12


@dataclass
class _Statistics:
    evaluated_points: int = 0
    outside_points: int = 0
    singular_area_points: int = 0
    invalid_curvature_points: int = 0
    invalid_fairness_points: int = 0
    lprop_mismatch_points: int = 0
    accepted_cells: int = 0
    subdivided_cells: int = 0
    unconverged_cells_at_maximum_depth: int = 0
    maximum_depth_reached: int = 0
    minimum_jacobian: float = float("inf")
    maximum_jacobian: float = 0.0
    minimum_metric_determinant: float = float("inf")
    maximum_metric_condition: float = 0.0
    maximum_abs_mean_curvature: float = 0.0
    maximum_abs_gaussian_curvature: float = 0.0
    maximum_gradient_mean: float = 0.0
    maximum_depth_integrals: np.ndarray = field(
        default_factory=lambda: np.zeros(_N_QUANTITIES, dtype=float)
    )
    unconverged_integrals: np.ndarray = field(
        default_factory=lambda: np.zeros(_N_QUANTITIES, dtype=float)
    )
    top_cells: list[tuple[float, int, dict[str, Any]]] = field(default_factory=list)
    top_cell_counter: int = 0
    face_failures: list[dict[str, Any]] = field(default_factory=list)

    def record_cell(self, record: dict[str, Any], *, score: float) -> None:
        """Retain a bounded set of dominant physical integration cells."""
        item = (float(score), self.top_cell_counter, record)
        self.top_cell_counter += 1
        if len(self.top_cells) < 40:
            heappush(self.top_cells, item)
        elif score > self.top_cells[0][0]:
            heapreplace(self.top_cells, item)


@dataclass(frozen=True)
class BRepIntegralResult:
    """Direct BRep integrals and quadrature audit metadata."""

    values: dict[str, float]
    class_areas: dict[str, float]
    metadata: dict[str, Any]


def _inside(classifier: Any, u: float, v: float) -> bool:
    from OCP.gp import gp_Pnt2d
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON

    return classifier.Perform(gp_Pnt2d(float(u), float(v)), False) in (
        TopAbs_IN,
        TopAbs_ON,
    )


def _point_quantities(
    face: Any,
    surface: Any,
    classifier: Any,
    u: float,
    v: float,
    *,
    mode: str,
    length_ref: float,
    h_threshold: float,
    k_threshold: float,
    statistics: _Statistics,
) -> np.ndarray | None:
    statistics.evaluated_points += 1
    if not _inside(classifier, u, v):
        statistics.outside_points += 1
        return None
    output = np.zeros(_N_QUANTITIES, dtype=float)
    differential = None
    try:
        differential = evaluate_surface_differential(
            face,
            u,
            v,
            need_third=mode == "fairness",
            surface=surface,
            crosscheck_lprop=False,
        )
    except Exception:
        if mode == "fairness":
            statistics.invalid_fairness_points += 1
            return output
        try:
            differential = evaluate_surface_differential(
                face,
                u,
                v,
                need_third=False,
                surface=surface,
                crosscheck_lprop=False,
            )
        except Exception:
            try:
                _, jacobian = evaluate_surface_jacobian(face, u, v, surface=surface)
            except Exception:
                statistics.singular_area_points += 1
                return None
            if jacobian <= 1.0e-30:
                statistics.singular_area_points += 1
                return None
            output[_AREA] = jacobian
            statistics.invalid_curvature_points += 1
            return output
    jacobian = differential.jacobian
    statistics.minimum_jacobian = min(statistics.minimum_jacobian, jacobian)
    statistics.maximum_jacobian = max(statistics.maximum_jacobian, jacobian)
    statistics.minimum_metric_determinant = min(
        statistics.minimum_metric_determinant,
        differential.metric_determinant,
    )
    statistics.maximum_metric_condition = max(
        statistics.maximum_metric_condition,
        differential.metric_condition,
    )
    output[_AREA] = jacobian
    if mode == "fairness":
        if differential.gradient_mean_squared is not None and np.isfinite(
            differential.gradient_mean_squared
        ):
            output[_FAIRNESS_AREA] = jacobian
            output[_GRADIENT_H2] = differential.gradient_mean_squared * jacobian
            statistics.maximum_gradient_mean = max(
                statistics.maximum_gradient_mean,
                float(np.sqrt(max(differential.gradient_mean_squared, 0.0))),
            )
        else:
            statistics.invalid_fairness_points += 1
        return output
    h_value = differential.mean_curvature
    k_value = differential.gaussian_curvature
    statistics.maximum_abs_mean_curvature = max(statistics.maximum_abs_mean_curvature, abs(h_value))
    statistics.maximum_abs_gaussian_curvature = max(
        statistics.maximum_abs_gaussian_curvature, abs(k_value)
    )
    output[_CURVATURE_AREA] = jacobian
    output[_H2] = (h_value * length_ref) ** 2 * jacobian
    output[_K_ABS] = abs(k_value) * length_ref**2 * jacobian
    output[_K_POSITIVE] = max(k_value, 0.0) * length_ref**2 * jacobian
    output[_K_NEGATIVE] = max(-k_value, 0.0) * length_ref**2 * jacobian
    output[_K_THRESHOLD_AREA] = float(abs(k_value) > k_threshold) * jacobian
    if abs(k_value) <= k_threshold and abs(h_value) <= h_threshold:
        output[_CLASS_FLAT] = jacobian
    elif abs(k_value) <= k_threshold:
        output[_CLASS_SINGLE] = jacobian
    elif k_value > k_threshold:
        output[_CLASS_ELLIPTIC] = jacobian
    else:
        output[_CLASS_SADDLE] = jacobian
    return output


def _integrate_cell(
    face: Any,
    surface: Any,
    classifier: Any,
    bounds: tuple[float, float, float, float],
    *,
    mode: str,
    order: int,
    length_ref: float,
    h_threshold: float,
    k_threshold: float,
    statistics: _Statistics,
) -> tuple[np.ndarray, int, int]:
    u0, u1, v0, v1 = bounds
    nodes, weights = np.polynomial.legendre.leggauss(order)
    u_values = 0.5 * ((u1 - u0) * nodes + u1 + u0)
    v_values = 0.5 * ((v1 - v0) * nodes + v1 + v0)
    uv_scale = 0.25 * (u1 - u0) * (v1 - v0)
    integral = np.zeros(_N_QUANTITIES, dtype=float)
    inside_count = 0
    outside_count = 0
    for i, u in enumerate(u_values):
        for j, v in enumerate(v_values):
            value = _point_quantities(
                face,
                surface,
                classifier,
                float(u),
                float(v),
                mode=mode,
                length_ref=length_ref,
                h_threshold=h_threshold,
                k_threshold=k_threshold,
                statistics=statistics,
            )
            if value is None:
                outside_count += 1
                continue
            inside_count += 1
            integral += weights[i] * weights[j] * value
    return uv_scale * integral, inside_count, outside_count


def _relative_error(low: np.ndarray, high: np.ndarray, *, mode: str) -> float:
    active = np.array(
        ([_AREA, _GRADIENT_H2] if mode == "fairness" else [_AREA, _H2, _K_ABS, _K_THRESHOLD_AREA]),
        dtype=int,
    )
    # An exactly zero density (notably fairness on spheres/cylinders) should
    # not trigger refinement down to round-off.  Every active quantity is an
    # area integral of a dimensionless density, so a small area-proportional
    # absolute floor is dimensionally consistent.
    area_scale = max(abs(low[_AREA]), abs(high[_AREA]), 1.0e-30)
    scale = np.maximum.reduce(
        (
            np.abs(low[active]),
            np.abs(high[active]),
            np.full(len(active), 1.0e-12 * area_scale),
        )
    )
    return float(np.max(np.abs(high[active] - low[active]) / scale))


def _adaptive_cell(
    face: Any,
    surface: Any,
    classifier: Any,
    bounds: tuple[float, float, float, float],
    *,
    mode: str,
    order: int,
    tolerance: float,
    maximum_depth: int,
    depth: int,
    length_ref: float,
    h_threshold: float,
    k_threshold: float,
    statistics: _Statistics,
    face_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    low_order = max(2, order - 2)
    low, low_inside, low_outside = _integrate_cell(
        face,
        surface,
        classifier,
        bounds,
        mode=mode,
        order=low_order,
        length_ref=length_ref,
        h_threshold=h_threshold,
        k_threshold=k_threshold,
        statistics=statistics,
    )
    high, high_inside, high_outside = _integrate_cell(
        face,
        surface,
        classifier,
        bounds,
        mode=mode,
        order=order,
        length_ref=length_ref,
        h_threshold=h_threshold,
        k_threshold=k_threshold,
        statistics=statistics,
    )
    error = _relative_error(low, high, mode=mode)
    mixed_trim = (
        (low_inside > 0 and low_outside > 0)
        or (high_inside > 0 and high_outside > 0)
        or ((low_inside == 0) != (high_inside == 0))
    )
    refine = depth < maximum_depth and (error > tolerance or mixed_trim)
    if not refine:
        if depth >= maximum_depth and (error > tolerance or mixed_trim):
            statistics.unconverged_cells_at_maximum_depth += 1
            statistics.unconverged_integrals += high
        statistics.accepted_cells += 1
        statistics.maximum_depth_reached = max(statistics.maximum_depth_reached, depth)
        if depth >= maximum_depth:
            statistics.maximum_depth_integrals += high
        score = float(high[_GRADIENT_H2] if mode == "fairness" else max(high[_H2], high[_K_ABS]))
        if score > 0.0:
            statistics.record_cell(
                {
                    "face_index": face_index,
                    "mode": mode,
                    "uv_bounds": [float(value) for value in bounds],
                    "depth": depth,
                    "relative_estimate": error,
                    "mixed_trim": mixed_trim,
                    "area": float(high[_AREA]),
                    "curvature_energy_integral": float(high[_H2]),
                    "developability_integral": float(high[_K_ABS]),
                    "fairness_integral_unscaled": float(high[_GRADIENT_H2]),
                },
                score=score,
            )
        return high, high, high, error
    statistics.subdivided_cells += 1
    u0, u1, v0, v1 = bounds
    um = 0.5 * (u0 + u1)
    vm = 0.5 * (v0 + v1)
    total = np.zeros(_N_QUANTITIES, dtype=float)
    previous_total = np.zeros(_N_QUANTITIES, dtype=float)
    previous_previous_total = np.zeros(_N_QUANTITIES, dtype=float)
    errors = []
    for child in (
        (u0, um, v0, vm),
        (um, u1, v0, vm),
        (u0, um, vm, v1),
        (um, u1, vm, v1),
    ):
        value, child_previous, child_previous_previous, child_error = _adaptive_cell(
            face,
            surface,
            classifier,
            child,
            mode=mode,
            order=order,
            tolerance=tolerance,
            maximum_depth=maximum_depth,
            depth=depth + 1,
            length_ref=length_ref,
            h_threshold=h_threshold,
            k_threshold=k_threshold,
            statistics=statistics,
            face_index=face_index,
        )
        total += value
        previous_total += child_previous
        previous_previous_total += child_previous_previous
        errors.append(child_error)
    # This is the same adaptive tree truncated by exactly one terminal level.
    # It provides a bounded refinement-growth diagnostic without a second full
    # integration.  Accepted cells above the terminal level are unchanged.
    if depth == maximum_depth - 1:
        previous_total = high
        previous_previous_total = high
    elif depth == maximum_depth - 2:
        previous_previous_total = high
    return total, previous_total, previous_previous_total, max(errors, default=error)


def integrate_brep_metrics(
    model: BRepModel,
    config: ProducibilityConfig,
    *,
    length_ref: float,
) -> BRepIntegralResult:
    """Integrate canonical metric densities directly over trimmed BRep faces."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepTopAdaptor import BRepTopAdaptor_FClass2d

    if config.brep_quadrature_order < 3:
        raise ValueError("brep_quadrature_order must be at least 3")
    if config.brep_quadrature_tolerance <= 0.0:
        raise ValueError("brep_quadrature_tolerance must be positive")
    if config.brep_quadrature_max_depth < 0:
        raise ValueError("brep_quadrature_max_depth must be non-negative")
    if config.brep_quadrature_base_subdivisions < 1:
        raise ValueError("brep_quadrature_base_subdivisions must be positive")
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
    face_audit = [{"face_index": index} for index in range(len(model.faces))]
    pass_errors: dict[str, float] = {}
    pass_seconds: dict[str, float] = {}
    pass_totals: dict[str, np.ndarray] = {}
    pass_previous_totals: dict[str, np.ndarray] = {}
    pass_previous_previous_totals: dict[str, np.ndarray] = {}
    pass_statistics: dict[str, _Statistics] = {}
    passes = [("core", 2)]
    if config.brep_compute_fairness:
        passes.append(("fairness", 3))
    for mode, continuity_order in passes:
        statistics = _Statistics()
        pass_statistics[mode] = statistics
        pass_started = perf_counter()
        pass_total = np.zeros(_N_QUANTITIES, dtype=float)
        pass_previous_total = np.zeros(_N_QUANTITIES, dtype=float)
        pass_previous_previous_total = np.zeros(_N_QUANTITIES, dtype=float)
        maximum_error = 0.0
        for face_index, face in enumerate(model.faces):
            surface = BRepAdaptor_Surface(face, True)
            classifier = BRepTopAdaptor_FClass2d(
                face,
                float(config.brep_classifier_tolerance),
            )
            try:
                u_intervals, v_intervals = face_continuity_intervals(
                    face,
                    order=continuity_order,
                )
            except Exception as error:
                statistics.face_failures.append(
                    {
                        "face_index": face_index,
                        "stage": f"{mode}_continuity",
                        "error": str(error),
                    }
                )
                u_intervals = np.array(
                    [surface.FirstUParameter(), surface.LastUParameter()], dtype=float
                )
                v_intervals = np.array(
                    [surface.FirstVParameter(), surface.LastVParameter()], dtype=float
                )
            face_total = np.zeros(_N_QUANTITIES, dtype=float)
            face_previous_total = np.zeros(_N_QUANTITIES, dtype=float)
            face_previous_previous_total = np.zeros(_N_QUANTITIES, dtype=float)
            face_error = 0.0
            initial_cells = 0
            base = config.brep_quadrature_base_subdivisions
            for u_start, u_stop in pairwise(u_intervals):
                for v_start, v_stop in pairwise(v_intervals):
                    u_edges = np.linspace(u_start, u_stop, base + 1)
                    v_edges = np.linspace(v_start, v_stop, base + 1)
                    for u0, u1 in pairwise(u_edges):
                        for v0, v1 in pairwise(v_edges):
                            initial_cells += 1
                            (
                                value,
                                previous_value,
                                previous_previous_value,
                                error,
                            ) = _adaptive_cell(
                                face,
                                surface,
                                classifier,
                                (float(u0), float(u1), float(v0), float(v1)),
                                mode=mode,
                                order=config.brep_quadrature_order,
                                tolerance=config.brep_quadrature_tolerance,
                                maximum_depth=config.brep_quadrature_max_depth,
                                depth=0,
                                length_ref=length_ref,
                                h_threshold=float(h_threshold),
                                k_threshold=float(k_threshold),
                                statistics=statistics,
                                face_index=face_index,
                            )
                            face_total += value
                            face_previous_total += previous_value
                            face_previous_previous_total += previous_previous_value
                            face_error = max(face_error, error)
            pass_total += face_total
            pass_previous_total += face_previous_total
            pass_previous_previous_total += face_previous_previous_total
            maximum_error = max(maximum_error, face_error)
            face_audit[face_index].update(
                {
                    f"{mode}_area": float(face_total[_AREA]),
                    f"{mode}_valid_area": float(
                        face_total[_CURVATURE_AREA if mode == "core" else _FAIRNESS_AREA]
                    ),
                    f"{mode}_u_continuity_intervals": int(len(u_intervals) - 1),
                    f"{mode}_v_continuity_intervals": int(len(v_intervals) - 1),
                    f"{mode}_initial_cells": initial_cells,
                    f"{mode}_maximum_local_relative_estimate": face_error,
                    f"{mode}_one_level_relative_change": float(
                        np.max(
                            np.abs(face_total - face_previous_total)
                            / np.maximum.reduce(
                                (
                                    np.abs(face_total),
                                    np.abs(face_previous_total),
                                    np.full(_N_QUANTITIES, config.eps),
                                )
                            )
                        )
                    ),
                }
            )
        pass_totals[mode] = pass_total
        pass_previous_totals[mode] = pass_previous_total
        pass_previous_previous_totals[mode] = pass_previous_previous_total
        pass_errors[mode] = maximum_error
        pass_seconds[mode] = perf_counter() - pass_started

    if not config.brep_compute_fairness:
        pass_totals["fairness"] = np.zeros(_N_QUANTITIES, dtype=float)
        pass_previous_totals["fairness"] = np.zeros(_N_QUANTITIES, dtype=float)
        pass_previous_previous_totals["fairness"] = np.zeros(_N_QUANTITIES, dtype=float)
        pass_statistics["fairness"] = _Statistics()
        pass_errors["fairness"] = 0.0
        pass_seconds["fairness"] = 0.0

    total = pass_totals["core"]
    total[_FAIRNESS_AREA] = pass_totals["fairness"][_FAIRNESS_AREA]
    total[_GRADIENT_H2] = pass_totals["fairness"][_GRADIENT_H2]

    total_area = float(total[_AREA])
    curvature_area = float(total[_CURVATURE_AREA])
    fairness_area = float(total[_FAIRNESS_AREA])
    independent_area = float(model.metadata["independent_surface_area"])
    area_discrepancy = (
        (total_area - independent_area) / independent_area
        if independent_area > 0.0
        else float("nan")
    )
    area_acceptance = max(10.0 * config.brep_quadrature_tolerance, 1.0e-5)
    area_converged = bool(
        np.isfinite(area_discrepancy) and abs(area_discrepancy) <= area_acceptance
    )
    class_areas = {
        "flat": float(total[_CLASS_FLAT]),
        "single": float(total[_CLASS_SINGLE]),
        "elliptic": float(total[_CLASS_ELLIPTIC]),
        "saddle": float(total[_CLASS_SADDLE]),
    }
    values = {
        "surface_area": total_area,
        "curvature_valid_area": curvature_area,
        "curvature_energy": float(total[_H2] / max(curvature_area, config.eps)),
        "developability_deviation": float(total[_K_ABS] / max(curvature_area, config.eps)),
        "developability_deviation_positive": float(
            total[_K_POSITIVE] / max(curvature_area, config.eps)
        ),
        "developability_deviation_negative": float(
            total[_K_NEGATIVE] / max(curvature_area, config.eps)
        ),
        "developability_area_ratio": float(
            total[_K_THRESHOLD_AREA] / max(curvature_area, config.eps)
        ),
        "fairness_valid_area": fairness_area,
        "curvature_fairness": float(
            length_ref**4 * total[_GRADIENT_H2] / max(fairness_area, config.eps)
        ),
        "h_threshold": float(h_threshold),
        "k_threshold": float(k_threshold),
    }

    def normalized_values(raw: np.ndarray) -> dict[str, float]:
        core_area = max(float(raw[_CURVATURE_AREA]), config.eps)
        fair_area = max(float(raw[_FAIRNESS_AREA]), config.eps)
        return {
            "surface_area": float(raw[_AREA]),
            "curvature_energy": float(raw[_H2] / core_area),
            "developability_deviation": float(raw[_K_ABS] / core_area),
            "developability_deviation_positive": float(raw[_K_POSITIVE] / core_area),
            "developability_deviation_negative": float(raw[_K_NEGATIVE] / core_area),
            "developability_area_ratio": float(raw[_K_THRESHOLD_AREA] / core_area),
            "curvature_fairness": float(length_ref**4 * raw[_GRADIENT_H2] / fair_area),
        }

    previous = pass_previous_totals["core"].copy()
    previous[_FAIRNESS_AREA] = pass_previous_totals["fairness"][_FAIRNESS_AREA]
    previous[_GRADIENT_H2] = pass_previous_totals["fairness"][_GRADIENT_H2]
    one_level_shallower_values = normalized_values(previous)
    previous_previous = pass_previous_previous_totals["core"].copy()
    previous_previous[_FAIRNESS_AREA] = pass_previous_previous_totals["fairness"][_FAIRNESS_AREA]
    previous_previous[_GRADIENT_H2] = pass_previous_previous_totals["fairness"][_GRADIENT_H2]
    two_levels_shallower_values = normalized_values(previous_previous)

    def relative_change(name: str) -> float:
        current = values[name]
        prior = one_level_shallower_values[name]
        return float(abs(current - prior) / max(abs(current), abs(prior), config.eps))

    one_level_relative_change = {
        name: relative_change(name)
        for name in (
            "surface_area",
            "curvature_energy",
            "developability_deviation",
            "developability_deviation_positive",
            "developability_deviation_negative",
            "developability_area_ratio",
            "curvature_fairness",
        )
    }
    prior_level_relative_change = {
        name: float(
            abs(one_level_shallower_values[name] - two_levels_shallower_values[name])
            / max(
                abs(one_level_shallower_values[name]),
                abs(two_levels_shallower_values[name]),
                config.eps,
            )
        )
        for name in one_level_relative_change
    }
    all_statistics = list(pass_statistics.values())
    combined_unconverged = sum(item.unconverged_cells_at_maximum_depth for item in all_statistics)

    def finite_or_none(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    regularity_evidence = {
        mode: {
            "minimum_jacobian": finite_or_none(stats.minimum_jacobian),
            "maximum_jacobian": finite_or_none(stats.maximum_jacobian),
            "minimum_metric_determinant": finite_or_none(stats.minimum_metric_determinant),
            "maximum_first_fundamental_condition": finite_or_none(stats.maximum_metric_condition),
            "maximum_abs_mean_curvature": stats.maximum_abs_mean_curvature,
            "maximum_abs_gaussian_curvature": stats.maximum_abs_gaussian_curvature,
            "maximum_gradient_mean_curvature": stats.maximum_gradient_mean,
            "maximum_depth_integrals": stats.maximum_depth_integrals.tolist(),
            "unconverged_integrals": stats.unconverged_integrals.tolist(),
            "dominant_cells": [
                record
                for _, _, record in sorted(
                    stats.top_cells,
                    key=lambda item: item[0],
                    reverse=True,
                )
            ],
        }
        for mode, stats in pass_statistics.items()
    }
    concentration = {
        "curvature_energy_unconverged_cell_fraction": float(
            pass_statistics["core"].unconverged_integrals[_H2] / max(abs(total[_H2]), config.eps)
        ),
        "developability_unconverged_cell_fraction": float(
            pass_statistics["core"].unconverged_integrals[_K_ABS]
            / max(abs(total[_K_ABS]), config.eps)
        ),
        "fairness_unconverged_cell_fraction": float(
            pass_statistics["fairness"].unconverged_integrals[_GRADIENT_H2]
            / max(abs(total[_GRADIENT_H2]), config.eps)
        )
        if config.brep_compute_fairness
        else 0.0,
    }
    metadata = {
        "algorithm": "adaptive_tensor_gauss_legendre_over_trimmed_uv",
        "quadrature_order": int(config.brep_quadrature_order),
        "embedded_comparison_order": int(max(2, config.brep_quadrature_order - 2)),
        "relative_tolerance": float(config.brep_quadrature_tolerance),
        "maximum_depth": int(config.brep_quadrature_max_depth),
        "base_subdivisions_per_continuity_cell": int(config.brep_quadrature_base_subdivisions),
        "classifier_tolerance": float(config.brep_classifier_tolerance),
        "fairness_requested": bool(config.brep_compute_fairness),
        "evaluated_points": sum(item.evaluated_points for item in all_statistics),
        "outside_points": sum(item.outside_points for item in all_statistics),
        "singular_area_points": sum(item.singular_area_points for item in all_statistics),
        "invalid_curvature_points": pass_statistics["core"].invalid_curvature_points,
        "invalid_fairness_points": pass_statistics["fairness"].invalid_fairness_points,
        "lprop_mismatch_points": sum(item.lprop_mismatch_points for item in all_statistics),
        "opencascade_property_crosscheck_policy": (
            "analytical_test_points_only_not_repeated_at_every_quadrature_point"
        ),
        "accepted_cells": sum(item.accepted_cells for item in all_statistics),
        "subdivided_cells": sum(item.subdivided_cells for item in all_statistics),
        "unconverged_cells_at_maximum_depth": combined_unconverged,
        "unconverged_cells_at_maximum_depth_by_pass": {
            mode: stats.unconverged_cells_at_maximum_depth
            for mode, stats in pass_statistics.items()
        },
        "maximum_depth_reached": max(item.maximum_depth_reached for item in all_statistics),
        "maximum_local_relative_estimate": pass_errors,
        "one_level_shallower_values": one_level_shallower_values,
        "two_levels_shallower_values": two_levels_shallower_values,
        "finite_depth_candidate_values": {
            name: values[name]
            for name in (
                "surface_area",
                "curvature_energy",
                "developability_deviation",
                "developability_deviation_positive",
                "developability_deviation_negative",
                "developability_area_ratio",
                "curvature_fairness",
            )
        },
        "one_level_relative_change": one_level_relative_change,
        "prior_level_relative_change": prior_level_relative_change,
        "bounded_refinement_diagnostic": (
            "same adaptive tree with terminal refinement level replaced by its "
            "embedded parent estimate"
        ),
        "regularity_evidence": regularity_evidence,
        "contribution_concentration": concentration,
        "timing_seconds": pass_seconds,
        "independent_brep_area": independent_area,
        "independent_brep_area_estimated_error": model.metadata[
            "independent_surface_area_estimated_error"
        ],
        "area_relative_discrepancy": area_discrepancy,
        "area_relative_acceptance": area_acceptance,
        "area_converged_against_independent_brep_measure": area_converged,
        "convergence_status": (
            "converged" if area_converged and combined_unconverged == 0 else "caution"
        ),
        "curvature_valid_area_fraction": curvature_area / max(total_area, config.eps),
        "fairness_valid_area_fraction": fairness_area / max(total_area, config.eps),
        "face_audit": face_audit,
        "face_failures": [failure for item in all_statistics for failure in item.face_failures],
        "canonical_metrics_depend_on_triangulation": False,
    }
    return BRepIntegralResult(values=values, class_areas=class_areas, metadata=metadata)
