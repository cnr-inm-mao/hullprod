from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class CurveWaviness:
    """Dimensionless waviness values and the resampling audit for one curve."""

    physical: float
    fft: float
    valid: bool
    closed: bool
    resampled_points: int
    fft_cutoff_index: int


@dataclass(frozen=True)
class CanonicalSection:
    """One section after applying an explicit representation-side policy."""

    points: np.ndarray
    values: np.ndarray | None
    requested_policy: str
    actual_policy: str
    clipping_occurred: bool
    centerline_tolerance: float
    closed_before: bool
    closed_after: bool
    retained_component: str
    retained_component_length: float


SECTION_SIDE_POLICIES = ("as_represented", "starboard_half")


def _curve_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def canonicalize_section_side(
    points: np.ndarray,
    *,
    policy: str = "as_represented",
    centerline_tolerance: float | None = None,
    values: np.ndarray | None = None,
    closed: bool | None = None,
) -> CanonicalSection:
    """Apply a declared section-side policy in canonical ``(y, z)`` space.

    ``starboard_half`` clips at ``y=0`` without mirroring or closing the
    result. Optional point-associated values are carried through centerline
    intersections, which lets the native-BRep backend preserve D2 curvature.
    """
    if policy not in SECTION_SIDE_POLICIES:
        raise ValueError(
            f"Unknown section_side_policy {policy!r}; expected one of {SECTION_SIDE_POLICIES}"
        )
    source = np.asarray(points, dtype=float)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("section points must have shape (n, 2) in canonical y-z coordinates")
    associated = None if values is None else np.asarray(values, dtype=float)
    if associated is not None and len(associated) != len(source):
        raise ValueError("point-associated section values must match the point count")
    closed_before = _is_closed(source) if closed is None else bool(closed)
    length = _curve_length(source)
    tolerance = (
        float(centerline_tolerance)
        if centerline_tolerance is not None
        else max(1.0e-10 * max(length, 1.0), 1.0e-12)
    )
    if tolerance < 0.0:
        raise ValueError("section_centerline_tolerance must be non-negative")
    if policy == "as_represented" or len(source) < 2:
        return CanonicalSection(
            source.copy(),
            None if associated is None else associated.copy(),
            policy,
            "as_represented",
            False,
            tolerance,
            closed_before,
            closed_before,
            "original_longest_component",
            length,
        )

    work_points = source.copy()
    work_values = None if associated is None else associated.copy()
    if closed_before and np.linalg.norm(work_points[0] - work_points[-1]) > tolerance:
        work_points = np.vstack((work_points, work_points[0]))
        if work_values is not None:
            work_values = np.concatenate((work_values, work_values[:1]))
    outside = np.flatnonzero(work_points[:, 0] < -tolerance)
    if closed_before and len(outside):
        start = int(outside[0])
        work_points = np.vstack((work_points[start:-1], work_points[: start + 1]))
        if work_values is not None:
            work_values = np.concatenate((work_values[start:-1], work_values[: start + 1]))

    fragments: list[tuple[list[np.ndarray], list[float] | None]] = []
    current_points: list[np.ndarray] = []
    current_values: list[float] | None = [] if work_values is not None else None

    def append(point: np.ndarray, value: float | None) -> None:
        candidate = point.copy()
        if abs(candidate[0]) <= tolerance:
            candidate[0] = 0.0
        if not current_points or np.linalg.norm(candidate - current_points[-1]) > 1.0e-14:
            current_points.append(candidate)
            if current_values is not None and value is not None:
                current_values.append(float(value))

    def finish() -> None:
        nonlocal current_points, current_values
        if len(current_points) >= 2:
            fragments.append((current_points, current_values))
        current_points = []
        current_values = [] if work_values is not None else None

    for index in range(len(work_points) - 1):
        first, second = work_points[index], work_points[index + 1]
        first_inside = first[0] >= -tolerance
        second_inside = second[0] >= -tolerance
        first_value = None if work_values is None else float(work_values[index])
        second_value = None if work_values is None else float(work_values[index + 1])
        if first_inside:
            append(first, first_value)
        if first_inside != second_inside:
            denominator = second[0] - first[0]
            fraction = 0.0 if abs(denominator) <= 1.0e-30 else -first[0] / denominator
            fraction = min(max(float(fraction), 0.0), 1.0)
            crossing = first + fraction * (second - first)
            crossing[0] = 0.0
            crossing_value = (
                None
                if work_values is None
                else first_value + fraction * (second_value - first_value)
            )
            append(crossing, crossing_value)
            if first_inside:
                finish()
        if second_inside and index == len(work_points) - 2:
            append(second, second_value)
    finish()
    clipping = bool(np.any(source[:, 0] < -tolerance))
    if not fragments:
        retained = np.empty((0, 2), dtype=float)
        retained_values = None if associated is None else np.empty(0, dtype=float)
        retained_name = "none"
    else:
        points_list, values_list = max(
            fragments,
            key=lambda item: _curve_length(np.asarray(item[0], dtype=float)),
        )
        retained = np.asarray(points_list, dtype=float)
        retained_values = None if values_list is None else np.asarray(values_list, dtype=float)
        retained_name = "longest_starboard_fragment" if clipping else "original_component"
    return CanonicalSection(
        retained,
        retained_values,
        policy,
        "starboard_half",
        clipping,
        tolerance,
        closed_before,
        False if clipping else closed_before,
        retained_name,
        _curve_length(retained),
    )


def _is_closed(points: np.ndarray) -> bool:
    if len(points) < 4:
        return False
    length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    tolerance = max(1.0e-10 * length, 1.0e-14)
    return bool(np.linalg.norm(points[0] - points[-1]) <= tolerance)


def _resample(points: np.ndarray, n: int = 256, *, closed: bool = False) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 3 or n < 5:
        return points
    if closed and np.linalg.norm(points[0] - points[-1]) > 1.0e-14:
        points = np.vstack((points, points[0]))
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    keep = np.concatenate(([True], np.diff(s) > 1.0e-14))
    points, s = points[keep], s[keep]
    if len(points) < 3 or s[-1] <= 0.0:
        return points
    target = np.linspace(0.0, s[-1], n, endpoint=not closed)
    return np.column_stack(
        [np.interp(target, s, points[:, coordinate]) for coordinate in range(points.shape[1])]
    )


def _polyline_curvature(points: np.ndarray, *, closed: bool = False) -> np.ndarray:
    """Return signed planar curvature on an approximately arc-length grid."""
    points = np.asarray(points, dtype=float)
    if len(points) < 5:
        return np.array([])
    if closed:
        ds = float(np.mean(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)))
        if ds <= 0.0:
            return np.array([])
        first = (np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)) / (2.0 * ds)
        second = (np.roll(points, -1, axis=0) - 2.0 * points + np.roll(points, 1, axis=0)) / ds**2
    else:
        ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(ds)))
        keep = np.concatenate(([True], np.diff(s) > 1.0e-14))
        points, s = points[keep], s[keep]
        if len(points) < 5 or s[-1] <= 0.0:
            return np.array([])
        first = np.column_stack([np.gradient(points[:, i], s, edge_order=2) for i in range(2)])
        second = np.column_stack([np.gradient(first[:, i], s, edge_order=2) for i in range(2)])
    denominator = np.maximum(np.sum(first * first, axis=1) ** 1.5, 1.0e-30)
    return (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / denominator


def _spline_curvature(
    points: np.ndarray,
    n: int,
    *,
    closed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a section spline and evaluate curvature from its derivatives."""
    points = np.asarray(points, dtype=float)
    if closed and np.linalg.norm(points[0] - points[-1]) <= 1.0e-14:
        points = points[:-1]
    if len(points) >= 5:
        previous = points - np.roll(points, 1, axis=0)
        following = np.roll(points, -1, axis=0) - points
        cross = np.abs(previous[:, 0] * following[:, 1] - previous[:, 1] * following[:, 0])
        scale = np.linalg.norm(previous, axis=1) * np.linalg.norm(following, axis=1)
        noncollinear = cross > 1.0e-10 * np.maximum(scale, 1.0e-30)
        if not closed:
            noncollinear[[0, -1]] = True
        points = points[noncollinear]
    if closed and np.linalg.norm(points[0] - points[-1]) > 1.0e-14:
        points = np.vstack((points, points[0]))
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    keep = np.concatenate(([True], np.diff(s) > 1.0e-14))
    points, s = points[keep], s[keep]
    if len(points) < 5 or s[-1] <= 0.0:
        return points, np.array([])
    s /= s[-1]
    if closed:
        points[-1] = points[0]
    spline = CubicSpline(s, points, axis=0, bc_type="periodic" if closed else "not-a-knot")
    u = np.linspace(0.0, 1.0, n, endpoint=not closed)
    sampled = spline(u)
    first = spline(u, 1)
    second = spline(u, 2)
    denominator = np.maximum(np.sum(first * first, axis=1) ** 1.5, 1.0e-30)
    curvature = (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / denominator
    return sampled, curvature


def curve_waviness(
    points: np.ndarray,
    *,
    resample_points: int = 256,
    fft_cutoff_fraction: float = 0.25,
) -> CurveWaviness:
    """Evaluate the manuscript section-waviness definitions on a planar curve.

    The physical value is ``L^2 int(kappa_s^2 ds) / int(kappa^2 ds)``.
    The equivalent normalized arc coordinate is used numerically. The FFT
    cutoff is a dimensionless fraction of the non-negative frequency bins.
    """
    points = np.asarray(points, dtype=float)
    closed = _is_closed(points)
    sampled, curvature = _spline_curvature(points, resample_points, closed=closed)
    if len(curvature) < 5 or not np.all(np.isfinite(curvature)):
        return CurveWaviness(np.nan, np.nan, False, closed, len(sampled), 0)

    # Periodic curves need no trimming. Open curves omit only the two endpoint
    # stencil values, whose one-sided error is not a continuum contribution.
    working = curvature if closed else curvature[2:-2]
    if len(working) < 5:
        return CurveWaviness(np.nan, np.nan, False, closed, len(sampled), 0)
    u = np.linspace(0.0, 1.0, len(working), endpoint=not closed)
    if closed:
        du = 1.0 / len(working)
        derivative = (np.roll(working, -1) - working) / du
        numerator = float(np.mean(derivative**2))
        denominator = float(np.mean(working**2)) + 1.0e-30
    else:
        derivative = np.gradient(working, u, edge_order=2)
        numerator = float(np.trapezoid(derivative**2, u))
        denominator = float(np.trapezoid(working**2, u)) + 1.0e-30
    physical = numerator / denominator

    demeaned = working - float(np.mean(working))
    power = np.abs(np.fft.rfft(demeaned)) ** 2
    non_dc_bins = max(len(power) - 1, 0)
    cutoff = max(1, int(np.ceil(fft_cutoff_fraction * non_dc_bins)))
    fft = np.nan
    relative_variation = float(np.std(working)) / max(float(np.mean(np.abs(working))), 1.0e-30)
    if relative_variation <= 1.0e-8:
        fft = 0.0
    elif non_dc_bins > 2 and float(np.sum(power[1:])) > 0.0:
        fft = float(np.sum(power[cutoff + 1 :]) / np.sum(power[1:]))
    return CurveWaviness(physical, fft, True, closed, len(sampled), cutoff)


def _entity_polylines(path: trimesh.path.Path3D) -> list[np.ndarray]:
    """Extract ordered polylines from entities, including open half-hulls."""
    curves = []
    for entity in path.entities:
        points = np.asarray(getattr(entity, "points", []), dtype=int).ravel()
        if len(points) < 3:
            continue
        points = points[np.concatenate(([True], np.diff(points) != 0))]
        if len(points) >= 3:
            curves.append(path.vertices[points])
    return curves


def section_curves_yz(mesh: trimesh.Trimesh, x: float) -> list[np.ndarray]:
    """Extract section polylines in the yz plane at ``x=constant``."""
    path = mesh.section(plane_origin=[x, 0.0, 0.0], plane_normal=[1.0, 0.0, 0.0])
    if path is None:
        return []
    try:
        polylines = list(path.discrete)
    except Exception:
        polylines = []
    if not polylines:
        polylines = _entity_polylines(path)
    return [
        np.column_stack((poly[:, 1], poly[:, 2]))
        for poly in map(np.asarray, polylines)
        if len(poly) >= 3
    ]


def section_waviness(
    mesh: trimesh.Trimesh,
    n_stations: int = 80,
    margin: float = 0.02,
    min_points: int = 25,
    fft_cutoff_fraction: float = 0.25,
    resample_points: int = 256,
    length_ref: float | None = None,
    section_side_policy: str = "as_represented",
    section_centerline_tolerance: float | None = None,
) -> dict[str, float | int | str]:
    """Compute physical and FFT waviness with explicit section validity."""
    xmin, xmax = mesh.bounds[:, 0]
    span = xmax - xmin
    station_length_ref = float(length_ref) if length_ref is not None else float(span)
    base = {
        "requested_sections": int(n_stations),
        "resampling_points": int(resample_points),
        "station_margin_fraction": float(margin),
        "station_policy": "uniform_x_between_margin_limits",
        "fft_cutoff_fraction": float(fft_cutoff_fraction),
        "fft_cutoff_policy": "fraction_of_nonnegative_non_dc_frequency_bins",
        "section_component_policy": "longest_intersection_polyline",
        "section_side_policy_requested": section_side_policy,
        "section_side_policy_actual": section_side_policy,
        "section_centerline_tolerance_requested": section_centerline_tolerance,
        "representation_backend": "mesh_polyline_section",
    }
    if span <= 0.0:
        return {
            "section_waviness": np.nan,
            "section_waviness_fft": np.nan,
            "valid_sections": 0,
            "invalid_sections": int(n_stations),
            "multicomponent_sections": 0,
            "open_sections": 0,
            "stations": [],
            **base,
        }

    physical, fft = [], []
    multicomponent = open_sections = 0
    station_records = []
    xs = np.linspace(xmin + margin * span, xmax - margin * span, n_stations)
    for x in xs:
        curves = section_curves_yz(mesh, float(x))
        record = {
            "station_coordinate": float(x),
            "x": float(x),
            "x_over_L_ref": float(x / station_length_ref),
            "x_over_lref": float(x / station_length_ref),
            "longitudinal_fraction": float((x - xmin) / span),
            "physical_waviness": None,
            "fft_waviness": None,
            "valid": False,
            "component_count": len(curves),
            "source_component_count": len(curves),
            "closed": None,
            "open": None,
        }
        if not curves:
            record["reason"] = "no_mesh_intersection_curve"
            station_records.append(record)
            continue
        multicomponent += int(len(curves) > 1)
        curve = max(curves, key=lambda c: np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
        canonical = canonicalize_section_side(
            curve,
            policy=section_side_policy,
            centerline_tolerance=section_centerline_tolerance,
        )
        record.update(
            {
                "section_side_policy_requested": canonical.requested_policy,
                "section_side_policy_actual": canonical.actual_policy,
                "section_side_clipping_occurred": canonical.clipping_occurred,
                "centerline_tolerance": canonical.centerline_tolerance,
                "closed_before_canonicalization": canonical.closed_before,
                "open_before_canonicalization": not canonical.closed_before,
                "closed_after_canonicalization": canonical.closed_after,
                "open_after_canonicalization": not canonical.closed_after,
                "retained_component": canonical.retained_component,
                "retained_component_length": canonical.retained_component_length,
                "representation_backend": "mesh_polyline_section",
            }
        )
        curve = canonical.points
        if len(curve) < min_points:
            record["reason"] = "insufficient_section_points"
            station_records.append(record)
            continue
        value = curve_waviness(
            curve,
            resample_points=resample_points,
            fft_cutoff_fraction=fft_cutoff_fraction,
        )
        if not value.valid:
            record["reason"] = "section_waviness_evaluation_invalid"
            station_records.append(record)
            continue
        open_sections += int(not value.closed)
        physical.append(value.physical)
        if np.isfinite(value.fft):
            fft.append(value.fft)
        record.update(
            {
                "physical_waviness": float(value.physical),
                "fft_waviness": float(value.fft) if np.isfinite(value.fft) else None,
                "valid": True,
                "closed": bool(value.closed),
                "open": bool(not value.closed),
                "sampled_points": int(value.resampled_points),
                "fft_cutoff_index": int(value.fft_cutoff_index),
            }
        )
        station_records.append(record)
    return {
        "section_waviness": float(np.mean(physical)) if physical else np.nan,
        "section_waviness_fft": float(np.mean(fft)) if fft else np.nan,
        "valid_sections": len(physical),
        "invalid_sections": int(n_stations - len(physical)),
        "multicomponent_sections": multicomponent,
        "open_sections": open_sections,
        "stations": station_records,
        **base,
    }
