"""CAD-native transverse sections and section-waviness metrics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .brep_geometry import BRepModel
from .sections import canonicalize_section_side
from .types import ProducibilityConfig


@dataclass
class _SectionEdge:
    edge: Any
    adaptor: Any
    first: float
    last: float
    length: float
    start: np.ndarray
    end: np.ndarray
    node_start: int = -1
    node_end: int = -1


def _point_array(point: Any) -> np.ndarray:
    return np.array([point.X(), point.Y(), point.Z()], dtype=float)


def _section_edges(model: BRepModel, x_value: float) -> list[_SectionEdge]:
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCP.GCPnts import GCPnts_AbscissaPoint
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    plane = gp_Pln(gp_Pnt(float(x_value), 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0))
    operation = BRepAlgoAPI_Section(model.shape, plane, False)
    operation.ComputePCurveOn1(True)
    operation.Approximation(True)
    operation.Build()
    if not operation.IsDone():
        return []
    explorer = TopExp_Explorer(operation.Shape(), TopAbs_EDGE)
    result = []
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        adaptor = BRepAdaptor_Curve(edge)
        first = float(adaptor.FirstParameter())
        last = float(adaptor.LastParameter())
        try:
            length = float(GCPnts_AbscissaPoint.Length_s(adaptor, first, last, 1.0e-9))
            if not np.isfinite(length) or length <= 1.0e-12:
                explorer.Next()
                continue
            result.append(
                _SectionEdge(
                    edge=edge,
                    adaptor=adaptor,
                    first=first,
                    last=last,
                    length=length,
                    start=_point_array(adaptor.Value(first)),
                    end=_point_array(adaptor.Value(last)),
                )
            )
        except Exception:
            pass
        explorer.Next()
    return result


def _cluster_endpoints(edges: list[_SectionEdge], tolerance: float) -> list[np.ndarray]:
    nodes: list[np.ndarray] = []
    for edge in edges:
        indices = []
        for point in (edge.start, edge.end):
            match = next(
                (
                    index
                    for index, node in enumerate(nodes)
                    if np.linalg.norm(point - node) <= tolerance
                ),
                None,
            )
            if match is None:
                nodes.append(point.copy())
                match = len(nodes) - 1
            indices.append(match)
        edge.node_start, edge.node_end = indices
    return nodes


def _ordered_components(
    edges: list[_SectionEdge],
    *,
    tolerance: float,
) -> list[tuple[list[tuple[_SectionEdge, bool]], bool]]:
    nodes = _cluster_endpoints(edges, tolerance)
    adjacency: dict[int, list[int]] = {index: [] for index in range(len(nodes))}
    for edge_index, edge in enumerate(edges):
        adjacency[edge.node_start].append(edge_index)
        adjacency[edge.node_end].append(edge_index)
    remaining = set(range(len(edges)))
    components = []
    while remaining:
        seed = min(remaining)
        reachable_edges = {seed}
        frontier = [seed]
        component_nodes = set()
        while frontier:
            edge_index = frontier.pop()
            edge = edges[edge_index]
            component_nodes.update((edge.node_start, edge.node_end))
            for node in (edge.node_start, edge.node_end):
                for neighbor in adjacency[node]:
                    if neighbor not in reachable_edges:
                        reachable_edges.add(neighbor)
                        frontier.append(neighbor)
        endpoints = sorted(
            (node for node in component_nodes if len(adjacency[node]) == 1),
            key=lambda index: tuple(nodes[index]),
        )
        closed = not endpoints
        current = (
            endpoints[0]
            if endpoints
            else min(component_nodes, key=lambda index: tuple(nodes[index]))
        )
        ordered: list[tuple[_SectionEdge, bool]] = []
        unused = set(reachable_edges)
        while unused:
            candidates = sorted(set(adjacency[current]) & unused)
            if not candidates:
                break
            edge_index = candidates[0]
            unused.remove(edge_index)
            edge = edges[edge_index]
            forward = edge.node_start == current
            ordered.append((edge, forward))
            current = edge.node_end if forward else edge.node_start
        remaining -= reachable_edges
        if ordered:
            components.append((ordered, closed))
    return components


def _parameter_at_distance(edge: _SectionEdge, distance: float, forward: bool) -> float:
    from OCP.GCPnts import GCPnts_AbscissaPoint

    distance = min(max(float(distance), 0.0), edge.length)
    start = edge.first if forward else edge.last
    signed_distance = distance if forward else -distance
    if distance <= 1.0e-14 * max(edge.length, 1.0):
        return start
    if edge.length - distance <= 1.0e-14 * max(edge.length, 1.0):
        return edge.last if forward else edge.first
    solver = GCPnts_AbscissaPoint(1.0e-9, edge.adaptor, signed_distance, start)
    if not solver.IsDone():
        raise RuntimeError("OpenCascade arc-length inversion failed")
    return float(solver.Parameter())


def _sample_component(
    component: list[tuple[_SectionEdge, bool]],
    *,
    closed: bool,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    from OCP.gp import gp_Pnt, gp_Vec

    lengths = np.array([edge.length for edge, _ in component], dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    targets = np.linspace(0.0, total, samples, endpoint=not closed)
    points = []
    curvature = []
    plane_normal = np.array([1.0, 0.0, 0.0])
    for target in targets:
        edge_index = min(
            int(np.searchsorted(cumulative, target, side="right") - 1), len(component) - 1
        )
        edge, forward = component[edge_index]
        local_distance = float(target - cumulative[edge_index])
        parameter = _parameter_at_distance(edge, local_distance, forward)
        point = gp_Pnt()
        first = gp_Vec()
        second = gp_Vec()
        edge.adaptor.D2(parameter, point, first, second)
        first_array = _point_array(first)
        second_array = _point_array(second)
        if not forward:
            first_array *= -1.0
        speed = float(np.linalg.norm(first_array))
        if speed <= 1.0e-14:
            raise RuntimeError("singular CAD section curve derivative")
        signed_curvature = float(
            np.dot(np.cross(first_array, second_array), plane_normal) / speed**3
        )
        points.append(_point_array(point))
        curvature.append(signed_curvature)
    return np.asarray(points), np.asarray(curvature)


def _waviness_from_curvature(
    curvature: np.ndarray,
    *,
    closed: bool,
    fft_cutoff_fraction: float,
) -> tuple[float, float, int]:
    values = np.asarray(curvature, dtype=float)
    if len(values) < 5 or not np.all(np.isfinite(values)):
        return float("nan"), float("nan"), 0
    # Match the mesh realization: open curves omit the two endpoint stencil
    # values rather than treating one-sided endpoint error as a continuum term.
    working = values if closed else values[2:-2]
    if len(working) < 5:
        return float("nan"), float("nan"), 0
    coordinate = np.linspace(0.0, 1.0, len(working), endpoint=not closed)
    if closed:
        step = 1.0 / len(working)
        derivative = (np.roll(working, -1) - working) / step
        numerator = float(np.mean(derivative**2))
        denominator = float(np.mean(working**2)) + 1.0e-30
    else:
        derivative = np.gradient(working, coordinate, edge_order=2)
        numerator = float(np.trapezoid(derivative**2, coordinate))
        denominator = float(np.trapezoid(working**2, coordinate)) + 1.0e-30
    physical = numerator / denominator
    demeaned = working - float(np.mean(working))
    power = np.abs(np.fft.rfft(demeaned)) ** 2
    non_dc_bins = max(len(power) - 1, 0)
    cutoff = max(1, int(np.ceil(fft_cutoff_fraction * non_dc_bins)))
    relative_variation = float(np.std(working)) / max(
        float(np.mean(np.abs(working))), 1.0e-30
    )
    if relative_variation <= 1.0e-8:
        fft = 0.0
    elif non_dc_bins > 2 and float(np.sum(power[1:])) > 0.0:
        fft = float(np.sum(power[cutoff + 1 :]) / np.sum(power[1:]))
    else:
        fft = float("nan")
    return physical, fft, cutoff


def _resample_paired_by_arclength(
    points: np.ndarray,
    values: np.ndarray,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample point/value pairs on the retained open section fragment."""
    distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
    coordinate = np.concatenate(([0.0], np.cumsum(distance)))
    keep = np.concatenate(([True], np.diff(coordinate) > 1.0e-14))
    points = points[keep]
    values = values[keep]
    coordinate = coordinate[keep]
    if len(points) < 3 or coordinate[-1] <= 0.0:
        return points, values
    target = np.linspace(0.0, coordinate[-1], samples)
    sampled_points = np.column_stack(
        [np.interp(target, coordinate, points[:, axis]) for axis in range(points.shape[1])]
    )
    sampled_values = np.interp(target, coordinate, values)
    return sampled_points, sampled_values


def brep_section_waviness(
    model: BRepModel,
    config: ProducibilityConfig,
    *,
    length_ref: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Intersect the exact BRep with x-planes and evaluate exact curve D2."""
    started = perf_counter()
    bounds = np.asarray(model.metadata["bounds"], dtype=float)
    x_min, x_max = bounds[:, 0]
    span = float(x_max - x_min)
    base_metadata: dict[str, Any] = {
        "requested_sections": int(config.n_stations),
        "resampling_points": int(config.section_resample_points),
        "station_margin_fraction": float(config.station_margin),
        "station_policy": "uniform_x_between_brep_bounds_margin_limits",
        "fft_cutoff_fraction": float(config.fft_cutoff_fraction),
        "fft_cutoff_policy": "fraction_of_nonnegative_non_dc_frequency_bins",
        "section_component_policy": "longest_connected_exact_brep_curve",
        "curve_sampling": "OpenCascade_uniform_physical_abscissa_D2_curvature",
        "section_side_policy_requested": config.section_side_policy,
        "section_side_policy_actual": config.section_side_policy,
        "section_centerline_tolerance_requested": config.section_centerline_tolerance,
        "representation_backend": "brep_exact_plane_intersection_D2",
        "stations": [],
    }
    if span <= 0.0:
        return {
            "section_waviness": float("nan"),
            "section_waviness_fft": float("nan"),
            "valid_sections": 0.0,
        }, {**base_metadata, "invalid_sections": int(config.n_stations)}
    physical_values = []
    fft_values = []
    multicomponent = 0
    open_count = 0
    sections_with_edge_joins = 0
    total_edge_joins = 0
    tolerance = max(1.0e-8 * span, 1.0e-9)
    stations = np.linspace(
        x_min + config.station_margin * span,
        x_max - config.station_margin * span,
        config.n_stations,
    )
    for x_value in stations:
        record: dict[str, Any] = {
            "station_coordinate": float(x_value),
            "x": float(x_value),
            "x_over_L_ref": float(x_value / length_ref),
            "x_over_lref": float(x_value / length_ref),
            "longitudinal_fraction": float((x_value - x_min) / span),
            "valid": False,
            "physical_waviness": None,
            "fft_waviness": None,
            "component_count": 0,
            "source_component_count": 0,
            "closed": None,
            "open": None,
        }
        try:
            edges = _section_edges(model, float(x_value))
            components = _ordered_components(edges, tolerance=tolerance)
            record["edge_count"] = len(edges)
            record["component_count"] = len(components)
            record["source_component_count"] = len(components)
            if not components:
                record["reason"] = "no_brep_intersection_curve"
                base_metadata["stations"].append(record)
                continue
            multicomponent += int(len(components) > 1)
            component, closed = max(
                components,
                key=lambda value: sum(edge.length for edge, _ in value[0]),
            )
            component_length = float(sum(edge.length for edge, _ in component))
            edge_join_count = max(len(component) - int(closed), 0)
            sections_with_edge_joins += int(edge_join_count > 0)
            total_edge_joins += edge_join_count
            points_xyz, curvature = _sample_component(
                component,
                closed=closed,
                samples=config.section_resample_points,
            )
            canonical = canonicalize_section_side(
                points_xyz[:, 1:3],
                policy=config.section_side_policy,
                centerline_tolerance=config.section_centerline_tolerance,
                values=curvature,
                closed=closed,
            )
            dense_samples = config.section_resample_points
            if canonical.clipping_occurred:
                dense_samples = max(4 * config.section_resample_points, 1024)
                points_xyz, curvature = _sample_component(
                    component,
                    closed=closed,
                    samples=dense_samples,
                )
                canonical = canonicalize_section_side(
                    points_xyz[:, 1:3],
                    policy=config.section_side_policy,
                    centerline_tolerance=config.section_centerline_tolerance,
                    values=curvature,
                    closed=closed,
                )
            points = canonical.points
            curvature = canonical.values
            if curvature is None or len(points) < 5:
                raise RuntimeError("insufficient retained canonical section points")
            if dense_samples != config.section_resample_points:
                points, curvature = _resample_paired_by_arclength(
                    points,
                    curvature,
                    config.section_resample_points,
                )
            evaluated_closed = canonical.closed_after
            open_count += int(not evaluated_closed)
            physical, fft, cutoff = _waviness_from_curvature(
                curvature,
                closed=evaluated_closed,
                fft_cutoff_fraction=config.fft_cutoff_fraction,
            )
            if not np.isfinite(physical):
                raise RuntimeError("non-finite CAD section waviness")
            record.update(
                {
                    "valid": True,
                    "closed": evaluated_closed,
                    "open": not evaluated_closed,
                    "closed_before_canonicalization": canonical.closed_before,
                    "open_before_canonicalization": not canonical.closed_before,
                    "closed_after_canonicalization": canonical.closed_after,
                    "open_after_canonicalization": not canonical.closed_after,
                    "section_side_policy_requested": canonical.requested_policy,
                    "section_side_policy_actual": canonical.actual_policy,
                    "section_side_clipping_occurred": canonical.clipping_occurred,
                    "centerline_tolerance": canonical.centerline_tolerance,
                    "retained_component": canonical.retained_component,
                    "retained_component_length": canonical.retained_component_length,
                    "representation_backend": "brep_exact_plane_intersection_D2",
                    "physical_waviness": physical,
                    "fft_waviness": fft,
                    "fft_cutoff_index": cutoff,
                    "sampled_points": len(points),
                    "component_length": component_length,
                    "source_edge_join_count": edge_join_count,
                    "join_treatment": "piecewise_exact_D2_samples_no_smoothing",
                }
            )
            physical_values.append(physical)
            if np.isfinite(fft):
                fft_values.append(fft)
        except Exception as error:
            record["reason"] = f"section_evaluation_failed: {error}"
        base_metadata["stations"].append(record)
    valid_count = len(physical_values)
    metrics = {
        "section_waviness": (float(np.mean(physical_values)) if physical_values else float("nan")),
        "section_waviness_fft": (float(np.mean(fft_values)) if fft_values else float("nan")),
        "valid_sections": float(valid_count),
        "sections_with_edge_joins": float(sections_with_edge_joins),
        "source_edge_join_count": float(total_edge_joins),
    }
    base_metadata.update(
        {
            "invalid_sections": int(config.n_stations - valid_count),
            "multicomponent_sections": multicomponent,
            "open_sections": open_count,
            "sections_with_edge_joins": sections_with_edge_joins,
            "source_edge_join_count": total_edge_joins,
            "timing_seconds": perf_counter() - started,
        }
    )
    return metrics, base_metadata
