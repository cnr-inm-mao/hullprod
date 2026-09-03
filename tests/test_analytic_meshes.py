import numpy as np
import pytest
import trimesh

from hullprod import metrics as metrics_module
from hullprod.experimental_curvature import mean_curvature_legacy_v012
from hullprod.metrics import compute_metrics
from hullprod.types import ProducibilityConfig


def make_open_cylinder(radius=1.0, height=5.0, sections=64, axial_points=9):
    """Create an open cylindrical side surface aligned with the x axis."""
    theta = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    x_values = np.linspace(-0.5 * height, 0.5 * height, axial_points)

    vertices = []
    for x in x_values:
        for t in theta:
            vertices.append([x, radius * np.cos(t), radius * np.sin(t)])

    faces = []
    for axial in range(axial_points - 1):
        for i in range(sections):
            j = (i + 1) % sections
            a = axial * sections + i
            b = axial * sections + j
            c = (axial + 1) * sections + j
            d = (axial + 1) * sections + i
            faces.append([a, b, c])
            faces.append([a, c, d])

    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=int),
        process=False,
    )


def make_saddle_patch(nx=31, ny=31, size=2.0, amplitude=0.25):
    """Create an open hyperbolic-paraboloid patch z = a (x^2 - y^2)."""
    x = np.linspace(-0.5 * size, 0.5 * size, nx)
    y = np.linspace(-0.5 * size, 0.5 * size, ny)

    vertices = []
    for xi in x:
        for yi in y:
            zi = amplitude * (xi**2 - yi**2)
            vertices.append([xi, yi, zi])

    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b = (i + 1) * ny + j
            c = (i + 1) * ny + (j + 1)
            d = i * ny + (j + 1)
            faces.append([a, b, c])
            faces.append([a, c, d])

    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=int),
        process=False,
    )


def test_sphere_has_positive_signed_developability():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

    res = compute_metrics(
        mesh,
        ProducibilityConfig(length_ref=2.0, n_stations=10),
    )

    positive = res.metrics["developability_deviation_positive"]
    negative = res.metrics["developability_deviation_negative"]

    assert res.metrics["developability_deviation"] > 0.0
    assert positive > 0.0
    assert negative < 0.1 * positive
    assert res.curvature_classes["area_fraction_elliptic_double_curvature"] > 0.5


def test_open_cylinder_is_mostly_single_curvature():
    mesh = make_open_cylinder(radius=1.0, height=5.0, sections=64)

    res = compute_metrics(
        mesh,
        ProducibilityConfig(
            length_ref=5.0,
            n_stations=10,
            k_threshold_factor=1e-2,
        ),
    )

    assert res.metrics["developability_deviation"] < 1e-6
    assert res.metrics["developability_deviation_positive"] < 1e-6
    assert res.metrics["developability_deviation_negative"] < 1e-6
    assert res.metrics["developability_area_ratio"] < 1e-6
    assert res.curvature_classes["area_fraction_cylindrical_single_curvature"] > 0.9


def test_signed_developability_components_are_consistent_on_saddle_patch():
    mesh = make_saddle_patch(nx=35, ny=35, size=2.0, amplitude=0.25)

    res = compute_metrics(
        mesh,
        ProducibilityConfig(
            length_ref=2.0,
            n_stations=10,
            k_threshold_factor=1e-3,
        ),
    )

    positive = res.metrics["developability_deviation_positive"]
    negative = res.metrics["developability_deviation_negative"]

    assert positive >= 0.0
    assert negative >= 0.0
    assert positive + negative > 0.0
    assert res.metrics["developability_deviation"] == pytest.approx(
        positive + negative,
    )
    assert (
        res.curvature_classes["area_fraction_elliptic_double_curvature"]
        + res.curvature_classes["area_fraction_saddle_reverse_double_curvature"]
        > 0.0
    )


def test_metrics_are_scale_consistent_for_scaled_sphere():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    scaled = mesh.copy()
    scaled.apply_scale(3.0)

    cfg = ProducibilityConfig(length_ref=2.0, n_stations=10)
    cfg_scaled = ProducibilityConfig(length_ref=6.0, n_stations=10)

    res = compute_metrics(mesh, cfg)
    res_scaled = compute_metrics(scaled, cfg_scaled)

    for key in [
        "curvature_energy",
        "curvature_fairness",
        "developability_deviation",
        "section_waviness",
    ]:
        assert np.isfinite(res.metrics[key])
        assert np.isfinite(res_scaled.metrics[key])
        assert res_scaled.metrics[key] == pytest.approx(
            res.metrics[key],
            rel=1e-5,
            abs=1e-8,
        )


def test_edge_length_metadata_is_reported():
    mesh = make_open_cylinder(radius=1.0, height=5.0, sections=32)

    res = compute_metrics(
        mesh,
        ProducibilityConfig(length_ref=5.0, n_stations=10),
    )

    for key in [
        "edge_length_mean",
        "edge_length_median",
        "edge_length_min",
        "edge_length_p01",
        "edge_length_p05",
        "edge_length_p95",
        "edge_length_p99",
        "edge_length_max",
        "edge_length_min_to_median",
        "edge_length_max_to_median",
    ]:
        assert key in res.metadata
        assert np.isfinite(res.metadata[key])
        assert res.metadata[key] > 0.0


def test_robust_metrics_are_optional_and_reported():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

    base = compute_metrics(
        mesh,
        ProducibilityConfig(length_ref=2.0, n_stations=10),
    )
    assert "curvature_energy_robust" not in base.metrics
    assert base.metadata["robust_metrics_enabled"] is False

    robust = compute_metrics(
        mesh,
        ProducibilityConfig(
            length_ref=2.0,
            n_stations=10,
            robust_metrics=True,
        ),
    )

    for key in [
        "curvature_energy_robust",
        "curvature_fairness_robust",
        "developability_deviation_robust",
        "developability_deviation_positive_robust",
        "developability_deviation_negative_robust",
    ]:
        assert key in robust.metrics
        assert np.isfinite(robust.metrics[key])
        assert robust.metrics[key] >= 0.0

    assert robust.metadata["robust_metrics_enabled"] is True
    assert robust.metadata["robust_vertex_area_percentile"] == 5.0
    assert robust.metadata["robust_edge_length_percentile"] == 5.0
    assert robust.metadata["robust_h_clip_percentile"] == 95.0
    assert 0.0 < robust.metadata["robust_vertex_fraction"] <= 1.0
    assert 0.0 < robust.metadata["robust_area_fraction"] <= 1.0
    assert "robust_vertex_mask" in robust.local_fields


def test_release_curvature_backend_is_distinct_from_legacy_v012():
    """The paper backend must not silently restore the obsolete cotangent field."""
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    config = ProducibilityConfig(
        length_ref=2.0,
        n_stations=10,
        robust_metrics=True,
    )

    result = compute_metrics(mesh, config)
    legacy = mean_curvature_legacy_v012(mesh)

    assert result.metadata["estimator_metadata"]["curvature_method"].startswith("rusinkiewicz")
    assert np.median(result.local_fields["H"]) == pytest.approx(1.0, rel=1e-12)
    assert np.median(legacy) == pytest.approx(0.5, rel=0.03)


def test_curvature_classes_can_move_from_flat_to_cylindrical_when_h_doubles():
    areas = np.ones(2)
    gaussian = np.zeros(2)
    legacy = metrics_module._curvature_classes(
        np.array([0.4, 0.75]),
        gaussian,
        areas,
        h_thr=1.0,
        k_thr=1.0,
    )
    corrected = metrics_module._curvature_classes(
        np.array([0.8, 1.5]),
        gaussian,
        areas,
        h_thr=1.0,
        k_thr=1.0,
    )

    assert legacy["area_fraction_flat"] == 1.0
    assert corrected["area_fraction_flat"] == 0.5
    assert corrected["area_fraction_cylindrical_single_curvature"] == 0.5
