"""Scientific-contract tests for the HullProd 1.0 implementation."""

from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh

from hullprod import assess_hull
from hullprod import sections as sections_module
from hullprod.metrics import compute_metrics
from hullprod.plotting import plot_field_histogram
from hullprod.report import write_outputs
from hullprod.schema import SCHEMA_VERSION
from hullprod.sections import curve_waviness
from hullprod.types import ProducibilityConfig


def open_cylinder(radius: float = 1.0, length: float = 5.0, nt: int = 48, nx: int = 9):
    theta = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
    x = np.linspace(-0.5 * length, 0.5 * length, nx)
    vertices = np.array([[xi, radius * np.cos(t), radius * np.sin(t)] for xi in x for t in theta])
    faces = []
    for station in range(nx - 1):
        for i in range(nt):
            j = (i + 1) % nt
            a, b = station * nt + i, station * nt + j
            c, d = (station + 1) * nt + j, (station + 1) * nt + i
            faces.extend(((a, b, c), (a, c, d)))
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def saddle_patch(n: int = 21):
    axis = np.linspace(-1.0, 1.0, n)
    vertices = np.array([[x, y, 0.25 * (x**2 - y**2)] for x in axis for y in axis])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = (i + 1) * n + j
            c = (i + 1) * n + j + 1
            d = i * n + j + 1
            faces.extend(((a, b, c), (a, c, d)))
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def plane_patch(n: int = 9):
    axis = np.linspace(0.0, 1.0, n)
    vertices = np.array([[x, y, 0.0] for x in axis for y in axis])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = (i + 1) * n + j
            c = (i + 1) * n + j + 1
            d = i * n + j + 1
            faces.extend(((a, b, c), (a, c, d)))
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def config(length: float) -> ProducibilityConfig:
    return ProducibilityConfig(
        length_ref=length,
        n_stations=3,
        min_section_points=5,
        section_resample_points=64,
    )


def test_constant_local_field_histogram_is_renderable(tmp_path) -> None:
    output = tmp_path / "constant.png"
    plot_field_histogram(
        np.full(32, 1.0e-30),
        output,
        "constant analytical field",
        transform="raw",
    )
    assert output.is_file()

    nearly_constant = tmp_path / "nearly_constant.png"
    values = 1.0 + np.linspace(0.0, 8.0 * np.finfo(float).eps, 64)
    plot_field_histogram(values, nearly_constant, "nearly constant field")
    assert nearly_constant.is_file()


def test_release_backend_sphere_h_k_energy_and_class() -> None:
    result = compute_metrics(trimesh.creation.icosphere(subdivisions=2, radius=1.0), config(2.0))
    np.testing.assert_allclose(result.local_fields["H"], 1.0, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(result.local_fields["K"], 1.0, rtol=3e-12, atol=3e-12)
    assert result.metrics["curvature_energy"] == pytest.approx(4.0, rel=2e-12)
    assert result.metrics["developability_deviation"] == pytest.approx(4.0, rel=3e-12)
    assert result.curvature_classes["area_fraction_elliptic_double_curvature"] == 1.0


def test_release_backend_plane_h_k_and_flat_class() -> None:
    result = compute_metrics(plane_patch(), config(1.0))
    valid = result.local_fields["K_valid"].astype(bool)
    assert np.max(np.abs(result.local_fields["H"])) < 1e-12
    assert np.max(np.abs(result.local_fields["K"][valid])) < 1e-12
    assert result.curvature_classes["area_fraction_flat"] == 1.0
    assert result.metadata["metric_validity"]["developability"]["boundary_values"] == (
        "invalid_not_zero_filled"
    )


def test_open_cylinder_has_valid_interior_k_and_single_curvature_class() -> None:
    mesh = open_cylinder()
    result = compute_metrics(mesh, config(5.0))
    valid = result.local_fields["K_valid"].astype(bool)
    boundary_count = 2 * 48
    assert np.sum(~valid) == boundary_count
    assert np.all(np.isnan(result.local_fields["K"][~valid]))
    assert result.metadata["gaussian_boundary_vertex_count"] == boundary_count
    assert result.metadata["developability_valid_area_fraction"] == pytest.approx(0.875)
    assert result.metrics["developability_deviation"] < 1e-12
    assert result.curvature_classes["area_fraction_cylindrical_single_curvature"] == 1.0


def test_saddle_class_uses_negative_rusinkiewicz_gaussian_curvature() -> None:
    result = compute_metrics(saddle_patch(), config(2.0))
    assert result.metrics["developability_deviation_negative"] > 0.0
    assert result.curvature_classes["area_fraction_saddle_reverse_double_curvature"] > 0.8


def test_core_curvature_metrics_and_thresholds_are_scale_invariant() -> None:
    mesh = saddle_patch()
    scaled = mesh.copy()
    scaled.apply_scale(10.0)
    first = compute_metrics(mesh, config(2.0))
    second = compute_metrics(scaled, config(20.0))
    for name in (
        "curvature_energy",
        "developability_deviation",
        "developability_deviation_positive",
        "developability_deviation_negative",
        "developability_area_ratio",
        "curvature_fairness",
        "local_plate_twist",
    ):
        assert second.metrics[name] == pytest.approx(first.metrics[name], rel=2e-9, abs=1e-10)
    assert second.metrics["h_threshold"] == pytest.approx(first.metrics["h_threshold"] / 10.0)
    assert second.metrics["k_threshold"] == pytest.approx(first.metrics["k_threshold"] / 100.0)
    assert second.curvature_classes == pytest.approx(first.curvature_classes, rel=2e-9)


def test_release_metrics_are_invariant_to_global_orientation() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    reversed_mesh = mesh.copy()
    reversed_mesh.faces = reversed_mesh.faces[:, ::-1]
    first = compute_metrics(mesh, config(2.0))
    second = compute_metrics(reversed_mesh, config(2.0))
    np.testing.assert_allclose(second.local_fields["H"], -first.local_fields["H"], atol=2e-12)
    np.testing.assert_allclose(second.local_fields["K"], first.local_fields["K"], atol=2e-12)
    for name in ("curvature_energy", "developability_deviation", "curvature_fairness"):
        assert second.metrics[name] == pytest.approx(first.metrics[name], abs=2e-12)


def test_section_waviness_curve_controls_and_scale_invariance() -> None:
    parameter = np.linspace(0.0, 2.0 * np.pi, 4097)
    circle = np.column_stack((np.cos(parameter), np.sin(parameter)))
    circle_value = curve_waviness(circle, resample_points=256)
    assert circle_value.physical < 1e-10
    assert circle_value.fft == 0.0

    ellipse = np.column_stack((2.0 * np.cos(parameter), np.sin(parameter)))
    values = [curve_waviness(ellipse, resample_points=n).physical for n in (64, 128, 256, 512)]
    assert abs(values[-1] - values[-2]) < abs(values[-2] - values[-3])
    assert curve_waviness(10.0 * ellipse, resample_points=512).physical == pytest.approx(
        values[-1], rel=2e-10
    )


def test_section_waviness_responds_to_frequency_and_fft_cutoff() -> None:
    parameter = np.linspace(0.0, 2.0 * np.pi, 4097)

    def perturbed(frequency: int, amplitude: float):
        radius = 1.0 + amplitude * np.sin(frequency * parameter)
        curve = np.column_stack((radius * np.cos(parameter), radius * np.sin(parameter)))
        return curve_waviness(curve, resample_points=512, fft_cutoff_fraction=0.1)

    low = perturbed(8, 0.005)
    high = perturbed(64, 0.005)
    larger = perturbed(8, 0.01)
    assert high.physical > low.physical
    assert larger.physical > low.physical
    assert high.fft > low.fft
    assert high.fft_cutoff_index == 26


def test_section_intersection_validity_and_component_policy(monkeypatch) -> None:
    parameter = np.linspace(0.0, 2.0 * np.pi, 257)
    circle = np.column_stack((np.cos(parameter), np.sin(parameter)))
    shorter = 0.2 * circle
    mesh = plane_patch()
    monkeypatch.setattr(
        sections_module,
        "section_curves_yz",
        lambda _mesh, _x: [shorter, circle],
    )
    values = sections_module.section_waviness(mesh, n_stations=4, min_points=5)
    assert values["valid_sections"] == 4
    assert values["multicomponent_sections"] == 4
    assert values["section_component_policy"] == "longest_intersection_polyline"

    monkeypatch.setattr(sections_module, "section_curves_yz", lambda _mesh, _x: [])
    invalid = sections_module.section_waviness(mesh, n_stations=4, min_points=5)
    assert invalid["valid_sections"] == 0
    assert invalid["invalid_sections"] == 4


def test_local_normal_variation_is_zero_on_plane_mesh_dependent_and_scale_invariant() -> None:
    plane = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    assert compute_metrics(plane, config(1.0)).metrics["local_plate_twist"] == 0.0

    coarse = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    fine = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    coarse_value = compute_metrics(coarse, config(2.0)).metrics["local_plate_twist"]
    fine_value = compute_metrics(fine, config(2.0)).metrics["local_plate_twist"]
    assert fine_value < coarse_value
    scaled = coarse.copy()
    scaled.apply_scale(7.0)
    assert compute_metrics(scaled, config(14.0)).metrics["local_plate_twist"] == pytest.approx(
        coarse_value
    )


def test_report_schema_and_api_smoke(tmp_path) -> None:
    mesh_path = tmp_path / "sphere.ply"
    trimesh.creation.icosphere(subdivisions=1).export(mesh_path)
    result = assess_hull(mesh_path, config=config(2.0))
    assert result.metadata["schema_version"] == SCHEMA_VERSION
    assert result.metadata["metric_status"]["curvature_fairness"] == (
        "screened_experimental_nonrecommended"
    )
    assert result.metadata["metric_status"]["local_plate_twist"] == "diagnostic_mesh_dependent"
    assert result.metadata["estimator_metadata"]["fairness_integrator"].startswith("p1_")
    out = tmp_path / "results"
    write_outputs(result, out)
    payload = json.loads((out / "metrics.json").read_text())
    assert payload["metadata"]["section_settings"]["resampling_points"] == 64
    csv_text = (out / "signature.csv").read_text()
    assert "I_D_plus" in csv_text
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["estimator_metadata"]["curvature_method"]
    assert "supplied triangle mesh is the assessed representation" in (
        out / "report.html"
    ).read_text()
