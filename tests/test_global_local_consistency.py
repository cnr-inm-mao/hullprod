"""Deterministic global/local consistency gates for the mesh backend."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from hullprod.metrics import compute_metrics
from hullprod.schema import CurvatureClassID
from hullprod.types import ProducibilityConfig


def _result():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    result = compute_metrics(
        mesh,
        ProducibilityConfig(
            length_ref=2.0,
            n_stations=7,
            min_section_points=5,
            section_resample_points=64,
        ),
    )
    return mesh, result


def test_mesh_density_integrals_reproduce_global_curvature_metrics() -> None:
    _, result = _result()
    fields = result.local_fields
    area = fields["vertex_area"]
    h_valid = fields["H_valid"].astype(bool)
    k_valid = fields["K_valid"].astype(bool)

    energy = np.sum(area[h_valid] * fields["curvature_energy_density"][h_valid]) / np.sum(
        area[h_valid]
    )
    assert energy == pytest.approx(result.metrics["curvature_energy"], rel=1e-14)

    for field, metric in (
        ("developability_density", "developability_deviation"),
        ("developability_positive_density", "developability_deviation_positive"),
        ("developability_negative_density", "developability_deviation_negative"),
    ):
        value = np.sum(area[k_valid] * fields[field][k_valid]) / np.sum(area[k_valid])
        assert value == pytest.approx(result.metrics[metric], rel=1e-14)
    assert result.metrics["developability_deviation_positive"] + result.metrics[
        "developability_deviation_negative"
    ] == pytest.approx(result.metrics["developability_deviation"], rel=1e-14)


def test_mesh_threshold_classes_and_fairness_aggregate_exactly() -> None:
    mesh, result = _result()
    fields = result.local_fields
    area = fields["vertex_area"]
    k_valid = fields["K_valid"].astype(bool)
    threshold = fields["developability_threshold_mask"]
    ratio = np.sum(area[k_valid] * threshold[k_valid]) / np.sum(area[k_valid])
    assert ratio == pytest.approx(result.metrics["developability_area_ratio"], rel=1e-14)

    class_ids = fields["curvature_class_id"]
    mapping = {
        CurvatureClassID.FLAT: "area_fraction_flat",
        CurvatureClassID.SINGLE_CURVATURE: ("area_fraction_cylindrical_single_curvature"),
        CurvatureClassID.ELLIPTIC_DOUBLE_CURVATURE: ("area_fraction_elliptic_double_curvature"),
        CurvatureClassID.SADDLE_REVERSE_DOUBLE_CURVATURE: (
            "area_fraction_saddle_reverse_double_curvature"
        ),
    }
    for class_id, name in mapping.items():
        fraction = np.sum(area[k_valid & (class_ids == int(class_id))]) / np.sum(area[k_valid])
        assert fraction == pytest.approx(result.curvature_classes[name], rel=1e-14)
    assert np.all(class_ids[~k_valid] == int(CurvatureClassID.INVALID))

    face_valid = np.isfinite(fields["curvature_fairness_density"])
    fairness = np.sum(
        mesh.area_faces[face_valid] * fields["curvature_fairness_density"][face_valid]
    ) / np.sum(mesh.area_faces[face_valid])
    assert fairness == pytest.approx(result.metrics["curvature_fairness"], rel=1e-14)


def test_mesh_stationwise_values_reproduce_global_waviness() -> None:
    _, result = _result()
    stations = result.metadata["section_settings"]["stations"]
    valid = [station for station in stations if station["valid"]]
    assert len(valid) == result.metrics["valid_sections"]
    assert np.mean([station["physical_waviness"] for station in valid]) == pytest.approx(
        result.metrics["section_waviness"], rel=1e-14
    )
    fft = [station["fft_waviness"] for station in valid if station["fft_waviness"] is not None]
    assert np.mean(fft) == pytest.approx(result.metrics["section_waviness_fft"], rel=1e-14)
