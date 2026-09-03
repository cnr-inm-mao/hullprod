"""Paper-signature roles and canonical section-side contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from hullprod import sections as sections_module
from hullprod.metrics import compute_metrics
from hullprod.schema import METRIC_STATUS, RECOMMENDED_SIGNATURE_METRICS
from hullprod.sections import canonicalize_section_side, curve_waviness
from hullprod.types import ProducibilityConfig


def _grid_patch(*, saddle: bool, n: int = 17) -> trimesh.Trimesh:
    axis = np.linspace(-1.0, 1.0, n)
    vertices = np.array(
        [[x, y, 0.25 * (x**2 - y**2) if saddle else 0.0] for x in axis for y in axis]
    )
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = (i + 1) * n + j
            c = (i + 1) * n + j + 1
            d = i * n + j + 1
            faces.extend(((a, b, c), (a, c, d)))
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def _symmetric_curve() -> tuple[np.ndarray, np.ndarray]:
    full_parameter = np.linspace(0.0, 2.0 * np.pi, 4097)
    half_parameter = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 2049)
    full_radius = 1.0 + 0.015 * np.cos(12.0 * full_parameter)
    half_radius = 1.0 + 0.015 * np.cos(12.0 * half_parameter)
    full = np.column_stack(
        (full_radius * np.cos(full_parameter), full_radius * np.sin(full_parameter))
    )
    half = np.column_stack(
        (half_radius * np.cos(half_parameter), half_radius * np.sin(half_parameter))
    )
    return full, half


def test_full_and_half_curves_are_invariant_under_starboard_policy() -> None:
    full, half = _symmetric_curve()
    canonical_full = canonicalize_section_side(full, policy="starboard_half")
    canonical_half = canonicalize_section_side(half, policy="starboard_half")
    assert canonical_full.clipping_occurred
    assert not canonical_half.clipping_occurred
    assert canonical_full.closed_before and not canonical_full.closed_after
    assert canonical_full.points[0, 0] == pytest.approx(0.0, abs=1.0e-12)
    assert canonical_full.points[-1, 0] == pytest.approx(0.0, abs=1.0e-12)
    full_value = curve_waviness(canonical_full.points, resample_points=512)
    half_value = curve_waviness(canonical_half.points, resample_points=512)
    assert full_value.physical == pytest.approx(half_value.physical, rel=2.0e-5)
    assert full_value.fft == pytest.approx(half_value.fft, abs=2.0e-5)


def test_as_represented_preserves_supplied_curve_and_closure() -> None:
    full, _ = _symmetric_curve()
    retained = canonicalize_section_side(full, policy="as_represented")
    np.testing.assert_array_equal(retained.points, full)
    assert retained.closed_before and retained.closed_after
    assert not retained.clipping_occurred


def test_centerline_tolerance_retains_and_snaps_endpoints() -> None:
    curve = np.array([[-1.0, 0.0], [-1.0e-10, -0.2], [0.5, -0.1], [1.0e-10, 0.2], [-1, 0]])
    retained = canonicalize_section_side(
        curve,
        policy="starboard_half",
        centerline_tolerance=1.0e-8,
    )
    assert np.all(retained.points[:, 0] >= 0.0)
    assert retained.points[0, 0] == 0.0
    assert retained.points[-1, 0] == 0.0
    assert retained.centerline_tolerance == 1.0e-8


def test_mesh_section_policy_provenance_and_configuration_propagation(monkeypatch) -> None:
    full, _ = _symmetric_curve()
    monkeypatch.setattr(sections_module, "section_curves_yz", lambda _mesh, _x: [full])
    result = compute_metrics(
        _grid_patch(saddle=False),
        ProducibilityConfig(
            length_ref=1.0,
            n_stations=3,
            min_section_points=5,
            section_side_policy="starboard_half",
            section_resample_points=128,
            fft_cutoff_fraction=0.30,
        ),
    )
    settings = result.metadata["section_settings"]
    assert settings["section_side_policy_requested"] == "starboard_half"
    assert settings["section_side_policy_actual"] == "starboard_half"
    assert settings["resampling_points"] == 128
    assert settings["fft_cutoff_fraction"] == 0.30
    assert settings["requested_sections"] == 3
    assert all(station["section_side_clipping_occurred"] for station in settings["stations"])
    assert result.metadata["metric_validity"]["section_waviness"]["status"] == (
        "mesh_representation_sensitive"
    )


def test_threshold_area_identity_and_developability_independence() -> None:
    mesh = _grid_patch(saddle=True)
    results = [
        compute_metrics(
            mesh,
            ProducibilityConfig(
                length_ref=2.0,
                n_stations=0,
                h_threshold_factor=factor,
                k_threshold_factor=factor,
            ),
        )
        for factor in (1.0e-5, 1.0e-4, 1.0e-3)
    ]
    for result in results:
        double = (
            result.curvature_classes["area_fraction_elliptic_double_curvature"]
            + result.curvature_classes["area_fraction_saddle_reverse_double_curvature"]
        )
        assert result.metrics["developability_area_ratio"] == pytest.approx(double, abs=1e-14)
    for name in (
        "developability_deviation",
        "developability_deviation_positive",
        "developability_deviation_negative",
    ):
        assert results[0].metrics[name] == pytest.approx(results[1].metrics[name], abs=1e-14)
        assert results[2].metrics[name] == pytest.approx(results[1].metrics[name], abs=1e-14)


def test_recommended_signature_and_non_signature_roles() -> None:
    assert RECOMMENDED_SIGNATURE_METRICS == (
        "developability_deviation",
        "developability_deviation_positive",
        "developability_deviation_negative",
        "curvature_classes",
    )
    assert METRIC_STATUS["curvature_energy"] == "screened_experimental_nonrecommended"
    assert METRIC_STATUS["curvature_fairness"] == "screened_experimental_nonrecommended"
    assert METRIC_STATUS["section_waviness"] == "screened_experimental_nonrecommended"
    assert METRIC_STATUS["section_waviness_fft"] == "screened_experimental_nonrecommended"
    assert METRIC_STATUS["developability_area_ratio"] == "auxiliary_redundant"
    assert METRIC_STATUS["local_plate_twist"] == "diagnostic_mesh_dependent"


def test_retained_signature_campaign_has_exact_public_cases() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "benchmarks/config/retained_signature.json").read_text())
    assert config["cases"] == ["dtmb5415", "kcs", "jbc", "kvlcc2m"]
    assert config["section_side_policy"] == "starboard_half"
    assert config["stations"] == 40
    assert "kvlcc2" not in config["cases"]
    assert "onrt" not in config["cases"]
    assert "wigley" not in config["cases"]


def test_native_brep_full_and_half_section_policy_provenance() -> None:
    pytest.importorskip("OCP")
    from math import pi

    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    from hullprod.brep_geometry import model_from_shape
    from hullprod.brep_sections import brep_section_waviness

    full_axis = gp_Ax2(gp_Pnt(), gp_Dir(1.0, 0.0, 0.0))
    half_axis = gp_Ax2(
        gp_Pnt(),
        gp_Dir(1.0, 0.0, 0.0),
        gp_Dir(0.0, 0.0, -1.0),
    )
    full = model_from_shape(BRepPrimAPI_MakeCylinder(full_axis, 1.0, 4.0).Face())
    half = model_from_shape(BRepPrimAPI_MakeCylinder(half_axis, 1.0, 4.0, pi).Face())
    config = ProducibilityConfig(
        length_ref=4.0,
        n_stations=3,
        section_resample_points=128,
        section_side_policy="starboard_half",
        brep_display_mesh=False,
    )
    full_values, full_metadata = brep_section_waviness(full, config, length_ref=4.0)
    half_values, half_metadata = brep_section_waviness(half, config, length_ref=4.0)
    assert full_values["section_waviness"] == pytest.approx(
        half_values["section_waviness"], abs=2.0e-28
    )
    assert full_values["section_waviness_fft"] == half_values["section_waviness_fft"] == 0.0
    assert full_metadata["stations"][0]["section_side_clipping_occurred"]
    assert not half_metadata["stations"][0]["section_side_clipping_occurred"]
    assert full_metadata["representation_backend"] == "brep_exact_plane_intersection_D2"
