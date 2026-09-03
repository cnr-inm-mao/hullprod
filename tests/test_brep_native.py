"""Analytical gates for the optional native-BRep backend."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("OCP")

from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_Transform,
)
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeSphere,
    BRepPrimAPI_MakeTorus,
)
from OCP.Geom import Geom_SphericalSurface
from OCP.gp import gp_Ax3, gp_Pln, gp_Pnt, gp_Trsf
from OCP.IFSelect import IFSelect_RetDone
from OCP.IGESControl import IGESControl_Writer
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopoDS import TopoDS_Compound

from hullprod import cli
from hullprod.api import assess_hull
from hullprod.backends import BRepBackend, MeshBackend, backend_for_path
from hullprod.brep_display import generate_brep_display
from hullprod.brep_geometry import (
    BRepEvaluationError,
    evaluate_surface_differential,
    load_brep,
    model_from_shape,
)
from hullprod.brep_metrics import compute_brep_metrics
from hullprod.brep_quadrature import integrate_brep_metrics
from hullprod.brep_sections import brep_section_waviness
from hullprod.types import ProducibilityConfig


def _config(**updates) -> ProducibilityConfig:
    values = {
        "length_ref": 2.0,
        "n_stations": 0,
        "brep_display_mesh": False,
        "brep_quadrature_order": 7,
        "brep_quadrature_tolerance": 1.0e-4,
        "brep_quadrature_max_depth": 0,
        "brep_quadrature_base_subdivisions": 4,
        "brep_compute_fairness": True,
    }
    values.update(updates)
    return ProducibilityConfig(**values)


def _write_iges(path: Path, shape) -> None:
    writer = IGESControl_Writer("MM", 0)
    assert writer.AddShape(shape)
    assert writer.Write(str(path))


def _write_step(path: Path, shape) -> None:
    writer = STEPControl_Writer()
    assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
    assert writer.Write(str(path)) == IFSelect_RetDone


def _compound(*shapes):
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for shape in shapes:
        builder.Add(compound, shape)
    return compound


def _polygon_face(points):
    polygon = BRepBuilderAPI_MakePolygon()
    for point in points:
        polygon.Add(gp_Pnt(*point))
    polygon.Close()
    return BRepBuilderAPI_MakeFace(polygon.Wire()).Face()


def test_backend_dispatch_is_representation_explicit() -> None:
    assert isinstance(backend_for_path("hull.stl"), MeshBackend)
    assert isinstance(backend_for_path("hull.iges"), BRepBackend)
    assert isinstance(backend_for_path("hull.step"), BRepBackend)
    with pytest.raises(ValueError, match="Unsupported"):
        backend_for_path("hull.xyz")


def test_d2_curvature_matches_opencascade_properties() -> None:
    model = model_from_shape(BRepPrimAPI_MakeTorus(3.0, 1.0).Face())
    value = evaluate_surface_differential(model.faces[0], 0.71, 1.13)
    assert value.mean_curvature == pytest.approx(value.lprop_mean_curvature, abs=1e-12)
    assert value.gaussian_curvature == pytest.approx(
        value.lprop_gaussian_curvature,
        abs=1e-12,
    )


def test_rank_deficient_parameter_point_is_explicitly_invalid() -> None:
    """A removable CAD pole must not be assigned a fabricated point value."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    model = model_from_shape(BRepPrimAPI_MakeSphere(1.0).Face())
    surface = BRepAdaptor_Surface(model.faces[0], True)
    with pytest.raises(BRepEvaluationError, match="singular surface parameterization"):
        evaluate_surface_differential(
            model.faces[0],
            0.5,
            surface.LastVParameter(),
            surface=surface,
        )


def test_trimmed_plane_has_zero_native_curvature_and_fairness() -> None:
    face = BRepBuilderAPI_MakeFace(gp_Pln(), -2.0, 3.0, -1.0, 2.0).Face()
    result = integrate_brep_metrics(
        model_from_shape(face),
        _config(length_ref=5.0),
        length_ref=5.0,
    )
    assert result.values["surface_area"] == pytest.approx(15.0)
    assert result.values["curvature_energy"] == pytest.approx(0.0, abs=1e-14)
    assert result.values["developability_deviation"] == pytest.approx(0.0, abs=1e-14)
    assert result.values["curvature_fairness"] == pytest.approx(0.0, abs=1e-14)
    assert result.class_areas["flat"] == pytest.approx(15.0)


def test_smooth_sphere_face_partition_does_not_change_integrated_metrics() -> None:
    surface = Geom_SphericalSurface(gp_Ax3(), 1.0)
    half_a = BRepBuilderAPI_MakeFace(
        surface,
        0.0,
        math.pi,
        -math.pi / 2.0,
        math.pi / 2.0,
        1.0e-9,
    ).Face()
    half_b = BRepBuilderAPI_MakeFace(
        surface,
        math.pi,
        2.0 * math.pi,
        -math.pi / 2.0,
        math.pi / 2.0,
        1.0e-9,
    ).Face()
    partitioned = model_from_shape(_compound(half_a, half_b))
    single = model_from_shape(BRepPrimAPI_MakeSphere(1.0).Face())
    first = integrate_brep_metrics(single, _config(), length_ref=2.0)
    second = integrate_brep_metrics(partitioned, _config(), length_ref=2.0)
    for name in (
        "surface_area",
        "curvature_energy",
        "developability_deviation",
        "curvature_fairness",
    ):
        assert second.values[name] == pytest.approx(first.values[name], rel=1e-10, abs=1e-12)


def test_sharp_shared_edge_is_reported_not_smoothed() -> None:
    first = _polygon_face([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    second = _polygon_face([(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)])
    sewing = BRepBuilderAPI_Sewing(1.0e-9)
    sewing.Add(first)
    sewing.Add(second)
    sewing.Perform()
    result = compute_brep_metrics(model_from_shape(sewing.SewedShape()), _config()).result
    continuity = result.metadata["continuity"]
    assert continuity["shared_edge_continuity_counts"]["C0"] == 1
    assert continuity["sharp_or_C0_shared_edge_count"] == 1
    assert result.metrics["curvature_fairness"] == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize(
    ("shape", "length_ref", "area", "energy", "developability", "fairness"),
    [
        (BRepPrimAPI_MakeSphere(1.0).Face(), 2.0, 4.0 * math.pi, 4.0, 4.0, 0.0),
        (BRepPrimAPI_MakeCylinder(2.0, 5.0).Face(), 5.0, 20.0 * math.pi, 1.5625, 0.0, 0.0),
        (
            BRepPrimAPI_MakeTorus(3.0, 1.0).Face(),
            8.0,
            12.0 * math.pi**2,
            16.97056274847714,
            13.5812218105084,
            67.88225099390856,
        ),
    ],
)
def test_analytical_brep_integrals(
    shape,
    length_ref: float,
    area: float,
    energy: float,
    developability: float,
    fairness: float,
) -> None:
    model = model_from_shape(shape)
    result = integrate_brep_metrics(
        model,
        _config(length_ref=length_ref),
        length_ref=length_ref,
    )
    assert result.values["surface_area"] == pytest.approx(area, rel=2e-8)
    assert result.values["curvature_energy"] == pytest.approx(energy, rel=2e-6)
    assert result.values["developability_deviation"] == pytest.approx(
        developability,
        rel=2e-6,
        abs=1e-12,
    )
    assert result.values["curvature_fairness"] == pytest.approx(
        fairness,
        rel=3e-5,
        abs=1e-12,
    )
    assert abs(result.metadata["area_relative_discrepancy"]) < 2e-8


def test_torus_quadrature_converges_to_independent_reference() -> None:
    model = model_from_shape(BRepPrimAPI_MakeTorus(3.0, 1.0).Face())
    reference = 67.88225099390856
    errors = []
    for order in (3, 5, 7):
        config = _config(
            length_ref=8.0,
            brep_quadrature_order=order,
            brep_quadrature_base_subdivisions=4,
        )
        value = integrate_brep_metrics(model, config, length_ref=8.0)
        errors.append(abs(value.values["curvature_fairness"] - reference))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1.0e-4


def test_orientation_and_uniform_scale_semantics() -> None:
    face = BRepPrimAPI_MakeSphere(1.0).Face()
    forward = model_from_shape(face)
    reversed_model = model_from_shape(face.Reversed())
    first = evaluate_surface_differential(forward.faces[0], 0.7, 0.2)
    second = evaluate_surface_differential(reversed_model.faces[0], 0.7, 0.2)
    assert second.mean_curvature == pytest.approx(-first.mean_curvature)
    assert second.gaussian_curvature == pytest.approx(first.gaussian_curvature)
    assert second.gradient_mean_squared == pytest.approx(first.gradient_mean_squared)

    transform = gp_Trsf()
    transform.SetScaleFactor(10.0)
    scaled_shape = BRepBuilderAPI_Transform(face, transform, True).Shape()
    original = integrate_brep_metrics(forward, _config(), length_ref=2.0)
    scaled = integrate_brep_metrics(
        model_from_shape(scaled_shape),
        _config(length_ref=20.0),
        length_ref=20.0,
    )
    assert scaled.values["curvature_energy"] == pytest.approx(
        original.values["curvature_energy"], rel=1e-10
    )
    assert scaled.values["developability_deviation"] == pytest.approx(
        original.values["developability_deviation"], rel=1e-10
    )
    assert scaled.values["curvature_fairness"] == pytest.approx(
        original.values["curvature_fairness"], abs=1e-12
    )


def test_exact_sphere_sections_have_zero_waviness() -> None:
    model = model_from_shape(BRepPrimAPI_MakeSphere(1.0).Shape())
    config = _config(n_stations=7, section_resample_points=96)
    metrics, metadata = brep_section_waviness(model, config, length_ref=2.0)
    assert metrics["valid_sections"] == 7
    assert metrics["section_waviness"] < 1e-20
    assert metrics["section_waviness_fft"] == 0.0
    assert all(station["valid"] for station in metadata["stations"])


def test_visualization_resolution_does_not_change_canonical_values() -> None:
    model = model_from_shape(BRepPrimAPI_MakeSphere(1.0).Shape())
    config = _config()
    before = integrate_brep_metrics(model, config, length_ref=2.0).values
    coarse = generate_brep_display(
        model,
        _config(
            brep_display_linear_deflection=0.3,
            brep_display_angular_deflection=0.8,
        ),
        length_ref=2.0,
    )
    fine = generate_brep_display(
        model,
        _config(
            brep_display_linear_deflection=0.03,
            brep_display_angular_deflection=0.1,
        ),
        length_ref=2.0,
    )
    after = integrate_brep_metrics(model, config, length_ref=2.0).values
    assert len(fine.mesh.faces) > len(coarse.mesh.faces)
    assert after == pytest.approx(before, rel=1e-12, abs=1e-12)
    assert not coarse.metadata["canonical_metrics_depend_on_display_mesh"]


def test_display_presets_increase_analytical_brep_resolution() -> None:
    model = model_from_shape(BRepPrimAPI_MakeSphere(1.0).Shape())
    displays = [
        generate_brep_display(
            model,
            _config(brep_display_quality=quality),
            length_ref=2.0,
        )
        for quality in ("draft", "standard", "fine")
    ]
    counts = [len(display.mesh.faces) for display in displays]
    assert counts[0] < counts[1] < counts[2]
    assert [display.metadata["display_quality"] for display in displays] == [
        "draft",
        "standard",
        "fine",
    ]
    assert all(
        display.metadata["canonical_metrics_depend_on_display_mesh"] is False
        for display in displays
    )
    assert all(display.metadata["field_source"].startswith("direct_BRep") for display in displays)


def test_brep_constant_density_and_station_profiles_reproduce_sphere_globals() -> None:
    model = model_from_shape(BRepPrimAPI_MakeSphere(1.0).Shape())
    result = compute_brep_metrics(
        model,
        _config(
            n_stations=7,
            section_resample_points=96,
            brep_display_mesh=True,
            brep_display_linear_deflection=0.08,
        ),
    ).result
    valid = result.local_fields["K_valid"].astype(bool)
    assert result.local_fields["curvature_energy_density"][valid] == pytest.approx(4.0)
    assert result.local_fields["developability_density"][valid] == pytest.approx(4.0)
    assert result.metrics["curvature_energy"] == pytest.approx(4.0, rel=1e-8)
    assert result.metrics["developability_deviation"] == pytest.approx(4.0, rel=1e-8)
    assert result.metrics["developability_deviation_positive"] + result.metrics[
        "developability_deviation_negative"
    ] == pytest.approx(result.metrics["developability_deviation"], rel=1e-10)
    assert np.all(result.local_fields["developability_threshold_mask"][valid] == 1)
    assert result.metrics["developability_area_ratio"] == pytest.approx(1.0)
    assert np.all(result.local_fields["curvature_class_id"][valid] == 2)
    assert result.curvature_classes["area_fraction_elliptic_double_curvature"] == (
        pytest.approx(1.0)
    )
    assert np.nanmax(np.abs(result.local_fields["curvature_fairness_density"])) < 1e-20
    assert abs(result.metrics["curvature_fairness"]) < 1e-20
    stations = result.metadata["section_settings"]["stations"]
    valid_stations = [station for station in stations if station["valid"]]
    assert np.mean([station["physical_waviness"] for station in valid_stations]) == (
        pytest.approx(result.metrics["section_waviness"])
    )
    assert np.mean([station["fft_waviness"] for station in valid_stations]) == (
        pytest.approx(result.metrics["section_waviness_fft"])
    )


def test_iges_and_step_are_imported_and_assessed_natively(tmp_path: Path) -> None:
    shape = BRepPrimAPI_MakeSphere(1.0).Shape()
    iges_path = tmp_path / "sphere.igs"
    step_path = tmp_path / "sphere.step"
    _write_iges(iges_path, shape)
    _write_step(step_path, shape)
    for path in (iges_path, step_path):
        model = load_brep(path)
        assert model.metadata["source_sha256"]
        assert model.metadata["face_count"] >= 1
        result = assess_hull(path, config=_config())
        assert result.metadata["representation"] == "brep"
        assert result.metadata["backend"] == "brep_native"
        assert result.metrics["curvature_energy"] == pytest.approx(4.0, rel=1e-8)
        assert result.metrics["local_plate_twist"] is None
        assert not result.metadata["canonical_metrics_depend_on_display_mesh"]


def test_direct_cad_cli_writes_common_report(tmp_path: Path, capsys) -> None:
    path = tmp_path / "sphere.step"
    _write_step(path, BRepPrimAPI_MakeSphere(1.0).Shape())
    out = tmp_path / "native_results"
    assert (
        cli.main(
            [
                str(path),
                "--out",
                str(out),
                "--stations",
                "0",
                "--no-plots",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "Backend: native BRep" in captured.out
    assert "Recommended signature" in captured.out
    assert "Loading native BRep" in captured.err
    assert "Source unit:" in captured.err
    assert "Working unit:" in captured.err
    assert "Working unit: mm" in captured.err
    assert "Reference length:" in captured.err
    assert "auto principal-axis span" in captured.err
    assert "Computing native BRep metric integrals" in captured.err
    assert captured.err.index("Reference length:") < captured.err.index(
        "Computing native BRep metric integrals"
    )
    assert "exact transverse sections" not in captured.err
    assert "Writing report and manifests" in captured.err
    assert "Done in" in captured.err
    assert "native BRep assessment is experimental" not in captured.err
    assert (out / "metrics.json").is_file()
    assert (out / "provenance.json").is_file()
    report = (out / "report.html").read_text(encoding="utf-8")
    assert "direct BRep derivatives and trimmed-domain quadrature" in report
    assert "Scientific status" in report
    assert "Working unit</th><td>millimetre" in report
    assert "Display tessellation affects canonical metrics</th><td>NO" in report


def test_explicit_iges_assess_keeps_json_stdout_machine_readable(tmp_path: Path) -> None:
    path = tmp_path / "sphere.igs"
    _write_iges(path, BRepPrimAPI_MakeSphere(1.0).Shape())
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "hullprod.cli",
            "assess",
            str(path),
            "--out",
            str(tmp_path / "results"),
            "--stations",
            "0",
            "--length-ref",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["metadata"]["backend"] == "brep_native"
    assert "Loading native BRep" in completed.stderr
    assert "Total number of loaded entities" in completed.stderr


def test_common_brep_result_reports_validity_and_no_twist() -> None:
    model = model_from_shape(BRepPrimAPI_MakeSphere(1.0).Face())
    assessment = compute_brep_metrics(model, _config())
    result = assessment.result
    assert result.metadata["metric_validity"]["curvature_energy"][
        "valid_area_fraction"
    ] == pytest.approx(1.0)
    assert result.metadata["metric_validity"]["curvature_fairness"][
        "valid_area_fraction"
    ] == pytest.approx(1.0)
    assert result.metadata["metric_validity"]["local_plate_twist"]["status"] == "not_applicable"


def test_authoritative_dtmb_has_metric_specific_native_validity() -> None:
    source = Path(__file__).parents[1] / "benchmarks/dtmb5415/sources/nmri_cfdws2005/5415.igs"
    if not source.is_file():
        pytest.skip("local authoritative DTMB source is not redistributed")
    model = load_brep(source, root_indices=(1, 2))
    result = compute_brep_metrics(
        model,
        ProducibilityConfig(
            length_ref=5720.0,
            n_stations=5,
            section_resample_points=64,
            brep_quadrature_max_depth=4,
            brep_compute_fairness=False,
            brep_display_mesh=False,
            brep_cache=False,
        ),
    ).result
    validity = result.metadata["metric_validity"]
    assert result.metrics["curvature_energy"] is None
    assert validity["curvature_energy"]["status"] == ("geometric_singularity_nonintegrable")
    assert math.isfinite(result.metrics["developability_deviation"])
    assert validity["developability_deviation"]["status"] == ("valid_improper_integral_convergent")
    assert result.metrics["curvature_fairness"] is None
    assert validity["curvature_fairness"]["status"] == ("geometric_singularity_nonintegrable")
    assert "insufficient_C3_for_fairness" in validity["curvature_fairness"]["validity_codes"]
    assert result.metrics["valid_sections"] == 5
    assert validity["section_waviness"]["status"] == "caution_singular_measure_zero"
    assert validity["section_waviness"]["sections_with_edge_joins"] > 0
    assert not validity["section_waviness"]["sampling_convergence_evidence_supplied"]
    assert result.to_dict()["metrics"]["curvature_energy"] is None
