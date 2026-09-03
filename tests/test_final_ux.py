"""Final compact-report, reference-unit, and human-status UX gates."""

from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh

from hullprod import cli
from hullprod.reference_length import reference_length_preflight, resolve_reference_length
from hullprod.report import write_outputs
from hullprod.types import MetricResult
from hullprod.units import brep_unit_metadata
from hullprod.validity import brep_quadrature_note, metric_validity_record, public_status_label


def _valid(value: float, *, representation: str = "brep") -> dict:
    return metric_validity_record(
        value=value,
        validity="valid",
        reason="The retained global metric satisfies its numerical contract.",
        representation=representation,
        backend="brep_native" if representation == "brep" else "mesh_rusinkiewicz",
        represented_area=10.0,
        valid_area=10.0,
    )


def _valid_brep_with_local_caution() -> MetricResult:
    metric_validity = {
        "developability_deviation": _valid(54.591522),
        "developability_deviation_positive": _valid(29.551654),
        "developability_deviation_negative": _valid(25.039868),
        "curvature_classes": _valid(1.0),
    }
    return MetricResult(
        metrics={
            "surface_area": 10.0,
            "length_ref": 200000.0,
            "developability_deviation": 54.591522,
            "developability_deviation_positive": 29.551654,
            "developability_deviation_negative": 25.039868,
            "developability_area_ratio": 0.75,
            "curvature_energy": None,
            "curvature_fairness": None,
            "section_waviness": None,
            "section_waviness_fft": None,
            "local_plate_twist": None,
        },
        curvature_classes={
            "area_fraction_flat": 0.1,
            "area_fraction_cylindrical_single_curvature": 0.15,
            "area_fraction_elliptic_double_curvature": 0.4,
            "area_fraction_saddle_reverse_double_curvature": 0.35,
        },
        metadata={
            "hullprod_version": "1.0.0",
            "representation": "brep",
            "backend": "brep_native",
            "n_faces": 48,
            "input_geometry": {
                "path": "real_hull.igs",
                "name": "real_hull.igs",
                "extension": ".igs",
                "sha256": "abc123",
            },
            "source_format": "iges",
            "units": brep_unit_metadata(
                {"source_units": {"declared_name": "MM", "working_unit": "millimetre"}}
            ),
            "reference_length": {
                "value": 200000.0,
                "mode": "explicit_user",
                "method": "explicit_user_value",
            },
            "curvature_thresholds": {
                "h_threshold_factor": 1.0e-4,
                "k_threshold_factor": 1.0e-4,
            },
            "metric_validity": metric_validity,
            "quadrature": {
                "convergence_status": "caution",
                "unconverged_cells_at_maximum_depth": 15333,
                "unconverged_cells_at_maximum_depth_by_pass": {"core": 15333},
                "one_level_relative_change": {"developability_deviation": 6.87e-5},
                "dominant_cells": [{"face": 1, "uv_bounds": [0.0, 1.0]}],
                "one_level_shallower_values": {"developability_deviation": 54.5878},
                "two_levels_shallower_values": {"developability_deviation": 54.58},
            },
            "phase_timings": {"total_elapsed_seconds": 210.0},
            "canonical_metrics_depend_on_display_mesh": False,
            "experimental_metrics_enabled": False,
        },
    )


def test_reference_mismatch_uses_working_units_and_keeps_user_value() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [200000.0, 0.0, 0.0], [50000.0, 1000.0, 500.0]]
    )
    value, metadata = resolve_reference_length(points, 200.0, source="test_geometry")
    messages, warning = reference_length_preflight(
        metadata,
        brep_unit_metadata(
            {"source_units": {"declared_name": "MM", "working_unit": "millimetre"}}
        ),
    )
    assert value == 200.0
    assert metadata["mode"] == "explicit_user"
    assert metadata["automatic_comparison"]["value"] == pytest.approx(200000.0, rel=2e-5)
    assert metadata["plausibility"]["user_to_automatic_ratio"] == pytest.approx(
        0.001, rel=2e-5
    )
    assert metadata["plausibility"]["warning_triggered"] is True
    assert messages == [
        "Source unit: millimetre",
        "Working unit: mm",
        "Reference length: 200 mm (user supplied)",
    ]
    assert "200 mm" in warning
    assert "automatic geometric span is approximately" in warning
    assert "199999.7 mm" in warning
    assert "working unit" in warning


def test_plausible_explicit_reference_has_no_warning() -> None:
    points = np.array([[0.0, 0.0, 0.0], [200000.0, 0.0, 0.0], [100000.0, 1.0, 0.0]])
    value, metadata = resolve_reference_length(points, 200000.0, source="test_geometry")
    _, warning = reference_length_preflight(
        metadata,
        brep_unit_metadata(
            {"source_units": {"declared_name": "MM", "working_unit": "millimetre"}}
        ),
    )
    assert value == 200000.0
    assert warning is None
    assert metadata["plausibility"]["warning_triggered"] is False


def test_mesh_cli_preflight_warning_provenance_and_specific_status(tmp_path, capsys) -> None:
    source = tmp_path / "long_box.stl"
    trimesh.creation.box(extents=[200000.0, 1000.0, 500.0]).export(source)

    automatic_out = tmp_path / "automatic"
    assert cli.main([str(source), "--out", str(automatic_out), "--no-plots"]) == 0
    captured = capsys.readouterr()
    assert "Source unit: not declared" in captured.err
    assert "Working unit: input-length units" in captured.err
    assert "auto principal-axis span" in captured.err
    assert captured.err.index("Reference length:") < captured.err.index(
        "Computing triangle-mesh curvature"
    )

    mismatch_out = tmp_path / "mismatch"
    assert cli.main(
        [str(source), "--out", str(mismatch_out), "--lref", "200", "--no-plots"]
    ) == 0
    captured = capsys.readouterr()
    assert "WARNING: User-supplied reference length is 200 input-length units" in captured.err
    assert captured.err.index("WARNING:") < captured.err.index(
        "Computing triangle-mesh curvature"
    )
    assert "MESH-SENSITIVE" in captured.out

    provenance = json.loads((mismatch_out / "provenance.json").read_text())
    reference = provenance["reference_length"]
    assert reference["value"] == 200.0
    assert reference["mode"] == "explicit_user"
    assert reference["automatic_comparison"]["value"] == pytest.approx(200000.0)
    assert reference["plausibility"]["warning_triggered"] is True
    assert reference["plausibility"]["message"] in provenance["warnings"]
    validity = json.loads((mismatch_out / "validity.json").read_text())
    assert validity["developability_deviation"]["status"] == "mesh_representation_sensitive"
    report = (mismatch_out / "report.html").read_text()
    assert "MESH-SENSITIVE" in report

    quiet_out = tmp_path / "quiet_mismatch"
    assert cli.main(
        [
            str(source),
            "--out",
            str(quiet_out),
            "--lref",
            "200",
            "--no-plots",
            "--quiet",
        ]
    ) == 0
    captured = capsys.readouterr()
    assert "WARNING: User-supplied reference length" in captured.err
    assert "Loading triangle mesh" not in captured.err


def test_valid_brep_quadrature_caution_is_secondary_in_console_and_html(
    monkeypatch, capsys, tmp_path
) -> None:
    result = _valid_brep_with_local_caution()

    def fake_assess(*_args, **_kwargs):
        return result

    monkeypatch.setattr(cli, "assess_hull", fake_assess)
    assert cli.main(["real_hull.igs", "--out", str(tmp_path / "unused"), "--quiet"]) == 0
    console = capsys.readouterr().out
    assert "I_D" in console and "VALID" in console
    assert "BRep quadrature: CAUTION" in console
    assert "15,333" in console
    assert "Last relative change in I_D: 6.87e-05" in console
    assert result.metadata["metric_validity"]["developability_deviation"]["status"] == "valid"

    note = brep_quadrature_note(result.metadata)
    assert note is not None and note["scientific_status_unchanged"] is True
    report_dir = tmp_path / "report"
    write_outputs(result, report_dir)
    report = (report_dir / "report.html").read_text()
    assert "BRep quadrature CAUTION" in report
    assert "15,333" in report and "6.87e-05" in report
    assert "Scientific status" in report and "VALID" in report
    for internal in (
        "dominant_cells",
        "uv_bounds",
        "one_level_shallower_values",
        "two_levels_shallower_values",
    ):
        assert internal not in report


def test_unresolved_and_nonintegrable_human_statuses_are_unchanged() -> None:
    unresolved = metric_validity_record(
        value=None,
        validity="quadrature_unconverged",
        reason="bounded test did not converge",
        representation="brep",
        backend="brep_native",
    )
    singular = metric_validity_record(
        value=None,
        validity="geometric_singularity_nonintegrable",
        reason="integral diverges",
        representation="brep",
        backend="brep_native",
    )
    assert unresolved["status"] == "quadrature_unconverged"
    assert singular["status"] == "geometric_singularity_nonintegrable"
    assert public_status_label(unresolved) == "UNCONVERGED"
    assert public_status_label(singular) == "SINGULAR"
