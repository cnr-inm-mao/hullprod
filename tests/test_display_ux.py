"""User-facing native display-quality controls and invariance gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hullprod import cli
from hullprod.brep_display import DISPLAY_QUALITY_PRESETS, resolve_display_settings
from hullprod.report import write_outputs
from hullprod.types import MetricResult, ProducibilityConfig
from hullprod.units import brep_unit_metadata, mesh_unit_metadata


def test_display_quality_cli_parsing_and_invalid_rejection() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["assess", "hull.step", "--display-quality", "fine"])
    assert args.display_quality == "fine"
    with pytest.raises(SystemExit):
        parser.parse_args(["assess", "hull.step", "--display-quality", "ultra"])


@pytest.mark.parametrize("quality", ("draft", "standard", "fine"))
def test_display_quality_preset_mapping(quality: str) -> None:
    span = 5000.0
    settings = resolve_display_settings(
        ProducibilityConfig(brep_display_quality=quality),
        geometric_max_span=span,
    )
    preset = DISPLAY_QUALITY_PRESETS[quality]
    assert settings["linear_deflection"] == pytest.approx(
        span * preset["linear_deflection_span_fraction"]
    )
    assert settings["angular_deflection"] == preset["angular_deflection"]
    assert settings["relative_deflection"] is False


def test_explicit_display_deflections_override_selected_preset() -> None:
    settings = resolve_display_settings(
        ProducibilityConfig(
            brep_display_quality="fine",
            brep_display_linear_deflection=12.5,
            brep_display_angular_deflection=0.42,
        ),
        geometric_max_span=5000.0,
    )
    assert settings["display_quality"] == "fine"
    assert settings["linear_deflection"] == 12.5
    assert settings["angular_deflection"] == 0.42
    assert settings["linear_deflection_source"] == "explicit_override"
    assert settings["angular_deflection_source"] == "explicit_override"


def test_programmatic_invalid_display_quality_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown display quality"):
        resolve_display_settings(
            ProducibilityConfig(brep_display_quality="ultra"),
            geometric_max_span=1.0,
        )


def _cli_result() -> MetricResult:
    return MetricResult(
        metrics={"surface_area": 1.0, "local_plate_twist": None},
        curvature_classes={},
        metadata={"representation": "mesh", "backend": "mesh_rusinkiewicz"},
    )


def test_assess_progress_uses_stderr_and_stdout_remains_json(monkeypatch, capsys, tmp_path) -> None:
    def fake_assess(*_args, progress=None, **_kwargs):
        progress("Measured phase completed in 1.2 s.")
        return _cli_result()

    monkeypatch.setattr(cli, "assess_hull", fake_assess)
    assert cli.main(["assess", "hull.stl", "--out", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["metadata"]["backend"] == "mesh_rusinkiewicz"
    assert "Measured phase completed" in captured.err
    assert "Done in" in captured.err


def test_quiet_suppresses_phase_messages_but_keeps_json_stdout(
    monkeypatch, capsys, tmp_path
) -> None:
    def fake_assess(*_args, progress=None, **_kwargs):
        progress("Measured phase completed in 1.2 s.")
        return _cli_result()

    monkeypatch.setattr(cli, "assess_hull", fake_assess)
    assert cli.main(["assess", "hull.stl", "--out", str(tmp_path), "--quiet"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["metadata"]["backend"] == "mesh_rusinkiewicz"
    assert captured.err == ""


def test_default_output_directory_is_cwd_and_explicit_out_wins(
    monkeypatch, capsys, tmp_path
) -> None:
    observed = []

    def fake_assess(*_args, out_dir=None, **_kwargs):
        observed.append(out_dir)
        return _cli_result()

    monkeypatch.setattr(cli, "assess_hull", fake_assess)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["hull.stl", "--quiet", "--no-plots"]) == 0
    assert Path(observed[-1]) == tmp_path / "hull_hullprod"
    explicit = tmp_path / "chosen" / "result"
    assert cli.main(["hull.stl", "--quiet", "--no-plots", "--out", str(explicit)]) == 0
    assert Path(observed[-1]) == explicit
    assert str(explicit.resolve()) in capsys.readouterr().out


def test_unit_labels_preserve_native_numbers_and_do_not_claim_mesh_si() -> None:
    native = brep_unit_metadata(
        {
            "source_units": {
                "declared_name": "M",
                "declared_flag": 6,
                "declared_value_mm": 1000.0,
                "working_unit": "millimetre",
            }
        }
    )
    assert native["source_declared_unit"] == "metre"
    assert native["working_length_unit_symbol"] == "mm"
    assert native["working_area_unit_symbol"] == "mm²"
    assert native["numerical_values_rescaled"] is False
    mesh = mesh_unit_metadata()
    assert mesh["working_length_unit_symbol"] == "input-length units"
    assert mesh["working_area_unit_symbol"] == "input-length units²"
    assert "mm" not in mesh["working_length_unit_symbol"]


def test_report_foregrounds_units_display_separation_and_links(tmp_path) -> None:
    result = MetricResult(
        metrics={"surface_area": 7.5e6, "length_ref": 4970.0},
        curvature_classes={},
        metadata={
            "hullprod_version": "1.0.0",
            "representation": "brep",
            "backend": "brep_native",
            "n_faces": 29,
            "input_geometry": {"path": "hull.igs", "sha256": "abc"},
            "units": brep_unit_metadata(
                {"source_units": {"declared_name": "M", "working_unit": "millimetre"}}
            ),
            "display_mesh": {
                "display_quality": "standard",
                "triangle_count": 19277,
                "canonical_metrics_depend_on_display_mesh": False,
            },
            "canonical_metrics_depend_on_display_mesh": False,
            "phase_timings": {"canonical_core_metrics_seconds": 81.25},
        },
    )
    write_outputs(result, tmp_path)
    report = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "HullProd version</th><td>1.0.0" in report
    assert "Source unit</th><td>metre" in report
    assert "Working unit</th><td>millimetre" in report
    assert "4970 mm" in report
    assert "Canonical metrics from native BRep</th><td>YES" in report
    assert "Display tessellation affects canonical metrics</th><td>NO" in report
    assert "Display triangles carry visualization fields only" in report
    assert "<pre>" not in report
    for entry in json.loads((tmp_path / "output_manifest.json").read_text())["files"]:
        assert entry["exists"]
