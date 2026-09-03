"""Public HullProd 1.0 CLI, schema, export, and API acceptance gates."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import trimesh
import vtk
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from hullprod import __version__, assess, cli
from hullprod.io import clean_mesh, load_mesh
from hullprod.metrics import compute_metrics
from hullprod.plotting import validated_colorbar_ticks
from hullprod.schema import CURVATURE_CLASS_MAPPING, RECOMMENDED_SIGNATURE_METRICS
from hullprod.types import ProducibilityConfig


def _mesh_input(tmp_path: Path) -> Path:
    path = tmp_path / "simple.stl"
    trimesh.creation.icosphere(subdivisions=1, radius=1.0).export(path)
    return path


def test_version_and_primary_help_are_user_facing() -> None:
    version = subprocess.run(
        [sys.executable, "-m", "hullprod.cli", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert version.stdout.strip() == "hullprod 1.0.1"
    assert __version__ == "1.0.1"
    help_result = subprocess.run(
        [sys.executable, "-m", "hullprod.cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "geometry" in help_result.stdout
    assert "IGES/STEP" in help_result.stdout
    assert "--lref" in help_result.stdout
    assert "--experimental" in help_result.stdout
    assert "--brep-quadrature" not in help_result.stdout


def test_root_cli_outputs_and_nonempty_directory_contract(tmp_path: Path) -> None:
    source = _mesh_input(tmp_path)
    out = tmp_path / "custom_dir"
    assert cli.main([str(source), "--out", str(out), "--quiet"]) == 0
    required = {
        "signature.json",
        "signature.csv",
        "validity.json",
        "provenance.json",
        "summary.txt",
        "plots/developability_density.png",
        "plots/curvature_classes.png",
        "fields/surface_fields.vtp",
        "fields/surface_fields.csv",
    }
    assert all((out / relative).is_file() for relative in required)
    assert cli.main([str(source), "--out", str(out), "--quiet"]) != 0
    assert cli.main([str(source), "--out", str(out), "--overwrite", "--quiet"]) == 0


def test_no_plots_unsupported_and_malformed_fail_cleanly(tmp_path: Path, capsys) -> None:
    source = _mesh_input(tmp_path)
    out = tmp_path / "without_plots"
    assert cli.main([str(source), "--out", str(out), "--no-plots", "--quiet"]) == 0
    assert not (out / "plots").exists()

    unsupported = tmp_path / "surface.xyz"
    unsupported.write_text("not geometry", encoding="utf-8")
    assert cli.main([str(unsupported), "--quiet"]) != 0
    assert "Unsupported geometry format '.xyz'" in capsys.readouterr().err

    malformed = tmp_path / "broken.stl"
    malformed.write_text("not an STL", encoding="utf-8")
    assert cli.main([str(malformed), "--quiet"]) != 0
    error = capsys.readouterr().err
    assert "Error:" in error
    assert "Traceback" not in error


def test_experimental_nonfinite_values_serialize_as_json_null(tmp_path: Path) -> None:
    source = _mesh_input(tmp_path)
    out = tmp_path / "experimental"
    assert cli.main(
        [str(source), "--out", str(out), "--experimental", "--no-plots", "--quiet"]
    ) == 0
    payload = json.loads((out / "signature.json").read_text(encoding="utf-8"))
    assert set(payload["recommended_signature"]) == {"I_D", "I_D_plus", "I_D_minus", "a_C"}
    assert "curvature_energy" in payload["experimental"]
    assert "curvature_fairness" in payload["experimental"]
    report = (out / "report.html").read_text(encoding="utf-8")
    assert "Experimental metrics" in report
    assert "curvature_energy" in report
    assert report.index("Recommended signature") < report.index("Experimental metrics")


def test_default_schema_and_api_internal_cli_consistency(tmp_path: Path) -> None:
    source = _mesh_input(tmp_path)
    config = ProducibilityConfig(length_ref=2.0)
    direct = compute_metrics(clean_mesh(load_mesh(source)), config=config)
    api_result = assess(source, lref=2.0)
    out = tmp_path / "cli_result"
    assert cli.main(
        [str(source), "--out", str(out), "--lref", "2", "--no-plots", "--quiet"]
    ) == 0
    payload = json.loads((out / "signature.json").read_text(encoding="utf-8"))

    assert RECOMMENDED_SIGNATURE_METRICS == (
        "developability_deviation",
        "developability_deviation_positive",
        "developability_deviation_negative",
        "curvature_classes",
    )
    assert set(payload["recommended_signature"]) == {"I_D", "I_D_plus", "I_D_minus", "a_C"}
    assert set(payload["recommended_signature"]["a_C"]) == {
        "flat",
        "single",
        "elliptic",
        "saddle",
    }
    assert payload["experimental"] == {}
    assert "developability_area_ratio" in payload["auxiliary"]
    assert direct.metrics["curvature_energy"] is None
    assert direct.metrics["curvature_fairness"] is None
    assert direct.metrics["section_waviness"] is None

    for key in ("I_D", "I_D_plus", "I_D_minus"):
        assert payload["recommended_signature"][key] == pytest.approx(api_result.signature[key])
        assert api_result.signature[key] == pytest.approx(direct.signature[key])
    for key in ("flat", "single", "elliptic", "saddle"):
        assert payload["recommended_signature"]["a_C"][key] == pytest.approx(
            api_result.signature["a_C"][key]
        )
        assert api_result.signature["a_C"][key] == pytest.approx(direct.signature["a_C"][key])
    assert api_result.signature["I_D"] == pytest.approx(
        api_result.signature["I_D_plus"] + api_result.signature["I_D_minus"]
    )
    assert sum(api_result.signature["a_C"].values()) == pytest.approx(1.0)


def test_reference_length_override_and_automatic_provenance(tmp_path: Path) -> None:
    source = _mesh_input(tmp_path)
    automatic = assess(source)
    assert automatic.metrics["length_ref"] == pytest.approx(2.0)
    assert automatic.metadata["reference_length"]["mode"] == "auto_principal_span"
    assert automatic.metadata["reference_length"]["method"] == (
        "centroid_radial_diameter_isotropic_fallback"
    )
    explicit = assess(source, lref=7.5)
    assert explicit.metrics["length_ref"] == 7.5
    assert explicit.metadata["reference_length"]["mode"] == "explicit_user"
    assert explicit.metadata["reference_length"]["is_lpp"] is False
    assert explicit.metadata["reference_length"]["automatic_comparison"]["value"] == (
        pytest.approx(2.0)
    )


def test_exported_vtp_has_required_point_fields(tmp_path: Path) -> None:
    source = _mesh_input(tmp_path)
    out = tmp_path / "result"
    result = assess(source, out_dir=out)
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(out / "fields" / "surface_fields.vtp"))
    reader.Update()
    polydata = reader.GetOutput()
    assert reader.GetErrorCode() == 0
    assert polydata.GetNumberOfPoints() > 0
    assert polydata.GetNumberOfPolys() > 0
    fields = {
        polydata.GetPointData().GetArrayName(index)
        for index in range(polydata.GetPointData().GetNumberOfArrays())
    }
    assert {
        "developability_density",
        "developability_positive",
        "developability_negative",
        "curvature_class_id",
        "valid_mask",
    } <= fields
    assert Path(result.output_paths["surface_fields_vtp"]).is_file()


def test_continuous_colorbar_ticks_are_normalized_monotonic_and_numeric() -> None:
    fig, axis = plt.subplots()
    mappable = ScalarMappable(norm=Normalize(vmin=0.125, vmax=9.875), cmap="viridis")
    colorbar = fig.colorbar(mappable, ax=axis)
    ticks = validated_colorbar_ticks(colorbar, 0.125, 9.875)
    fig.canvas.draw()
    labels = [
        label.get_text().replace(chr(8722), "-") for label in colorbar.ax.get_yticklabels()
    ]
    numeric_labels = np.asarray([float(label) for label in labels if label])
    assert np.all(np.diff(ticks) >= 0.0)
    assert ticks[0] >= 0.125 and ticks[-1] <= 9.875
    np.testing.assert_allclose(numeric_labels, ticks)
    plt.close(fig)
    assert CURVATURE_CLASS_MAPPING[-1] == "invalid"
    assert len(CURVATURE_CLASS_MAPPING) == 5
