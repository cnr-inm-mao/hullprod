"""Stable 1.0 output-directory, manifest, plot, and report gates."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import trimesh

from hullprod import assess_hull
from hullprod.schema import FIELD_SCHEMA_VERSION, OUTPUT_LAYOUT_VERSION
from hullprod.types import ProducibilityConfig


def test_complete_mesh_output_directory_is_manifested_and_linked(tmp_path: Path) -> None:
    source = tmp_path / "sphere.ply"
    trimesh.creation.icosphere(subdivisions=1, radius=1.0).export(source)
    output = tmp_path / "sphere_hullprod"
    result = assess_hull(
        source,
        config=ProducibilityConfig(
            length_ref=2.0,
        ),
        out_dir=output,
        make_plots=True,
    )
    required = {
        "report.html",
        "signature.json",
        "signature.csv",
        "validity.json",
        "summary.txt",
        "metrics.json",
        "provenance.json",
        "output_manifest.json",
        "field_manifest.json",
        "fields/surface_fields.vtp",
        "fields/surface_fields.csv",
        "geometry/display_mesh.stl",
        "plots/developability_density.png",
        "plots/curvature_classes.png",
    }
    for relative in required:
        assert (output / relative).is_file(), relative

    manifest = json.loads((output / "output_manifest.json").read_text())
    assert manifest["output_layout_version"] == OUTPUT_LAYOUT_VERSION
    manifested = {entry["path"] for entry in manifest["files"]}
    assert required <= manifested
    assert all(entry["exists"] for entry in manifest["files"])

    field_manifest = json.loads((output / "field_manifest.json").read_text())
    assert field_manifest["field_schema_version"] == FIELD_SCHEMA_VERSION
    fields = {field["name"]: field for field in field_manifest["fields"]}
    assert fields["developability_density"]["association"] == "point"
    assert fields["developability_positive"]["association"] == "point"
    assert fields["developability_negative"]["association"] == "point"
    assert fields["valid_mask"]["association"] == "point"
    assert fields["curvature_class_id"]["categorical_mapping"] == {
        "-1": "invalid",
        "0": "flat",
        "1": "single curvature",
        "2": "elliptic double curvature",
        "3": "saddle/reverse double curvature",
    }
    assert fields["developability_density"]["corresponding_global_metric"] == (
        "developability_deviation"
    )
    assert set(fields["H"]["exported_files_containing_field"]) == {
        "fields/surface_fields.vtp",
        "fields/surface_fields.csv",
    }

    report = (output / "report.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in report
    assert "geometry-based screening descriptors" in report
    sections = (
        "1. Input and representation",
        "2. Recommended signature",
        "3. Curvature-class composition",
        "4. Standard plots",
        "5. Validity and numerical notes",
        "6. Concise provenance",
        "7. Output files",
        "8. Scope and limitations",
    )
    positions = [report.index(section) for section in sections]
    assert positions == sorted(positions)
    assert "I_D_plus" in report and "a_C_elliptic" in report
    assert "developability density" in report and "curvature classes" in report
    assert "MESH-SENSITIVE" in report
    assert "Full provenance: provenance.json" in report
    assert "Full validity record: validity.json" in report
    assert "Full machine-readable metrics: metrics.json" in report
    for hidden in (
        "curvature_energy",
        "curvature_fairness",
        "section_waviness",
        "section_waviness_fft",
        "dominant_cells",
        "uv_bounds",
        "one_level_shallower_values",
        "two_levels_shallower_values",
        "<pre>",
    ):
        assert hidden not in report
    for relative in required:
        assert f"href='{relative}'" in report or relative == "report.html"

    for relative in required:
        if relative.endswith(".png"):
            image = mpimg.imread(output / relative)
            assert image.shape[0] >= 800
            assert image.shape[1] >= 1200
    assert result.metadata["canonical_metrics_depend_on_display_mesh"] is False

    signature = json.loads((output / "signature.json").read_text())
    assert set(signature["recommended_signature"]) == {"I_D", "I_D_plus", "I_D_minus", "a_C"}
    assert set(signature["recommended_signature"]["a_C"]) == {
        "flat",
        "single",
        "elliptic",
        "saddle",
    }
    assert signature["experimental"] == {}
