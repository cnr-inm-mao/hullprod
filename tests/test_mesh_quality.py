import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from test_analytic_meshes import make_open_cylinder

from hullprod import analyze_mesh_quality, cli
from hullprod.io import clean_mesh
from hullprod.metrics import compute_metrics
from hullprod.types import MetricResult, ProducibilityConfig


def test_regular_closed_sphere_has_good_curvature_reliability():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    quality = analyze_mesh_quality(mesh)

    assert quality["curvature_reliability"]["status"] == "good"
    assert quality["curvature_reliability"]["reasons"] == []
    assert quality["topology"]["boundary_edge_count"] == 0
    assert quality["triangle"]["normalized_shape_quality"]["minimum"] > 0.9


def test_regular_open_cylinder_is_advisory_but_not_poor():
    mesh = make_open_cylinder(radius=1.0, height=5.0, sections=64)
    quality = analyze_mesh_quality(mesh)
    reliability = quality["curvature_reliability"]

    assert reliability["status"] == "caution"
    assert "open_boundary" in {reason["code"] for reason in reliability["reasons"]}
    assert not any(reason["severity"] == "poor" for reason in reliability["reasons"])


def test_extreme_synthetic_sliver_triggers_poor_advisory():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0e-9, 1.0e-12, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    quality = analyze_mesh_quality(mesh)
    reliability = quality["curvature_reliability"]

    assert reliability["status"] == "poor"
    codes = {reason["code"] for reason in reliability["reasons"]}
    assert "extreme_sliver_triangle" in codes
    assert "extremely_short_edge" in codes


def test_metric_result_contains_machine_readable_mesh_quality():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    result = compute_metrics(
        mesh,
        ProducibilityConfig(length_ref=2.0, n_stations=5),
    )

    assert result.metadata["curvature_reliability_status"] == "good"
    quality = result.metadata["mesh_quality"]
    assert quality["schema_version"] == 1
    assert quality["edge"]["dimensionless_ratios"]["minimum_to_median"] > 0.0
    json.dumps(quality, allow_nan=False)


def test_cli_prints_one_concise_warning_for_poor_mesh(monkeypatch, capsys, tmp_path):
    result = MetricResult(
        metrics={},
        curvature_classes={},
        metadata={
            "curvature_reliability_status": "poor",
            "mesh_quality": {"curvature_reliability": {"status": "poor"}},
        },
    )
    monkeypatch.setattr(cli, "assess_hull", lambda *args, **kwargs: result)

    exit_code = cli.main(["assess", "synthetic.stl", "--out", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err.count("Warning:") == 1
    assert "mesh_quality metadata" in captured.err
    assert json.loads(captured.out)["metadata"]["curvature_reliability_status"] == "poor"


_DTMB_COARSE = Path(__file__).parents[1] / (
    "benchmarks/dtmb5415/data/DTMB5415_half_hull_no_transom_coarse_m.stl"
)


@pytest.mark.skipif(not _DTMB_COARSE.exists(), reason="local ignored DTMB mesh is unavailable")
def test_dtmb_coarse_reports_curvature_reliability_concerns():
    mesh = clean_mesh(trimesh.load_mesh(_DTMB_COARSE, force="mesh"))
    quality = analyze_mesh_quality(mesh)
    reliability = quality["curvature_reliability"]

    assert reliability["status"] == "poor"
    codes = {reason["code"] for reason in reliability["reasons"]}
    assert "extreme_sliver_triangle" in codes
    assert "open_boundary" in codes
