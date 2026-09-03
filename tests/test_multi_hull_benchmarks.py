from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

common = pytest.importorskip("benchmarks.scripts.multi_hull_common")
wigley = pytest.importorskip("benchmarks.scripts.wigley_control")

DEFAULT_CONFIG = common.DEFAULT_CONFIG
cut_full_mesh = common.cut_full_mesh
mirror_half_mesh = common.mirror_half_mesh
tessellation_deflections = common.tessellation_deflections
wigley_half_mesh = wigley.wigley_half_mesh


def test_campaign_config_uses_dimensionless_tessellation_ladder() -> None:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    ladder = config["tessellation_ladder"]

    assert ladder["coarse"]["linear_deflection_over_lref"] == 3.50e-3
    assert ladder["medium"]["linear_deflection_over_lref"] == 1.75e-3
    assert ladder["fine"]["linear_deflection_over_lref"] == 8.75e-4
    assert all(not level.get("relative_deflection", False) for level in ladder.values())

    cases = config["benchmarks"]
    dtmb_physical, dtmb_ocp = tessellation_deflections(cases["dtmb5415"], 3.5e-3)
    kvlcc_physical, kvlcc_ocp = tessellation_deflections(cases["kvlcc2"], 3.5e-3)
    onrt_physical, onrt_ocp = tessellation_deflections(cases["onrt"], 3.5e-3)
    assert np.isclose(dtmb_physical, 0.02002)
    assert np.isclose(dtmb_ocp, 20.02)
    assert np.isclose(kvlcc_physical, 1.12)
    assert np.isclose(kvlcc_ocp, 1.12)
    assert np.isclose(onrt_physical, 0.0110145)
    assert np.isclose(onrt_ocp, 3.5)


def test_wigley_half_mesh_matches_analytical_surface() -> None:
    length = 2.0
    breadth = 0.3
    draft = 0.2
    nx = 12
    nz = 6
    mesh = wigley_half_mesh(
        length=length,
        breadth=breadth,
        draft=draft,
        longitudinal_panels=nx,
        vertical_panels=nz,
    )

    x, y, z = mesh.vertices.T
    expected = 0.5 * breadth * (1.0 - (2.0 * x / length) ** 2) * (1.0 - (z / draft) ** 2)
    assert len(mesh.vertices) == (nx + 1) * (nz + 1)
    assert len(mesh.faces) == 2 * nx * nz
    assert np.allclose(y, expected, atol=1e-14)
    assert np.allclose(mesh.bounds, [[-1.0, 0.0, -0.2], [1.0, 0.15, 0.0]])


def test_wigley_mirror_and_cut_preserve_half_geometry() -> None:
    half = wigley_half_mesh(longitudinal_panels=20, vertical_panels=8)
    full, mirror_audit = mirror_half_mesh(half, seam_tolerance=1e-12)
    recovered, cut_audit = cut_full_mesh(full, seam_tolerance=1e-12)

    assert np.isclose(full.area, 2.0 * half.area, rtol=1e-12, atol=1e-12)
    assert np.isclose(recovered.area, half.area, rtol=1e-12, atol=1e-12)
    assert np.isclose(full.bounds[0, 1], -full.bounds[1, 1], atol=1e-14)
    assert np.min(recovered.vertices[:, 1]) >= -1e-14
    assert mirror_audit["maximum_projection_m"] <= 1e-12
    assert cut_audit["maximum_projection_m"] <= 1e-12
