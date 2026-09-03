"""Packaging-facing checks for the unified mesh/native-BRep installation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_import_and_mesh_assessment_leave_ocp_lazy(tmp_path: Path) -> None:
    """The mandatory CAD dependency must not impose eager OCP startup cost."""
    mesh_path = tmp_path / "sphere.ply"
    code = """
import sys
import trimesh

import hullprod

assert not any(name == "OCP" or name.startswith("OCP.") for name in sys.modules)
mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
mesh.export(sys.argv[1])
result = hullprod.assess_hull(sys.argv[1], n_stations=0)
assert result.metadata["backend"] == "mesh_rusinkiewicz"
assert not any(name == "OCP" or name.startswith("OCP.") for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(mesh_path)],
        check=True,
        capture_output=True,
        text=True,
    )
