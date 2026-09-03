#!/usr/bin/env python3
"""Exercise an installed HullProd wheel across every supported input family."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import trimesh


def _write_cad_inputs(directory: Path) -> tuple[Path, Path]:
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.IGESControl import IGESControl_Writer
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    shape = BRepPrimAPI_MakeSphere(1.0).Shape()
    iges = directory / "generated_sphere.igs"
    iges_writer = IGESControl_Writer("MM", 0)
    if not iges_writer.AddShape(shape) or not iges_writer.Write(str(iges)):
        raise RuntimeError("failed to write analytical IGES sphere")
    step = directory / "generated_sphere.step"
    step_writer = STEPControl_Writer()
    if step_writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone:
        raise RuntimeError("failed to transfer analytical STEP sphere")
    if step_writer.Write(str(step)) != IFSelect_RetDone:
        raise RuntimeError("failed to write analytical STEP sphere")
    return iges, step


def _point_field_names(path: Path) -> set[str]:
    root = ET.parse(path).getroot()
    return {
        element.attrib["Name"]
        for element in root.findall("./PolyData/Piece/PointData/DataArray")
    }


def _run_case(executable: str, geometry: Path, directory: Path) -> dict:
    output = directory / f"{geometry.stem}_{geometry.suffix[1:]}_result"
    completed = subprocess.run(
        [executable, str(geometry), "--out", str(output), "--no-plots"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads((output / "signature.json").read_text(encoding="utf-8"))
    signature = payload["recommended_signature"]
    if set(signature) != {"I_D", "I_D_plus", "I_D_minus", "a_C"}:
        raise AssertionError("installed-wheel signature keys do not match the v1 contract")
    if set(signature["a_C"]) != {"flat", "single", "elliptic", "saddle"}:
        raise AssertionError("installed-wheel curvature classes do not match the v1 contract")
    if not math.isclose(
        signature["I_D"],
        signature["I_D_plus"] + signature["I_D_minus"],
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise AssertionError("installed-wheel signed developability is not additive")

    required_files = {
        "signature.csv",
        "validity.json",
        "provenance.json",
        "report.html",
        "fields/surface_fields.vtp",
        "fields/surface_fields.csv",
    }
    missing = [name for name in required_files if not (output / name).is_file()]
    if missing:
        raise AssertionError(f"installed-wheel output is incomplete: {missing}")
    expected_fields = {
        "developability_density",
        "developability_positive",
        "developability_negative",
        "curvature_class_id",
        "valid_mask",
    }
    point_fields = _point_field_names(output / "fields/surface_fields.vtp")
    if not expected_fields <= point_fields:
        raise AssertionError(f"missing VTP fields: {sorted(expected_fields - point_fields)}")

    return {
        "input": geometry.name,
        "backend": payload["backend"],
        "representation": payload["representation"],
        "reference_length_mode": payload["reference_length"]["mode"],
        "signature": signature,
        "stdout": completed.stdout,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON evidence path.")
    args = parser.parse_args(argv)
    executable = shutil.which("hullprod")
    if executable is None:
        raise RuntimeError("hullprod executable is not installed in the active environment")

    with tempfile.TemporaryDirectory(prefix="hullprod-wheel-smoke-") as temporary:
        directory = Path(temporary)
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        mesh_inputs = []
        for suffix in ("stl", "obj", "ply"):
            path = directory / f"generated_sphere.{suffix}"
            sphere.export(path)
            mesh_inputs.append(path)
        iges, step = _write_cad_inputs(directory)
        cases = [_run_case(executable, path, directory) for path in (*mesh_inputs, iges, step)]

    evidence = {"python": sys.version, "executable": sys.executable, "cases": cases}
    rendered = json.dumps(evidence, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
