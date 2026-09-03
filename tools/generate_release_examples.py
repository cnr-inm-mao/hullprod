"""Regenerate the small repository-owned HullProd 1.0 smoke inputs."""

from pathlib import Path

import trimesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCP.IGESControl import IGESControl_Writer

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "hullprod" / "examples"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    trimesh.creation.icosphere(subdivisions=1, radius=1.0).export(
        OUTPUT / "simple_sphere.stl", file_type="stl_ascii"
    )
    writer = IGESControl_Writer("MM", 0)
    if not writer.AddShape(BRepPrimAPI_MakeSphere(1.0).Shape()):
        raise RuntimeError("OpenCascade refused the analytical sphere shape")
    if not writer.Write(str(OUTPUT / "simple_sphere.iges")):
        raise RuntimeError("OpenCascade failed to write the analytical IGES fixture")


if __name__ == "__main__":
    main()
