#!/usr/bin/env python3
"""Fail if a wheel or sdist contains private paper/benchmark geometry assets."""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path

_FORBIDDEN_SUFFIXES = {
    ".igs",
    ".iges",
    ".stp",
    ".step",
    ".stl",
    ".obj",
    ".ply",
    ".pdf",
    ".vtp",
    ".vtk",
    ".dat",
    ".zip",
}

_ALLOWED_PROJECT_OWNED_GEOMETRY = {
    "hullprod/examples/simple_sphere.iges",
    "hullprod/examples/simple_sphere.stl",
}


def _members(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return sorted(archive.namelist())
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return sorted(member.name for member in archive.getmembers() if member.isfile())
    raise ValueError(f"unsupported artifact: {path}")


def audit(path: Path) -> dict:
    members = _members(path)
    forbidden = []
    for member in members:
        relative = member.split("/", 1)[-1] if path.name.endswith(".tar.gz") else member
        parts = Path(relative).parts
        if parts and parts[0] in {"paper", "benchmarks"}:
            forbidden.append(member)
            continue
        if (
            Path(relative).suffix.lower() in _FORBIDDEN_SUFFIXES
            and relative not in _ALLOWED_PROJECT_OWNED_GEOMETRY
        ):
            forbidden.append(member)
    if forbidden:
        raise AssertionError(f"forbidden publication/data assets in {path}: {forbidden}")
    return {"artifact": str(path), "member_count": len(members), "forbidden_count": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    evidence = [audit(path) for path in args.artifacts]
    rendered = json.dumps(evidence, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
