#!/usr/bin/env python3
"""Generate deterministic structured meshes of the standard Wigley surface."""

from __future__ import annotations

import argparse
import json

import numpy as np
import trimesh

try:
    from benchmarks.scripts.multi_hull_common import (
        REPO_ROOT,
        mesh_metadata,
        mirror_half_mesh,
    )
except ModuleNotFoundError:  # direct execution from benchmarks/scripts
    from multi_hull_common import REPO_ROOT, mesh_metadata, mirror_half_mesh

DEFAULT_LEVELS = {
    "coarse": (40, 10),
    "medium": (80, 20),
    "fine": (160, 40),
}


def wigley_half_mesh(
    *,
    length: float = 1.0,
    breadth: float = 0.1,
    draft: float = 0.0625,
    longitudinal_panels: int = 80,
    vertical_panels: int = 20,
) -> trimesh.Trimesh:
    """Return the starboard half of the standard Wigley hull.

    The analytical surface is

    ``y = B/2 * (1 - (2x/L)^2) * (1 - (z/T)^2)``

    for ``-L/2 <= x <= L/2`` and ``-T <= z <= 0``.  The sampling is
    structured and uniform in ``x`` and ``z`` so CAD tessellation quality is
    not a confounding variable.
    """
    if min(length, breadth, draft) <= 0.0:
        raise ValueError("Wigley dimensions must be positive")
    if longitudinal_panels < 2 or vertical_panels < 2:
        raise ValueError("Wigley panel counts must be at least two")

    x = np.linspace(-0.5 * length, 0.5 * length, longitudinal_panels + 1)
    z = np.linspace(-draft, 0.0, vertical_panels + 1)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    yy = 0.5 * breadth * (1.0 - (2.0 * xx / length) ** 2) * (1.0 - (zz / draft) ** 2)
    vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    row = vertical_panels + 1
    faces: list[list[int]] = []
    for i in range(longitudinal_panels):
        for j in range(vertical_panels):
            a = i * row + j
            b = a + 1
            c = (i + 1) * row + j
            d = c + 1
            # Alternating diagonals avoid a persistent one-direction bias.
            if (i + j) % 2:
                faces.extend(([a, c, b], [b, c, d]))
            else:
                faces.extend(([a, c, d], [a, d, b]))
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--breadth", type=float, default=0.1)
    parser.add_argument("--draft", type=float, default=0.0625)
    parser.add_argument("--resolutions", nargs="+", default=["coarse", "medium", "fine"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = set(args.resolutions) - set(DEFAULT_LEVELS)
    if unknown:
        raise SystemExit(f"Unknown resolutions: {', '.join(sorted(unknown))}")
    for resolution in args.resolutions:
        nx, nz = DEFAULT_LEVELS[resolution]
        half = wigley_half_mesh(
            length=args.length,
            breadth=args.breadth,
            draft=args.draft,
            longitudinal_panels=nx,
            vertical_panels=nz,
        )
        full, seam_audit = mirror_half_mesh(half, seam_tolerance=1e-12)
        output_directory = REPO_ROOT / f"benchmarks/wigley/data/multi_hull/{resolution}"
        output_directory.mkdir(parents=True, exist_ok=True)
        half_path = output_directory / "wigley_half.stl"
        full_path = output_directory / "wigley_full.stl"
        half.export(half_path, file_type="stl")
        full.export(full_path, file_type="stl")
        payload = {
            "schema_version": 1,
            "benchmark": "wigley",
            "classification": "analytical_control_not_real_hull",
            "resolution": resolution,
            "definition": {
                "equation": "y=+/-B/2*(1-(2*x/L)^2)*(1-(z/T)^2)",
                "length": args.length,
                "breadth": args.breadth,
                "draft": args.draft,
                "longitudinal_panels": nx,
                "vertical_panels": nz,
                "sampling": "uniform structured x-z grid with alternating diagonals",
            },
            "representations": {
                "half": mesh_metadata(half, lref=args.length, output_path=half_path),
                "full": {
                    **mesh_metadata(full, lref=args.length, output_path=full_path),
                    "derivation": {
                        "operation": "mirror_and_weld_y0",
                        **seam_audit,
                    },
                },
            },
        }
        result_path = REPO_ROOT / f"benchmarks/wigley/results/multi_hull/{resolution}/geometry.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"{resolution}: half={len(half.vertices)}v/{len(half.faces)}f, "
            f"full={len(full.vertices)}v/{len(full.faces)}f"
        )


if __name__ == "__main__":
    main()
