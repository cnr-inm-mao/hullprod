"""Command-line interface for the HullProd 1.0 assessment pipeline."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter

from . import __version__
from .api import assess_hull
from .backends import BREP_SUFFIXES
from .types import ProducibilityConfig
from .validity import brep_quadrature_note, public_status_label


@contextlib.contextmanager
def _native_library_stdout_to_stderr(enabled: bool):
    """Keep native-library diagnostics away from the concise stdout summary."""
    if not enabled:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
        sys.stdout.flush()
    finally:
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)


def _advanced(parser: argparse.ArgumentParser, *names: str, **kwargs) -> None:
    kwargs.setdefault("help", argparse.SUPPRESS)
    parser.add_argument(*names, **kwargs)


class _HullProdParser(argparse.ArgumentParser):
    """Parser that accepts the legacy ``assess`` token transparently."""

    def parse_args(self, args=None, namespace=None):
        values = list(sys.argv[1:] if args is None else args)
        if values and values[0] == "assess":
            values = values[1:]
        return super().parse_args(values, namespace)


def build_parser() -> argparse.ArgumentParser:
    parser = _HullProdParser(
        prog="hullprod",
        description="Geometry-based hull producibility assessment.",
        epilog=(
            "INPUT: IGES/STEP BRep or STL/OBJ/PLY triangular surface. "
            "The legacy form 'hullprod assess INPUT' is also accepted."
        ),
    )
    parser.add_argument("geometry", type=Path, help="Input geometry file.")
    parser.add_argument("--version", action="version", version=f"hullprod {__version__}")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: ./<input-stem>_hullprod).",
    )
    parser.add_argument(
        "--lref",
        "--length-ref",
        dest="lref",
        type=float,
        default=None,
        help="Reference length override (default: automatic principal-axis span).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of HullProd files in a nonempty output directory.",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip the two default PNG surface plots."
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Also evaluate screened research metrics in a separately labelled section.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress messages on stderr."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show a traceback for unexpected failures."
    )

    _advanced(parser, "--plots", action="store_true")
    _advanced(parser, "--stations", type=int, default=None)
    _advanced(parser, "--section-resample-points", type=int, default=256)
    _advanced(parser, "--fft-cutoff-fraction", type=float, default=0.25)
    _advanced(
        parser,
        "--section-side-policy",
        choices=("as_represented", "starboard_half"),
        default="as_represented",
    )
    _advanced(parser, "--section-centerline-tolerance", type=float, default=None)
    _advanced(parser, "--k-threshold", type=float, default=None)
    _advanced(parser, "--h-threshold", type=float, default=None)
    _advanced(parser, "--smooth-iterations", type=int, default=0)
    _advanced(parser, "--largest-component", action="store_true")
    _advanced(parser, "--robust-metrics", action="store_true")
    _advanced(parser, "--robust-vertex-area-percentile", type=float, default=5.0)
    _advanced(parser, "--robust-edge-length-percentile", type=float, default=5.0)
    _advanced(parser, "--robust-h-clip-percentile", type=float, default=95.0)
    _advanced(parser, "--brep-root", type=int, action="append", default=None)
    _advanced(parser, "--brep-quadrature-order", type=int, default=5)
    _advanced(parser, "--brep-quadrature-tolerance", type=float, default=1.0e-4)
    _advanced(parser, "--brep-quadrature-max-depth", type=int, default=5)
    _advanced(parser, "--fairness", action="store_true")
    _advanced(parser, "--no-display-mesh", action="store_true")
    _advanced(
        parser,
        "--display-quality",
        choices=("draft", "standard", "fine"),
        default="standard",
    )
    _advanced(parser, "--display-linear-deflection", type=float, default=None)
    _advanced(parser, "--display-angular-deflection", type=float, default=None)
    return parser


def _format_elapsed(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)}m{remainder:04.1f}s"


def _print_summary(result, geometry: Path, out_dir: Path) -> None:
    metadata = result.metadata
    representation = metadata.get("representation", "unknown")
    reference = metadata.get("reference_length", {})
    mode = (
        "auto principal-axis span"
        if reference.get("mode") == "auto_principal_span"
        else "user supplied"
    )
    print(f"HullProd {__version__}")
    print(f"Input: {geometry}")
    print(f"Input format: {geometry.suffix.lower()}")
    print(f"Backend: {'native BRep' if representation == 'brep' else 'triangle mesh'}")
    print(f"Representation: {representation}")
    length_ref = result.metrics.get("length_ref")
    length_text = "--" if length_ref is None else f"{float(length_ref):.8g}"
    length_unit = metadata.get("units", {}).get(
        "working_length_unit_symbol", "input-length units"
    )
    print(f"Reference length: {length_text} {length_unit} ({mode})")
    print("\nRecommended signature")
    validity = metadata.get("metric_validity", {})

    def metric_line(label: str, key: str) -> None:
        value = result.metrics.get(key)
        text = f"{float(value):.8g}" if value is not None and math.isfinite(float(value)) else "--"
        record = validity.get(key) or validity.get("developability")
        print(f"  {label:<10} {text:<14} {public_status_label(record)}")

    metric_line("I_D", "developability_deviation")
    metric_line("I_D+", "developability_deviation_positive")
    metric_line("I_D-", "developability_deviation_negative")
    class_status = public_status_label(validity.get("curvature_classes"))
    for label, value in result.signature["a_C"].items():
        text = "--" if value is None else f"{float(value):.8g}"
        print(f"  {label:<10} {text:<14} {class_status}")
    print("\nValidity")
    print(
        "  developability  "
        f"{public_status_label(validity.get('developability_deviation'))}"
    )
    print(f"  curvature class {class_status}")
    quadrature_note = brep_quadrature_note(metadata)
    if quadrature_note or representation == "mesh":
        print("\nNumerical notes")
    if quadrature_note:
        print("  BRep quadrature: CAUTION")
        print(f"  {quadrature_note['summary']}")
        print(
            "  Local core cells at maximum depth: "
            f"{quadrature_note['unconverged_core_cells_at_maximum_depth']:,}"
        )
        relative_change = quadrature_note["developability_one_level_relative_change"]
        if isinstance(relative_change, (int, float)):
            print(f"  Last relative change in I_D: {float(relative_change):.3g}")
        print("  See validity.json and provenance.json for details.")
    if representation == "mesh":
        print(
            "  Mesh-sensitive means derivative estimates may depend on triangle "
            "resolution, connectivity, and quality."
        )
    elapsed = metadata.get("phase_timings", {}).get("cli_total_seconds")
    if isinstance(elapsed, (int, float)):
        print(f"\nCompleted in {_format_elapsed(float(elapsed))}")
    print(f"Results written to:\n  {out_dir.resolve()}")


def main(argv=None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    legacy_alias = bool(raw_argv and raw_argv[0] == "assess")
    if legacy_alias:
        raw_argv = raw_argv[1:]
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    experimental = bool(
        args.experimental or args.robust_metrics or args.fairness or args.stations is not None
    )
    stations = args.stations if args.stations is not None else (80 if experimental else 0)
    cfg = ProducibilityConfig(
        length_ref=args.lref,
        k_threshold=args.k_threshold,
        h_threshold=args.h_threshold,
        n_stations=stations,
        section_resample_points=args.section_resample_points,
        fft_cutoff_fraction=args.fft_cutoff_fraction,
        section_side_policy=args.section_side_policy,
        section_centerline_tolerance=args.section_centerline_tolerance,
        smooth_iterations=args.smooth_iterations,
        robust_metrics=args.robust_metrics,
        experimental_metrics=experimental,
        robust_vertex_area_percentile=args.robust_vertex_area_percentile,
        robust_edge_length_percentile=args.robust_edge_length_percentile,
        robust_h_clip_percentile=args.robust_h_clip_percentile,
        brep_root_indices=tuple(args.brep_root) if args.brep_root is not None else None,
        brep_quadrature_order=args.brep_quadrature_order,
        brep_quadrature_tolerance=args.brep_quadrature_tolerance,
        brep_quadrature_max_depth=args.brep_quadrature_max_depth,
        brep_compute_fairness=args.fairness,
        brep_display_mesh=not args.no_display_mesh,
        brep_display_quality=args.display_quality,
        brep_display_linear_deflection=args.display_linear_deflection,
        brep_display_angular_deflection=args.display_angular_deflection,
    )
    out_dir = args.out or (Path.cwd() / f"{args.geometry.stem}_hullprod")
    if not args.quiet:
        print(f"HullProd {__version__} — starting assessment", file=sys.stderr)

    def progress(message: str) -> None:
        if not args.quiet or message.startswith("WARNING:"):
            print(message, file=sys.stderr, flush=True)

    started = perf_counter()
    native_input = args.geometry.suffix.lower() in BREP_SUFFIXES
    try:
        with _native_library_stdout_to_stderr(native_input):
            result = assess_hull(
                args.geometry,
                config=cfg,
                out_dir=out_dir,
                make_plots=not args.no_plots,
                largest_component=args.largest_component,
                progress=progress,
                overwrite=args.overwrite,
            )
    except Exception as error:
        if args.debug:
            raise
        print(f"Error: {error}", file=sys.stderr)
        return 2
    elapsed = perf_counter() - started
    result.metadata.setdefault("phase_timings", {})["cli_total_seconds"] = elapsed
    if not args.quiet:
        progress(f"Done in {_format_elapsed(elapsed)}.")
    if result.metadata.get("curvature_reliability_status") == "poor":
        print(
            "Warning: mesh-quality diagnostics indicate poor curvature reliability; "
            "inspect mesh_quality metadata in provenance.json and validity.json.",
            file=sys.stderr,
        )
    if legacy_alias:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_summary(result, args.geometry, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
