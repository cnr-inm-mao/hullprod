from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from ._version import __version__
from .backends import backend_for_path
from .brep_geometry import sha256_file
from .export import export_visualization_fields
from .plotting import save_default_plots
from .report import write_outputs
from .types import ProducibilityConfig


def assess_hull(
    geometry: str | Path,
    config: ProducibilityConfig | None = None,
    n_stations: int | None = None,
    out_dir: str | Path | None = None,
    make_plots: bool = False,
    largest_component: bool = False,
    backend: str | None = None,
    progress: Callable[[str], None] | None = None,
    overwrite: bool = False,
):
    """Assess a mesh or native CAD BRep through the appropriate backend."""
    if config is None:
        config = ProducibilityConfig()
    if n_stations is not None:
        config.n_stations = n_stations

    started = perf_counter()
    emit = progress or (lambda _message: None)
    geometry = Path(geometry).resolve()
    if not geometry.exists():
        raise FileNotFoundError(f"Input file not found: {geometry}")
    if not geometry.is_file():
        raise ValueError(f"Input path is not a file: {geometry}")
    output_path = Path(out_dir).resolve() if out_dir is not None else None
    if output_path is not None and output_path.is_dir() and any(output_path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_path}. "
                "Choose --out elsewhere or pass --overwrite."
            )
    selected_backend = backend_for_path(geometry, override=backend)
    assessment = selected_backend.assess(
        geometry,
        config,
        largest_component=largest_component,
        progress=progress,
    )
    result = assessment.result
    result.metadata["hullprod_version"] = __version__
    result.metadata["input_geometry"] = {
        "path": str(geometry),
        "name": geometry.name,
        "extension": geometry.suffix.lower(),
        "sha256": sha256_file(geometry),
    }
    result.metadata["run_timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    if output_path is not None:
        out_dir = output_path
        if assessment.visualization_mesh is not None:
            emit("Writing visualization fields...")
            fields_started = perf_counter()
            field_paths = export_visualization_fields(
                assessment.visualization_mesh,
                result,
                out_dir / "fields",
            )
            result.metadata.setdefault("output_paths", {}).update(field_paths)
            geometry_dir = out_dir / "geometry"
            geometry_dir.mkdir(parents=True, exist_ok=True)
            assessment.visualization_mesh.export(geometry_dir / "display_mesh.stl")
            result.metadata.setdefault("phase_timings", {})["visualization_fields_seconds"] = (
                perf_counter() - fields_started
            )
            if make_plots:
                emit("Generating plots...")
                plots_started = perf_counter()
                plot_paths = save_default_plots(
                    assessment.visualization_mesh,
                    result,
                    out_dir / "plots",
                    experimental=config.experimental_metrics,
                )
                result.metadata.setdefault("output_paths", {}).update(plot_paths)
                result.metadata["phase_timings"]["plots_seconds"] = perf_counter() - plots_started
        else:
            raise RuntimeError(
                "Distributed field export requires a visualization surface; "
                "enable the BRep display tessellation."
            )
        emit("Writing report and manifests...")
        result.metadata.setdefault("phase_timings", {})["total_elapsed_seconds"] = (
            perf_counter() - started
        )
        output_paths = write_outputs(result, out_dir)
        result.metadata.setdefault("output_paths", {}).update(output_paths)

    result.metadata.setdefault("phase_timings", {})["total_elapsed_seconds"] = perf_counter() - started
    return result


def assess(
    geometry: str | Path,
    *,
    out_dir: str | Path | None = None,
    lref: float | None = None,
    plots: bool = False,
    experimental: bool = False,
    overwrite: bool = False,
    config: ProducibilityConfig | None = None,
    largest_component: bool = False,
    progress: Callable[[str], None] | None = None,
):
    """Assess one BRep or mesh file through the public HullProd 1.0 pipeline.

    The file extension selects the native BRep or triangle-mesh backend.  When
    ``out_dir`` is supplied, the same artifacts used by the CLI are written.
    """
    resolved = replace(config) if config is not None else ProducibilityConfig()
    if lref is not None:
        resolved.length_ref = float(lref)
    if experimental:
        resolved.experimental_metrics = True
        if resolved.n_stations == 0:
            resolved.n_stations = 80
    return assess_hull(
        geometry,
        config=resolved,
        out_dir=out_dir,
        make_plots=plots,
        largest_component=largest_component,
        progress=progress,
        overwrite=overwrite,
    )
