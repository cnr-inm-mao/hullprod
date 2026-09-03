"""Representation-specific backends behind HullProd's common assessment API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import trimesh

from .io import clean_mesh, load_mesh
from .mesh_ops import largest_connected_component
from .metrics import compute_metrics
from .types import MetricResult, ProducibilityConfig
from .units import mesh_unit_metadata

MESH_SUFFIXES = frozenset({".stl", ".obj", ".ply"})
BREP_SUFFIXES = frozenset({".igs", ".iges", ".stp", ".step"})


@dataclass(frozen=True)
class BackendAssessment:
    """Common result plus the optional geometry used only for output drawing."""

    result: MetricResult
    visualization_mesh: trimesh.Trimesh | None


class GeometryBackend(Protocol):
    """Minimal contract shared by the mesh and native-BRep implementations."""

    name: str
    representation: str

    def assess(
        self,
        geometry: Path,
        config: ProducibilityConfig,
        *,
        largest_component: bool,
        progress: Callable[[str], None] | None = None,
    ) -> BackendAssessment: ...


class MeshBackend:
    """Stable 1.0 triangulated-surface backend."""

    name = "mesh_rusinkiewicz"
    representation = "mesh"

    def assess(
        self,
        geometry: Path,
        config: ProducibilityConfig,
        *,
        largest_component: bool,
        progress: Callable[[str], None] | None = None,
    ) -> BackendAssessment:
        emit = progress or (lambda _message: None)
        emit("Loading triangle mesh...")
        mesh = clean_mesh(load_mesh(geometry))
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0 or mesh.area <= 0.0:
            raise ValueError(f"Mesh is empty or has no positive-area surface: {geometry}")
        if largest_component:
            mesh = largest_connected_component(mesh)
        emit(f"Loaded {len(mesh.vertices):,} vertices and {len(mesh.faces):,} triangles.")
        result = compute_metrics(mesh, config=config, progress=progress)
        result.metadata.setdefault("representation", self.representation)
        result.metadata.setdefault("backend", self.name)
        result.metadata.setdefault("source_format", geometry.suffix.lower().lstrip("."))
        result.metadata.setdefault("canonical_metrics_depend_on_display_mesh", False)
        result.metadata.setdefault("units", mesh_unit_metadata())
        return BackendAssessment(result=result, visualization_mesh=mesh)


class BRepBackend:
    """Direct parametric-surface backend; its display mesh is non-canonical."""

    name = "brep_native"
    representation = "brep"

    def assess(
        self,
        geometry: Path,
        config: ProducibilityConfig,
        *,
        largest_component: bool,
        progress: Callable[[str], None] | None = None,
    ) -> BackendAssessment:
        if largest_component:
            raise ValueError(
                "--largest-component is mesh-specific; select BRep roots explicitly "
                "with brep_root_indices"
            )
        from .brep_display import generate_brep_display
        from .brep_geometry import load_brep
        from .brep_metrics import compute_brep_metrics
        from .cache import brep_cache_key, load_cached_result, save_cached_result
        from .reference_length import reference_length_preflight
        from .units import brep_unit_metadata

        emit = progress or (lambda _message: None)
        emit("Loading native BRep...")
        load_started = perf_counter()
        model = load_brep(geometry, root_indices=config.brep_root_indices)
        load_elapsed = perf_counter() - load_started
        emit(f"Loaded {len(model.faces)} faces in {load_elapsed:.1f} s.")
        cache_key = brep_cache_key(model.metadata["source_sha256"], config)
        cache_started = perf_counter()
        cached = load_cached_result(cache_key) if config.brep_cache else None
        cache_elapsed = perf_counter() - cache_started
        if cached is not None:
            preflight_messages, reference_warning = reference_length_preflight(
                cached.metadata["reference_length"],
                brep_unit_metadata(model.metadata),
            )
            for message in preflight_messages:
                emit(message)
            if reference_warning:
                emit(f"WARNING: {reference_warning}")
            emit("Using cached native BRep metric integrals and exact sections.")
            display_mesh = None
            cached.metadata["cache"] = {"enabled": True, "key": cache_key, "hit": True}
            phase_timings = {
                "native_brep_load_seconds": load_elapsed,
                "canonical_cache_lookup_seconds": cache_elapsed,
            }
            if config.brep_display_mesh:
                quality = config.brep_display_quality.upper()
                emit(f"Generating {quality} display tessellation...")
                display_started = perf_counter()
                length_ref = float(cached.metrics["length_ref"])
                display = generate_brep_display(model, config, length_ref=length_ref)
                display_elapsed = perf_counter() - display_started
                display_mesh = display.mesh
                cached.local_fields = display.local_fields
                cached.metadata["display_mesh"] = display.metadata
                phase_timings["display_tessellation_seconds"] = display_elapsed
                emit(
                    "Display tessellation completed: "
                    f"{len(display_mesh.faces):,} triangles in {display_elapsed:.1f} s."
                )
            else:
                cached.metadata["display_mesh"] = {
                    "generated": False,
                    "display_quality": config.brep_display_quality,
                    "canonical_metrics_depend_on_display_mesh": False,
                }
            cached.metadata["phase_timings"] = phase_timings
            return BackendAssessment(result=cached, visualization_mesh=display_mesh)
        assessment = compute_brep_metrics(model, config=config, progress=progress)
        assessment.result.metadata.setdefault("phase_timings", {})["native_brep_load_seconds"] = (
            load_elapsed
        )
        assessment.result.metadata["phase_timings"]["canonical_cache_lookup_seconds"] = (
            cache_elapsed
        )
        assessment.result.metadata["cache"] = {
            "enabled": bool(config.brep_cache),
            "key": cache_key,
            "hit": False,
        }
        if config.brep_cache:
            save_cached_result(cache_key, assessment.result)
        return BackendAssessment(
            result=assessment.result,
            visualization_mesh=assessment.display_mesh,
        )


def backend_for_path(path: str | Path, *, override: str | None = None) -> GeometryBackend:
    """Select a validated backend from an explicit override or file extension."""
    suffix = Path(path).suffix.lower()
    if override is not None:
        normalized = override.strip().lower()
        if normalized == "mesh":
            return MeshBackend()
        if normalized in {"brep", "cad"}:
            return BRepBackend()
        raise ValueError(f"Unknown geometry backend: {override}")
    if suffix in MESH_SUFFIXES:
        return MeshBackend()
    if suffix in BREP_SUFFIXES:
        return BRepBackend()
    supported = ", ".join(sorted(MESH_SUFFIXES | BREP_SUFFIXES))
    raise ValueError(
        f"Unsupported geometry format {suffix!r}. Supported: {supported}"
    )


def backend_summary(backend: GeometryBackend) -> dict[str, Any]:
    """Return stable backend identity metadata for reports and services."""
    return {"representation": backend.representation, "backend": backend.name}
