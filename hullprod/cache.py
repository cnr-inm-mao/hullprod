"""Deterministic native-BRep result cache keyed by source and configuration."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ._version import __version__
from .types import MetricResult, ProducibilityConfig


def _software_version() -> str:
    return __version__


def _implementation_digest() -> str:
    """Invalidate development caches when native-backend source code changes."""
    digest = hashlib.sha256()
    package_directory = Path(__file__).resolve().parent
    for name in (
        "brep_geometry.py",
        "brep_quadrature.py",
        "brep_sections.py",
        "brep_metrics.py",
        "brep_display.py",
        "brep_validity.py",
        "reference_length.py",
        "validity.py",
    ):
        digest.update(name.encode())
        digest.update((package_directory / name).read_bytes())
    return digest.hexdigest()


def brep_cache_key(source_sha256: str, config: ProducibilityConfig) -> str:
    """Hash every setting that can affect canonical native-BRep values."""
    values = asdict(config)
    for name in tuple(values):
        if name.startswith("brep_display_") or name in {
            "brep_cache",
            "robust_metrics",
            "robust_vertex_area_percentile",
            "robust_edge_length_percentile",
            "robust_h_clip_percentile",
        }:
            values.pop(name)
    payload = {
        "source_sha256": source_sha256,
        "hullprod_version": _software_version(),
        "backend_contract": "brep_native_metric_specific_validity_v2",
        "implementation_sha256": _implementation_digest(),
        "configuration": values,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def default_cache_directory() -> Path:
    """Return the platform-conventional user cache location."""
    explicit = os.environ.get("HULLPROD_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "hullprod" / "brep"


def load_cached_result(key: str, directory: Path | None = None) -> MetricResult | None:
    """Load canonical scalars/metadata; visualization arrays are regenerated."""
    path = (directory or default_cache_directory()) / f"{key}.json"
    if not path.is_file():
        return None
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return MetricResult(
            metrics=payload["metrics"],
            curvature_classes=payload["curvature_classes"],
            metadata=payload["metadata"],
            local_fields={},
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None


def save_cached_result(
    key: str,
    result: MetricResult,
    directory: Path | None = None,
) -> Path:
    """Atomically cache only canonical values and machine-readable metadata."""
    cache_directory = directory or default_cache_directory()
    cache_directory.mkdir(parents=True, exist_ok=True)
    path = cache_directory / f"{key}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result.to_dict(), sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
