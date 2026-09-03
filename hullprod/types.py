from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any


@dataclass
class ProducibilityConfig:
    """Configuration for geometry-based hull producibility assessment."""

    length_ref: float | None = None
    eps: float = 1.0e-12
    k_threshold: float | None = None
    h_threshold: float | None = None
    k_threshold_factor: float = 1.0e-4
    h_threshold_factor: float = 1.0e-4
    n_stations: int = 0
    station_margin: float = 0.02
    fft_cutoff_fraction: float = 0.25
    smooth_iterations: int = 0
    min_section_points: int = 25
    section_resample_points: int = 256
    section_side_policy: str = "as_represented"
    section_centerline_tolerance: float | None = None
    robust_metrics: bool = False
    experimental_metrics: bool = False
    robust_vertex_area_percentile: float = 5.0
    robust_edge_length_percentile: float = 5.0
    robust_h_clip_percentile: float = 95.0
    brep_root_indices: tuple[int, ...] | None = None
    brep_quadrature_order: int = 5
    brep_quadrature_tolerance: float = 1.0e-4
    brep_quadrature_max_depth: int = 5
    brep_quadrature_base_subdivisions: int = 1
    brep_classifier_tolerance: float = 1.0e-9
    brep_compute_fairness: bool = False
    brep_display_mesh: bool = True
    brep_display_quality: str = "standard"
    brep_display_linear_deflection: float | None = None
    brep_display_angular_deflection: float | None = None
    brep_cache: bool = True

    def __post_init__(self) -> None:
        if self.section_side_policy not in {"as_represented", "starboard_half"}:
            raise ValueError(
                "section_side_policy must be 'as_represented' or 'starboard_half'"
            )
        if self.section_centerline_tolerance is not None and self.section_centerline_tolerance < 0:
            raise ValueError("section_centerline_tolerance must be non-negative")
        if self.section_resample_points < 5:
            raise ValueError("section_resample_points must be at least 5")
        if self.n_stations < 0:
            raise ValueError("n_stations must be non-negative")
        if not 0.0 < self.fft_cutoff_fraction < 1.0:
            raise ValueError("fft_cutoff_fraction must lie strictly between 0 and 1")


@dataclass
class MetricResult:
    """Container for raw and derived producibility metrics."""

    metrics: dict[str, float | None]
    curvature_classes: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    local_fields: dict[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> dict[str, Any]:
        """Return the exact HullProd 1.0 recommended signature."""
        return {
            "I_D": self.metrics.get("developability_deviation"),
            "I_D_plus": self.metrics.get("developability_deviation_positive"),
            "I_D_minus": self.metrics.get("developability_deviation_negative"),
            "a_C": {
                "flat": self.curvature_classes.get("area_fraction_flat"),
                "single": self.curvature_classes.get(
                    "area_fraction_cylindrical_single_curvature"
                ),
                "elliptic": self.curvature_classes.get(
                    "area_fraction_elliptic_double_curvature"
                ),
                "saddle": self.curvature_classes.get(
                    "area_fraction_saddle_reverse_double_curvature"
                ),
            },
        }

    @property
    def validity(self) -> dict[str, Any]:
        """Return metric-specific validity records."""
        return self.metadata.get("metric_validity", {})

    @property
    def provenance(self) -> dict[str, Any]:
        """Return representation and numerical provenance."""
        return self.metadata

    @property
    def output_paths(self) -> dict[str, str]:
        """Return paths written by the shared assessment pipeline."""
        return self.metadata.get("output_paths", {})

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["local_fields"] = {k: "array" for k in self.local_fields}

        def clean(value: Any) -> Any:
            if isinstance(value, float) and not isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(item) for item in value]
            return value

        return clean(d)
