from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from .schema import CURVATURE_CLASS_MAPPING
from .validity import public_status_label


def validated_colorbar_ticks(colorbar, vmin: float, vmax: float) -> np.ndarray:
    """Validate that library-generated numeric colorbar ticks match its norm."""
    ticks = np.asarray(colorbar.get_ticks(), dtype=float)
    tolerance = 1.0e-12 * max(abs(vmin), abs(vmax), 1.0)
    inside = ticks[(ticks >= vmin - tolerance) & (ticks <= vmax + tolerance)]
    if len(inside) < 2 and vmax > vmin:
        inside = np.linspace(vmin, vmax, 5)
    if len(inside) != len(ticks):
        colorbar.set_ticks(inside)
        ticks = np.asarray(colorbar.get_ticks(), dtype=float)
    if len(ticks) and np.any(np.diff(ticks) < -tolerance):
        raise ValueError("Continuous colorbar ticks are not monotonic.")
    if len(ticks) and (ticks[0] < vmin - tolerance or ticks[-1] > vmax + tolerance):
        raise ValueError("Continuous colorbar ticks lie outside the normalization range.")
    return ticks


def _face_average(values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return face-centered values from nodal or face-centered input."""
    if len(values) == len(faces):
        return np.asarray(values, dtype=float)

    return np.asarray(values, dtype=float)[faces].mean(axis=1)


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _positive_epsilon(values: np.ndarray) -> float:
    finite = np.abs(_finite(values))
    positive = finite[finite > 0.0]

    if len(positive) == 0:
        return 1e-30

    return max(float(np.nanpercentile(positive, 1.0)) * 1e-3, 1e-30)


def _robust_abs_scale(values: np.ndarray, percentile: float = 95.0) -> float:
    finite = np.abs(_finite(values))

    if len(finite) == 0:
        return 1.0

    scale = float(np.nanpercentile(finite, percentile))

    if not np.isfinite(scale) or scale <= 1e-30:
        scale = float(np.nanmax(finite)) if len(finite) else 1.0

    if not np.isfinite(scale) or scale <= 1e-30:
        return 1.0

    return scale


def _transform_values(
    values: np.ndarray,
    transform: str,
) -> tuple[np.ndarray, str]:
    """Transform values for robust visual interpretation."""
    arr = np.asarray(values, dtype=float)

    if transform == "raw":
        return arr, "raw value"

    if transform == "log10_abs":
        eps = _positive_epsilon(arr)
        return np.log10(np.abs(arr) + eps), f"log10(abs(value) + {eps:.2e})"

    if transform == "signed_asinh":
        scale = _robust_abs_scale(arr, percentile=95.0)
        return np.arcsinh(arr / scale), f"asinh(value / {scale:.2e})"

    msg = f"Unknown plotting transform: {transform}"
    raise ValueError(msg)


def _robust_limits(
    values: np.ndarray,
    lower: float = 2.0,
    upper: float = 98.0,
    symmetric: bool = False,
) -> tuple[float | None, float | None]:
    finite = _finite(values)

    if len(finite) == 0:
        return None, None

    if symmetric:
        limit = float(np.nanpercentile(np.abs(finite), upper))
        if not np.isfinite(limit) or limit <= 1e-30:
            return None, None
        return -limit, limit

    vmin, vmax = np.nanpercentile(finite, [lower, upper])

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None, None

    if abs(float(vmax) - float(vmin)) < 1e-30:
        return None, None

    return float(vmin), float(vmax)


def _field_stats(name: str, values: np.ndarray) -> dict[str, float | str]:
    finite = _finite(values)

    if len(finite) == 0:
        return {
            "field": name,
            "n": 0,
            "min": np.nan,
            "p01": np.nan,
            "p02": np.nan,
            "p05": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "p98": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }

    return {
        "field": name,
        "n": len(finite),
        "min": float(np.nanmin(finite)),
        "p01": float(np.nanpercentile(finite, 1)),
        "p02": float(np.nanpercentile(finite, 2)),
        "p05": float(np.nanpercentile(finite, 5)),
        "p50": float(np.nanpercentile(finite, 50)),
        "p95": float(np.nanpercentile(finite, 95)),
        "p98": float(np.nanpercentile(finite, 98)),
        "p99": float(np.nanpercentile(finite, 99)),
        "max": float(np.nanmax(finite)),
    }


def save_field_statistics(
    fields: dict[str, np.ndarray],
    out_path: str | Path,
) -> None:
    """Save raw field percentile statistics for plot interpretation."""
    out_path = Path(out_path)
    rows = [_field_stats(name, values) for name, values in fields.items()]

    if not rows:
        return

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_surface_field(
    mesh: trimesh.Trimesh,
    values: np.ndarray,
    out_path: str | Path,
    title: str,
    cmap: str = "viridis",
    robust: bool = True,
    transform: str = "raw",
    symmetric: bool = False,
    view_elev: float = 22.0,
    view_azim: float = -125.0,
    clip_lower: float = 2.0,
    clip_upper: float = 98.0,
    colorbar_label: str | None = None,
    status_banner: str | None = None,
) -> None:
    """Save a lightweight 3D trisurface plot with robust scalar coloring."""
    out_path = Path(out_path)
    vertices = mesh.vertices
    faces = mesh.faces

    face_values = _face_average(np.asarray(values), faces)
    plot_values, transform_label = _transform_values(face_values, transform)

    if robust:
        vmin, vmax = _robust_limits(
            plot_values,
            lower=clip_lower,
            upper=clip_upper,
            symmetric=symmetric,
        )
    else:
        finite = _finite(plot_values)
        vmin = float(np.nanmin(finite)) if len(finite) else None
        vmax = float(np.nanmax(finite)) if len(finite) else None

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    selected_cmap = plt.get_cmap(cmap).with_extremes(bad="#9e9e9e")
    collection = ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        linewidth=0.0,
        antialiased=False,
        shade=False,
        cmap=selected_cmap,
    )

    collection.set_array(plot_values)

    if vmin is not None and vmax is not None:
        collection.set_clim(vmin, vmax)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=view_elev, azim=view_azim)
    if status_banner:
        ax.text2D(
            0.5,
            0.96,
            status_banner,
            transform=ax.transAxes,
            ha="center",
            va="top",
            color="#a61b1b",
            weight="bold",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#a61b1b"},
        )
    ax.set_box_aspect(tuple(max(float(np.ptp(vertices[:, axis])), 1.0e-12) for axis in range(3)))

    colorbar = fig.colorbar(collection, ax=ax, shrink=0.65, pad=0.08)
    colorbar.set_label(colorbar_label or transform_label)
    norm = collection.norm
    validated_colorbar_ticks(colorbar, float(norm.vmin), float(norm.vmax))

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_geometry(mesh: trimesh.Trimesh, out_path: str | Path) -> None:
    """Plot the supplied or display-only geometry without scalar coloring."""
    vertices = np.asarray(mesh.vertices)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=mesh.faces,
        color="#9fb6c5",
        edgecolor="none",
        antialiased=False,
        shade=True,
    )
    ax.set_title("Hull geometry — longitudinal axis x")
    ax.set_xlabel("x (longitudinal)")
    ax.set_ylabel("y (transverse)")
    ax.set_zlabel("z (vertical)")
    ax.view_init(elev=22.0, azim=-125.0)
    ax.set_box_aspect(tuple(max(float(np.ptp(vertices[:, axis])), 1.0e-12) for axis in range(3)))
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _face_categories(mesh: trimesh.Trimesh, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=int).ravel()
    if len(array) == len(mesh.faces):
        return array
    face_values = array[np.asarray(mesh.faces)]
    result = np.full(len(mesh.faces), -1, dtype=int)
    for index, row in enumerate(face_values):
        if np.any(row < 0):
            continue
        counts = np.bincount(row, minlength=4)
        result[index] = int(np.argmax(counts))
    return result


def plot_curvature_classes(
    mesh: trimesh.Trimesh,
    values: np.ndarray,
    out_path: str | Path,
) -> None:
    """Plot the stable discrete curvature classes with an explicit legend."""
    vertices = np.asarray(mesh.vertices)
    face_values = _face_categories(mesh, values)
    colors = ["#9e9e9e", "#f4f1de", "#4ea8de", "#e76f51", "#6a4c93"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-1.5, 4.5, 1.0), cmap.N)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    collection = ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=mesh.faces,
        linewidth=0.0,
        antialiased=False,
        shade=False,
        cmap=cmap,
        norm=norm,
    )
    collection.set_array(face_values)
    ax.set_title("Curvature composition classes (invalid is not flat)")
    ax.set_xlabel("x (longitudinal)")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22.0, azim=-125.0)
    ax.set_box_aspect(tuple(max(float(np.ptp(vertices[:, axis])), 1.0e-12) for axis in range(3)))
    handles = [
        Patch(facecolor=colors[class_id + 1], label=label)
        for class_id, label in CURVATURE_CLASS_MAPPING.items()
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.98))
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_section_waviness_stations(result, out_path: str | Path) -> None:
    """Plot both stationwise waviness definitions and mark invalid stations."""
    records = result.metadata.get("section_settings", {}).get("stations", [])
    fig, physical_axis = plt.subplots(figsize=(10, 5.5))
    fft_axis = physical_axis.twinx()
    valid = [record for record in records if record.get("valid")]
    invalid = [record for record in records if not record.get("valid")]
    if valid:
        x = [record.get("x_over_L_ref", record.get("x_over_lref")) for record in valid]
        physical = [record.get("physical_waviness") for record in valid]
        fft = [record.get("fft_waviness") for record in valid]
        physical_axis.plot(x, physical, "o-", color="#1f77b4", label="physical waviness")
        fft_axis.plot(x, fft, "s-", color="#d95f02", label="FFT waviness")
    if invalid:
        x_invalid = [record.get("x_over_L_ref", record.get("x_over_lref")) for record in invalid]
        physical_axis.scatter(
            x_invalid,
            np.zeros(len(x_invalid)),
            marker="x",
            color="#555555",
            label="invalid station",
            zorder=5,
        )
    physical_axis.set_xlabel("station x / L_ref")
    physical_axis.set_ylabel("physical waviness (dimensionless)", color="#1f77b4")
    fft_axis.set_ylabel("FFT high-frequency ratio (dimensionless)", color="#d95f02")
    physical_axis.set_title("Stationwise section waviness — exact validity shown")
    physical_axis.grid(True, alpha=0.25)
    handles_1, labels_1 = physical_axis.get_legend_handles_labels()
    handles_2, labels_2 = fft_axis.get_legend_handles_labels()
    physical_axis.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_not_applicable(out_path: str | Path, title: str, reason: str) -> None:
    """Create an explicit unavailable/N/A panel rather than a finite-looking map."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.43, reason, ha="center", va="center", wrap=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_field_histogram(
    values: np.ndarray,
    out_path: str | Path,
    title: str,
    transform: str = "raw",
    bins: int = 120,
) -> None:
    """Save a histogram of a local scalar field."""
    out_path = Path(out_path)

    transformed, xlabel = _transform_values(np.asarray(values), transform)
    finite = _finite(transformed)

    if len(finite) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    value_span = float(np.ptp(finite))
    constant_tolerance = max(
        np.finfo(float).eps * max(float(np.max(np.abs(finite))), 1.0),
        1.0e-300,
    )
    if value_span <= constant_tolerance:
        ax.axvline(float(finite[0]), linewidth=2.0)
        ax.text(
            0.5,
            0.92,
            "constant field",
            ha="center",
            va="top",
            transform=ax.transAxes,
        )
    else:
        try:
            ax.hist(finite, bins=min(bins, max(1, int(np.sqrt(len(finite))))))
        except ValueError:
            # NumPy can reject finite bins when floating-point variation is
            # smaller than the representable bin spacing around the offset.
            ax.axvline(float(np.median(finite)), linewidth=2.0)
            ax.text(
                0.5,
                0.92,
                "numerically constant field",
                ha="center",
                va="top",
                transform=ax.transAxes,
            )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_default_plots(
    mesh: trimesh.Trimesh,
    result,
    out_dir: str | Path,
    *,
    experimental: bool = False,
) -> dict[str, str]:
    """Save the stable 1.0 plot set with explicit formulas and validity."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = result.local_fields
    validity = result.metadata.get("metric_validity", {})

    def record(name: str):
        return validity.get(name) or (
            validity.get("developability") if name.startswith("developability_") else None
        )

    def status(name: str, *, prefix: str | None = None) -> str | None:
        label = public_status_label(record(name))
        parts = [prefix] if prefix else []
        if label not in {"VALID", "VALID*"}:
            parts.append(label)
        return " — ".join(parts) if parts else None

    written: dict[str, str] = {}
    if "developability_density" in fields:
        path = out_dir / "developability_density.png"
        plot_surface_field(
            mesh,
            fields["developability_density"],
            path,
            "abs(K) L_ref² — normalized double-curvature intensity",
            transform="log10_abs",
            cmap="viridis",
            robust=False,
            colorbar_label="log10(abs(K) L_ref² + stated epsilon)",
            status_banner=status("developability_deviation"),
        )
        written["developability_density"] = str(path.resolve())
    if "curvature_class_id" in fields:
        path = out_dir / "curvature_classes.png"
        plot_curvature_classes(mesh, fields["curvature_class_id"], path)
        written["curvature_classes"] = str(path.resolve())
    if not experimental:
        return written

    save_field_statistics(fields, out_dir / "field_statistics.csv")
    plot_geometry(mesh, out_dir / "geometry.png")

    if "H" in fields:
        plot_surface_field(
            mesh,
            fields["H"],
            out_dir / "mean_curvature_H.png",
            "Signed mean curvature H=(k1+k2)/2 [inverse input length]",
            transform="raw",
            cmap="coolwarm",
            symmetric=True,
            robust=False,
            colorbar_label="signed H [1 / input length]",
            status_banner=status("curvature_energy"),
        )

    if "K" in fields:
        plot_surface_field(
            mesh,
            fields["K"],
            out_dir / "gaussian_curvature_K.png",
            "Signed Gaussian curvature K=k1*k2 [inverse input length squared]",
            transform="raw",
            cmap="coolwarm",
            symmetric=True,
            robust=False,
            colorbar_label="signed K [1 / input length²]",
            status_banner=status("developability_deviation"),
        )

    if "curvature_energy_density" in fields:
        plot_surface_field(
            mesh,
            fields["curvature_energy_density"],
            out_dir / "curvature_energy_density.png",
            "(H L_ref)² — local nondimensional mean-curvature content",
            transform="log10_abs",
            cmap="magma",
            robust=False,
            colorbar_label="log10((H L_ref)² + stated epsilon)",
            status_banner=status("curvature_energy"),
        )

    if "developability_density" in fields:
        plot_surface_field(
            mesh,
            fields["developability_density"],
            out_dir / "developability_density.png",
            "abs(K) L_ref² — local double-curvature / non-developability intensity",
            transform="log10_abs",
            cmap="viridis",
            robust=False,
            colorbar_label="log10(abs(K) L_ref² + stated epsilon)",
            status_banner=status("developability_deviation"),
        )

    if "developability_positive_density" in fields:
        signed = np.asarray(fields["developability_positive_density"]) - np.asarray(
            fields["developability_negative_density"]
        )
        plot_surface_field(
            mesh,
            signed,
            out_dir / "developability_signed.png",
            "Signed developability K L_ref²: positive elliptic / negative saddle-reverse",
            transform="signed_asinh",
            cmap="coolwarm",
            robust=False,
            symmetric=True,
            status_banner=status("developability_deviation"),
        )

    if "curvature_class_id" in fields:
        plot_curvature_classes(
            mesh,
            fields["curvature_class_id"],
            out_dir / "curvature_classes.png",
        )

    plot_section_waviness_stations(result, out_dir / "section_waviness_stations.png")

    fairness_record = record("curvature_fairness")
    fairness_reason = (
        fairness_record.get("reason", "Fairness was not evaluated.")
        if fairness_record
        else "Fairness status is unavailable."
    )
    fairness_values = fields.get("curvature_fairness_density")
    if fairness_values is not None and np.any(np.isfinite(fairness_values)):
        plot_surface_field(
            mesh,
            fairness_values,
            out_dir / "curvature_fairness_density.png",
            "L_ref⁴ |grad_s H|² — HIGH-SENSITIVITY GEOMETRIC DIAGNOSTIC",
            transform="log10_abs",
            cmap="magma",
            robust=False,
            colorbar_label="log10(L_ref⁴ |grad_s H|² + stated epsilon)",
            status_banner=status(
                "curvature_fairness", prefix="HIGH-SENSITIVITY GEOMETRIC DIAGNOSTIC"
            ),
        )
    else:
        plot_not_applicable(
            out_dir / "curvature_fairness_density.png",
            f"Curvature fairness: {public_status_label(fairness_record)}",
            fairness_reason,
        )

    twist_record = record("local_plate_twist")
    if result.metadata.get("representation") == "mesh" and "face_twist" in fields:
        plot_surface_field(
            mesh,
            fields["face_twist"],
            out_dir / "local_plate_twist.png",
            "Local plate twist — MESH-DEPENDENT DIHEDRAL DIAGNOSTIC",
            transform="log10_abs",
            cmap="viridis",
            robust=False,
            colorbar_label="log10(mean incident dihedral angle + stated epsilon)",
            status_banner="MESH-DEPENDENT DIAGNOSTIC",
        )
    else:
        plot_not_applicable(
            out_dir / "local_plate_twist.png",
            "Local plate twist: N/A for native BRep",
            twist_record.get("reason", "This diagnostic is defined only for mesh inputs.")
            if twist_record
            else "This diagnostic is defined only for mesh inputs.",
        )
    return written
