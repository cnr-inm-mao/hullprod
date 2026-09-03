from __future__ import annotations

import base64
import json
import math
from html import escape
from pathlib import Path

import pandas as pd

from .manifest import build_field_manifest, build_output_manifest
from .validity import brep_quadrature_note, public_status_label


def _json_ready(value):
    """Replace non-finite floats with JSON null throughout public payloads."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _validity_for(result, name: str):
    records = result.metadata.get("metric_validity", {})
    return records.get(name) or (
        records.get("developability") if name.startswith("developability_") else None
    )


def _embedded_plot(path: Path, relative: str) -> str:
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    label = escape(Path(relative).stem.replace("_", " "))
    return (
        f"<figure><a href='{escape(relative)}'><img src='data:image/png;base64,{encoded}' "
        f"alt='{label}'></a><figcaption>{label}</figcaption></figure>"
    )


def _html_report(
    result,
    out_dir: Path,
    relative_paths: list[str],
    _field_manifest: dict,
) -> str:
    """Render a concise self-contained user report with links to full audit data."""
    metadata = result.metadata
    representation = metadata.get("representation", "mesh")
    backend = metadata.get("backend", "unknown")
    input_geometry = metadata.get("input_geometry", {})
    units = metadata.get("units", {})
    length_unit = str(units.get("working_length_unit_symbol", "input-length units"))
    source_unit = units.get("source_declared_unit") or "not declared"
    phase_timings = metadata.get("phase_timings", {})

    def number(value: object) -> str:
        if value is None:
            return "--"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        return f"{numeric:.10g}" if math.isfinite(numeric) else "--"

    signature = result.signature
    validity = metadata.get("metric_validity", {})
    retained = (
        ("I_D", signature["I_D"], "developability_deviation", "Total intensity"),
        ("I_D_plus", signature["I_D_plus"], "developability_deviation_positive", "Elliptic intensity"),
        ("I_D_minus", signature["I_D_minus"], "developability_deviation_negative", "Saddle/reverse intensity"),
    )
    signature_rows = "".join(
        "<tr>"
        f"<th>{name}</th><td>{meaning}</td><td>{number(value)}</td>"
        f"<td class='status'>{escape(public_status_label(validity.get(key)))}</td>"
        "</tr>"
        for name, value, key, meaning in retained
    )
    class_status = public_status_label(validity.get("curvature_classes"))
    class_rows = "".join(
        "<tr>"
        f"<th>a_C_{escape(name)}</th><td>{number(value)}</td>"
        f"<td class='status'>{escape(class_status)}</td></tr>"
        for name, value in signature["a_C"].items()
    )

    experimental_keys = (
        "curvature_energy",
        "curvature_fairness",
        "section_waviness",
        "section_waviness_fft",
        "local_plate_twist",
        "curvature_energy_robust",
        "curvature_fairness_robust",
        "developability_deviation_robust",
        "developability_deviation_positive_robust",
        "developability_deviation_negative_robust",
    )
    experimental_rows = "".join(
        "<tr>"
        f"<th>{escape(name)}</th><td>{number(result.metrics.get(name))}</td>"
        f"<td class='status'>{escape(public_status_label(_validity_for(result, name)))}</td>"
        "</tr>"
        for name in experimental_keys
        if result.metrics.get(name) is not None
    )
    experimental_section = (
        "<h3>Experimental metrics</h3>"
        "<p>Explicitly requested screened quantities; these are not part of the "
        "recommended signature.</p>"
        f"<table><thead><tr><th>Metric</th><th>Value</th><th>Status</th></tr></thead>"
        f"<tbody>{experimental_rows}</tbody></table>"
        if metadata.get("experimental_metrics_enabled") and experimental_rows
        else ""
    )

    links = "".join(
        f"<li><a href='{escape(relative)}'>{escape(relative)}</a></li>"
        for relative in sorted(set(relative_paths))
    )
    plot_paths = [
        relative
        for relative in relative_paths
        if relative.startswith("plots/") and relative.endswith(".png")
    ]
    plots = "".join(_embedded_plot(out_dir / relative, relative) for relative in plot_paths)

    reference = metadata.get("reference_length", {})
    reference_text = f"{number(result.metrics.get('length_ref'))} {escape(length_unit)}"
    reference_mode = str(reference.get("mode", "unknown"))
    thresholds = metadata.get("curvature_thresholds", {})
    retained_validity_rows = "".join(
        "<tr>"
        f"<th>{name}</th><td class='status'>{escape(public_status_label(validity.get(key)))}</td>"
        f"<td>{escape(str((validity.get(key) or {}).get('reason', 'See validity.json.')))}</td>"
        "</tr>"
        for name, _, key, _ in retained
    )
    retained_validity_rows += (
        "<tr><th>Curvature classes</th>"
        f"<td class='status'>{escape(class_status)}</td>"
        f"<td>{escape(str((validity.get('curvature_classes') or {}).get('reason', 'See validity.json.')))}</td></tr>"
    )

    notes: list[str] = []
    plausibility = reference.get("plausibility") or {}
    if plausibility.get("warning_triggered") and plausibility.get("message"):
        notes.append(f"Reference-length warning: {plausibility['message']}")
    quadrature_note = brep_quadrature_note(metadata)
    if quadrature_note:
        detail = quadrature_note["summary"]
        cells = quadrature_note["unconverged_core_cells_at_maximum_depth"]
        detail += f" Local core cells at maximum depth: {cells:,}."
        relative_change = quadrature_note["developability_one_level_relative_change"]
        if isinstance(relative_change, (int, float)):
            detail += f" Last relative change in I_D: {float(relative_change):.3g}."
        notes.append(f"BRep quadrature CAUTION: {detail}")
    if representation == "mesh":
        notes.append(
            "MESH-SENSITIVE means triangle-mesh derivative estimates may depend on "
            "resolution, connectivity, and mesh quality."
        )
    for name, _, key, _ in retained:
        record = validity.get(key) or {}
        if public_status_label(record) not in {"VALID", "VALID*", "MESH-SENSITIVE"}:
            notes.append(f"{name}: {record.get('reason', 'See validity.json.')}")
    notes_html = "".join(f"<li>{escape(note)}</li>" for note in notes)
    if not notes_html:
        notes_html = "<li>No additional numerical caution is reported.</li>"

    if representation == "brep":
        count_label = "BRep faces"
        count_value = metadata.get("n_faces", "unknown")
        representation_rows = (
            "<tr><th>Canonical metrics from native BRep</th><td>YES</td></tr>"
            "<tr><th>Display tessellation affects canonical metrics</th><td>NO</td></tr>"
        )
        representation_note = (
            "Canonical global metrics use direct BRep derivatives and trimmed-domain "
            "quadrature. Display triangles carry visualization fields only."
        )
    else:
        count_label = "Mesh vertices / triangles"
        count_value = f"{metadata.get('n_vertices', 'unknown')} / {metadata.get('n_faces', 'unknown')}"
        representation_rows = (
            "<tr><th>Curvature reconstruction</th>"
            f"<td>{escape(str((metadata.get('estimator_metadata') or {}).get('curvature_method', 'unknown')))}</td></tr>"
            "<tr><th>Representation interpretation</th><td>MESH-SENSITIVE</td></tr>"
        )
        representation_note = "The supplied triangle mesh is the assessed representation."
    total_elapsed = phase_timings.get("total_elapsed_seconds")
    provenance_rows = (
        f"<tr><th>HullProd version</th><td>{escape(str(metadata.get('hullprod_version', 'unknown')))}</td></tr>"
        f"<tr><th>Input file</th><td>{escape(str(input_geometry.get('name', input_geometry.get('path', 'unknown'))))}</td></tr>"
        f"<tr><th>SHA256</th><td class='hash'>{escape(str(input_geometry.get('sha256', 'unknown')))}</td></tr>"
        f"<tr><th>Input format</th><td>{escape(str(input_geometry.get('extension', metadata.get('source_format', 'unknown'))))}</td></tr>"
        f"<tr><th>Source unit</th><td>{escape(str(source_unit))}</td></tr>"
        f"<tr><th>Working unit</th><td>{escape(str(units.get('working_length_unit', length_unit)))}</td></tr>"
        f"<tr><th>Representation / backend</th><td>{escape(str(representation))} / {escape(str(backend))}</td></tr>"
        f"<tr><th>{escape(count_label)}</th><td>{escape(str(count_value))}</td></tr>"
        f"<tr><th>Reference length</th><td>{reference_text}</td></tr>"
        f"<tr><th>Reference-length mode</th><td>{escape(reference_mode)}</td></tr>"
        f"<tr><th>Quasi-zero factors h_f / k_f</th><td>{number(thresholds.get('h_threshold_factor'))} / {number(thresholds.get('k_threshold_factor'))}</td></tr>"
        f"<tr><th>Developability valid area fraction</th><td>{number((validity.get('developability_deviation') or {}).get('valid_area_fraction'))}</td></tr>"
        f"<tr><th>Total elapsed time</th><td>{number(total_elapsed)} s</td></tr>"
        f"{representation_rows}"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>HullProd assessment</title><style>
body{{font-family:system-ui,sans-serif;max-width:1080px;margin:1.5rem auto;padding:0 1rem;color:#17202a;line-height:1.4}}
h1{{margin-bottom:.5rem}}h2{{margin-top:1.5rem;border-bottom:2px solid #dce4e8;padding-bottom:.25rem}}
table{{border-collapse:collapse;width:100%;margin:.5rem 0 1rem;font-size:.92rem}}th,td{{padding:.38rem .5rem;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}}
.summary{{padding:.8rem 1rem;background:#eef4f7;border-left:4px solid #277da1}}.notes{{padding:.7rem 1rem;background:#fff8e8;border-left:4px solid #d59600}}.scope{{padding:.8rem 1rem;background:#f4f4f4;border-left:4px solid #666}}.status{{font-weight:700}}.hash{{overflow-wrap:anywhere}}.plots{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}}figure{{margin:.5rem 0}}img{{max-width:100%;height:auto;border:1px solid #ddd}}figcaption{{color:#444}}code{{background:#f3f3f3;padding:.1rem .25rem}}
@media(max-width:760px){{.plots{{grid-template-columns:1fr}}}}@media print{{body{{font-size:10pt;margin:.5rem}}h2{{break-after:avoid}}table,figure{{break-inside:avoid}}}}
</style></head><body><h1>HullProd assessment</h1>
<div class="summary">Geometry-based hull-surface screening with a complete machine-readable audit trail.</div>
<h2>1. Input and representation</h2>
<table><tbody>
<tr><th>Input</th><td>{escape(str(input_geometry.get("name", input_geometry.get("path", "unknown"))))}</td></tr>
<tr><th>Representation / backend</th><td>{escape(str(representation))} / {escape(str(backend))}</td></tr>
<tr><th>Source / working unit</th><td>{escape(str(source_unit))} / {escape(length_unit)}</td></tr>
<tr><th>Reference length</th><td>{reference_text} ({escape(reference_mode)})</td></tr>
</tbody></table><p>{escape(representation_note)}</p>
<h2>2. Recommended signature</h2>
<table><thead><tr><th>Component</th><th>Meaning</th><th>Value</th><th>Scientific status</th></tr></thead><tbody>{signature_rows}</tbody></table>
{experimental_section}
<h2>3. Curvature-class composition</h2>
<table><thead><tr><th>Component</th><th>Valid-area fraction</th><th>Scientific status</th></tr></thead><tbody>{class_rows}</tbody></table>
<h2>4. Standard plots</h2><div class="plots">{plots or "<p>No plots were requested or no display geometry was available.</p>"}</div>
<h2>5. Validity and numerical notes</h2>
<table><thead><tr><th>Quantity</th><th>Scientific status</th><th>Reason</th></tr></thead><tbody>{retained_validity_rows}</tbody></table>
<div class="notes"><ul>{notes_html}</ul></div>
<p><a href="validity.json">Full validity record: validity.json</a></p>
<h2>6. Concise provenance</h2><table><tbody>{provenance_rows}</tbody></table>
<p><a href="provenance.json">Full provenance: provenance.json</a><br>
<a href="metrics.json">Full machine-readable metrics: metrics.json</a></p>
<h2>7. Output files</h2><ul>{links}</ul>
<h2>8. Scope and limitations</h2>
<div class="scope">These metrics are geometry-based screening descriptors. They are not direct fabrication-cost, man-hour, forming-route, or schedule predictions.</div>
</body></html>"""


def _flatten_metadata(values, prefix=""):
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _flatten_metadata(value, name)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            yield name, value


def _summary_text(result, out_dir: Path) -> str:
    signature = result.signature
    reference = result.metadata.get("reference_length", {})
    input_geometry = result.metadata.get("input_geometry", {})
    def number(value) -> str:
        return "--" if value is None else f"{float(value):.12g}"

    lines = [
        f"HullProd {result.metadata.get('hullprod_version', 'unknown')}",
        f"Input: {input_geometry.get('name', input_geometry.get('path', 'unknown'))}",
        f"Backend: {result.metadata.get('backend', 'unknown')}",
        f"Representation: {result.metadata.get('representation', 'unknown')}",
        (
            f"Reference length: {float(result.metrics['length_ref']):.12g} "
            f"({reference.get('mode', 'unknown')}; {reference.get('method', 'unknown')})"
        ),
        "",
        "Recommended signature",
        f"  I_D       {number(signature['I_D'])}",
        f"  I_D_plus  {number(signature['I_D_plus'])}",
        f"  I_D_minus {number(signature['I_D_minus'])}",
    ]
    lines.extend(
        f"  a_C_{name:<8} {number(value)}" for name, value in signature["a_C"].items()
    )
    lines.extend(["", f"Results written to: {out_dir.resolve()}"])
    return "\n".join(lines) + "\n"


def write_outputs(result, out_dir: str | Path) -> dict[str, str]:
    """Write the stable compact 1.0 result package and compatibility JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_geometry = dict(result.metadata.get("input_geometry", {}))
    input_path = Path(input_geometry.get("path", "input"))
    input_geometry.setdefault("name", input_path.name)
    input_geometry.setdefault("extension", input_path.suffix.lower())
    validity = result.metadata.get("metric_validity", {})
    recommended_validity = {
        "I_D": validity.get("developability_deviation"),
        "I_D_plus": validity.get("developability_deviation_positive"),
        "I_D_minus": validity.get("developability_deviation_negative"),
        "a_C": validity.get("curvature_classes"),
    }
    experimental_keys = (
        "curvature_energy",
        "curvature_fairness",
        "section_waviness",
        "section_waviness_fft",
        "curvature_energy_robust",
        "curvature_fairness_robust",
        "developability_deviation_robust",
        "developability_deviation_positive_robust",
        "developability_deviation_negative_robust",
    )
    experimental = {
        key: result.metrics.get(key)
        for key in experimental_keys
        if result.metrics.get(key) is not None
    }
    signature_payload = {
        "schema_version": result.metadata.get("schema_version"),
        "hullprod_version": result.metadata.get("hullprod_version"),
        "input": input_geometry,
        "backend": result.metadata.get("backend"),
        "representation": result.metadata.get("representation"),
        "reference_length": result.metadata.get("reference_length"),
        "recommended_signature": result.signature,
        "validity": recommended_validity,
        "auxiliary": {
            "developability_area_ratio": result.metrics.get("developability_area_ratio")
        },
        "diagnostics": {
            "mesh_quality": result.metadata.get("mesh_quality"),
            "local_plate_twist": result.metrics.get("local_plate_twist"),
        },
        "experimental": experimental,
    }
    (out_dir / "signature.json").write_text(
        json.dumps(_json_ready(signature_payload), indent=2, allow_nan=False), encoding="utf-8"
    )
    # Compatibility artifact: raw 0.x/early-1.0 metric keys retain their meanings.
    (out_dir / "metrics.json").write_text(
        json.dumps(result.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
    )

    provenance = {
        "schema_version": result.metadata.get("schema_version"),
        "hullprod_version": result.metadata.get("hullprod_version"),
        "run_timestamp_utc": result.metadata.get("run_timestamp_utc"),
        "field_schema_version": result.metadata.get("field_schema_version"),
        "output_layout_version": result.metadata.get("output_layout_version"),
        "representation": result.metadata.get("representation"),
        "backend": result.metadata.get("backend"),
        "backend_readiness": result.metadata.get("backend_readiness"),
        "input_geometry": result.metadata.get("input_geometry"),
        "source_format": result.metadata.get("source_format"),
        "reference_length": result.metadata.get("reference_length"),
        "brep_import": result.metadata.get("brep_import"),
        "estimator_metadata": result.metadata.get("estimator_metadata"),
        "curvature_thresholds": result.metadata.get("curvature_thresholds"),
        "quadrature": result.metadata.get("quadrature"),
        "section_settings": result.metadata.get("section_settings"),
        "display_mesh": result.metadata.get("display_mesh"),
        "units": result.metadata.get("units"),
        "mesh_quality": result.metadata.get("mesh_quality"),
        "continuity": result.metadata.get("continuity"),
        "warnings": result.metadata.get("warnings", []),
        "phase_timings": result.metadata.get("phase_timings"),
        "canonical_metrics_depend_on_display_mesh": result.metadata.get(
            "canonical_metrics_depend_on_display_mesh"
        ),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(_json_ready(provenance), indent=2, allow_nan=False), encoding="utf-8"
    )

    signature = result.signature
    csv_row = {
        "case": input_path.stem,
        "backend": result.metadata.get("backend"),
        "reference_length": result.metrics.get("length_ref"),
        "I_D": signature["I_D"],
        "I_D_plus": signature["I_D_plus"],
        "I_D_minus": signature["I_D_minus"],
        "a_C_flat": signature["a_C"]["flat"],
        "a_C_single": signature["a_C"]["single"],
        "a_C_elliptic": signature["a_C"]["elliptic"],
        "a_C_saddle": signature["a_C"]["saddle"],
        "developability_status": (recommended_validity["I_D"] or {}).get("status"),
        "curvature_class_status": (recommended_validity["a_C"] or {}).get("status"),
        "valid_area_fraction": (recommended_validity["I_D"] or {}).get(
            "valid_area_fraction"
        ),
    }
    pd.DataFrame([csv_row]).to_csv(out_dir / "signature.csv", index=False)
    (out_dir / "validity.json").write_text(
        json.dumps(_json_ready(validity), indent=2, allow_nan=False), encoding="utf-8"
    )
    (out_dir / "summary.txt").write_text(_summary_text(result, out_dir), encoding="utf-8")

    field_exports = [
        relative
        for relative in (
            "fields/surface_fields.vtp",
            "fields/surface_fields.csv",
        )
        if (out_dir / relative).is_file()
    ]
    field_manifest = build_field_manifest(result, exported_files=field_exports)
    (out_dir / "field_manifest.json").write_text(
        json.dumps(field_manifest, indent=2, allow_nan=False), encoding="utf-8"
    )

    relative_paths = [
        str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file()
    ]
    relative_paths.extend(["report.html", "output_manifest.json"])
    (out_dir / "report.html").write_text(
        _html_report(result, out_dir, relative_paths, field_manifest), encoding="utf-8"
    )
    manifest = build_output_manifest(out_dir, relative_paths)
    (out_dir / "output_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    paths = {
        "result_directory": str(out_dir.resolve()),
        "signature_json": str((out_dir / "signature.json").resolve()),
        "signature_csv": str((out_dir / "signature.csv").resolve()),
        "validity_json": str((out_dir / "validity.json").resolve()),
        "provenance_json": str((out_dir / "provenance.json").resolve()),
        "summary": str((out_dir / "summary.txt").resolve()),
        "report_html": str((out_dir / "report.html").resolve()),
    }
    result.metadata.setdefault("output_paths", {}).update(paths)
    return paths
