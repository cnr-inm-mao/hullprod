"""Human-readable coordinate-unit provenance without numerical conversion."""

from __future__ import annotations

from typing import Any

_DECLARED_UNIT_NAMES = {
    "M": "metre",
    "METRE": "metre",
    "METER": "metre",
    "MM": "millimetre",
    "MILLIMETRE": "millimetre",
    "MILLIMETER": "millimetre",
    "CM": "centimetre",
    "CENTIMETRE": "centimetre",
    "CENTIMETER": "centimetre",
    "IN": "inch",
    "INCH": "inch",
    "FT": "foot",
    "FOOT": "foot",
}


def _declared_unit_name(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return _DECLARED_UNIT_NAMES.get(text.upper(), text)


def mesh_unit_metadata() -> dict[str, Any]:
    """Return honest labels for meshes without physical-unit metadata."""
    return {
        "source_declared_unit": None,
        "source_declared_unit_raw": None,
        "working_length_unit": "input-length units",
        "working_length_unit_symbol": "input-length units",
        "working_area_unit": "input-length units squared",
        "working_area_unit_symbol": "input-length units²",
        "unit_provenance": "not_declared_by_mesh_format",
        "numerical_values_rescaled": False,
        "note": "Mesh coordinates retain the input file's unspecified length unit.",
    }


def brep_unit_metadata(import_metadata: dict[str, Any]) -> dict[str, Any]:
    """Label native CAD working coordinates while preserving source declarations."""
    source = dict(import_metadata.get("source_units") or {})
    working = str(source.get("working_unit") or "model_coordinate_unit")
    if working == "millimetre":
        working_length = "millimetre"
        working_symbol = "mm"
        working_area = "square millimetres"
        working_area_symbol = "mm²"
    else:
        working_length = "model-coordinate units"
        working_symbol = "model-coordinate units"
        working_area = "model-coordinate units squared"
        working_area_symbol = "model-coordinate units²"
    declared_raw = source.get("declared_name")
    result = {
        "source_declared_unit": _declared_unit_name(declared_raw),
        "source_declared_unit_raw": declared_raw,
        "source_declared_unit_flag": source.get("declared_flag"),
        "source_declared_unit_value_mm": source.get("declared_value_mm"),
        "working_length_unit": working_length,
        "working_length_unit_symbol": working_symbol,
        "working_area_unit": working_area,
        "working_area_unit_symbol": working_area_symbol,
        "unit_provenance": "native_cad_import_metadata",
        "numerical_values_rescaled": False,
    }
    if source.get("warning"):
        result["warning"] = source["warning"]
    if source.get("note"):
        result["note"] = source["note"]
    return result
