from __future__ import annotations

import base64
import csv
import re
import struct
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import trimesh

_PUBLIC_FIELD_ALIASES = {
    "developability_density": "developability_density",
    "developability_positive": "developability_positive_density",
    "developability_negative": "developability_negative_density",
    "curvature_class_id": "curvature_class_id",
    "valid_mask": "K_valid",
    "H": "H",
    "K": "K",
}


def _field_name(name: str) -> str:
    """Return a VTK/Tecplot-safe field name."""
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip())
    if not cleaned:
        return "field"
    if cleaned[0].isdigit():
        cleaned = f"field_{cleaned}"
    return cleaned


def _numeric_array(values: np.ndarray) -> np.ndarray:
    """Return a one-dimensional numeric array without erasing categorical type."""
    arr = np.asarray(values).ravel()
    if arr.dtype.kind == "b":
        return arr.astype(np.uint8)
    if arr.dtype.kind in {"i", "u"}:
        return arr
    arr = np.asarray(arr, dtype=np.float64)
    return np.where(np.isfinite(arr), arr, np.nan)


def _split_fields(
    mesh: trimesh.Trimesh,
    local_fields: dict[str, np.ndarray],
    associations: dict[str, str] | None = None,
):
    """Split local fields into point-data and cell-data dictionaries."""
    n_vertices = len(mesh.vertices)
    n_faces = len(mesh.faces)

    point_data = {}
    cell_data = {}

    for name, values in local_fields.items():
        arr = _numeric_array(values)
        safe_name = _field_name(name)
        declared = (associations or {}).get(name)

        if declared == "point" or (declared is None and len(arr) == n_vertices):
            if len(arr) != n_vertices:
                raise ValueError(
                    f"Point field {name!r} has {len(arr)} tuples, expected {n_vertices}"
                )
            point_data[safe_name] = arr
        elif declared == "cell" or (declared is None and len(arr) == n_faces):
            if len(arr) != n_faces:
                raise ValueError(f"Cell field {name!r} has {len(arr)} tuples, expected {n_faces}")
            cell_data[safe_name] = arr

    return point_data, cell_data


def _legacy_vtk_type(values: np.ndarray) -> str:
    if values.dtype.kind == "u" and values.dtype.itemsize == 1:
        return "unsigned_char"
    if values.dtype.kind in {"i", "u"}:
        return "int"
    return "double"


def _vtk_xml_type(values: np.ndarray) -> str:
    if values.dtype.kind == "u":
        return f"UInt{8 * values.dtype.itemsize}"
    if values.dtype.kind == "i":
        return f"Int{8 * values.dtype.itemsize}"
    return "Float64"


def _format_scalar(value: float | int | np.generic) -> str:
    if np.issubdtype(np.asarray(value).dtype, np.integer):
        return str(int(value))
    if np.isnan(float(value)):
        return "NaN"
    if np.isposinf(float(value)):
        return "Inf"
    if np.isneginf(float(value)):
        return "-Inf"
    return f"{float(value):.16e}"


def write_vtk_fields(
    mesh: trimesh.Trimesh,
    local_fields: dict[str, np.ndarray],
    out_path: str | Path,
    *,
    associations: dict[str, str] | None = None,
) -> None:
    """Write HullProd local fields to legacy binary VTK POLYDATA."""
    out_path = Path(out_path)
    point_data, cell_data = _split_fields(mesh, local_fields, associations)

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)

    def line(stream, value: str) -> None:
        stream.write(value.encode("ascii") + b"\n")

    def field_bytes(values: np.ndarray) -> bytes:
        if values.dtype.kind == "u" and values.dtype.itemsize == 1:
            return np.asarray(values, dtype=">u1").tobytes()
        if values.dtype.kind in {"i", "u"}:
            return np.asarray(values, dtype=">i4").tobytes()
        return np.asarray(values, dtype=">f8").tobytes()

    with out_path.open("wb") as f:
        line(f, "# vtk DataFile Version 3.0")
        line(f, "HullProd local fields")
        line(f, "BINARY")
        line(f, "DATASET POLYDATA")

        line(f, f"POINTS {len(vertices)} double")
        f.write(np.asarray(vertices, dtype=">f8").tobytes() + b"\n")

        line(f, f"POLYGONS {len(faces)} {4 * len(faces)}")
        polygon_records = np.column_stack(
            (np.full(len(faces), 3, dtype=np.int32), faces.astype(np.int32))
        )
        f.write(np.asarray(polygon_records, dtype=">i4").tobytes() + b"\n")

        if point_data:
            line(f, f"POINT_DATA {len(vertices)}")
            for name, values in point_data.items():
                line(f, f"SCALARS {name} {_legacy_vtk_type(values)} 1")
                line(f, "LOOKUP_TABLE default")
                f.write(field_bytes(values) + b"\n")

        if cell_data:
            line(f, f"CELL_DATA {len(faces)}")
            for name, values in cell_data.items():
                line(f, f"SCALARS {name} {_legacy_vtk_type(values)} 1")
                line(f, "LOOKUP_TABLE default")
                f.write(field_bytes(values) + b"\n")


def _xml_data_array(
    parent: ET.Element,
    values: np.ndarray,
    *,
    name: str | None = None,
    components: int | None = None,
) -> None:
    attributes = {"type": _vtk_xml_type(values), "format": "binary"}
    if name is not None:
        attributes["Name"] = name
    if components is not None:
        attributes["NumberOfComponents"] = str(components)
    element = ET.SubElement(parent, "DataArray", attributes)
    little_endian = np.asarray(values).astype(np.asarray(values).dtype.newbyteorder("<"))
    payload = little_endian.tobytes(order="C")
    element.text = base64.b64encode(struct.pack("<Q", len(payload)) + payload).decode("ascii")


def write_vtp_fields(
    mesh: trimesh.Trimesh,
    local_fields: dict[str, np.ndarray],
    out_path: str | Path,
    *,
    associations: dict[str, str] | None = None,
) -> None:
    """Write deterministic ASCII VTK XML PolyData without a VTK runtime."""
    out_path = Path(out_path)
    point_data, cell_data = _split_fields(mesh, local_fields, associations)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    root = ET.Element(
        "VTKFile",
        {
            "type": "PolyData",
            "version": "1.0",
            "byte_order": "LittleEndian",
            "header_type": "UInt64",
        },
    )
    polydata = ET.SubElement(root, "PolyData")
    piece = ET.SubElement(
        polydata,
        "Piece",
        {
            "NumberOfPoints": str(len(vertices)),
            "NumberOfVerts": "0",
            "NumberOfLines": "0",
            "NumberOfStrips": "0",
            "NumberOfPolys": str(len(faces)),
        },
    )
    point_element = ET.SubElement(piece, "PointData")
    for name, values in point_data.items():
        _xml_data_array(point_element, values, name=name)
    cell_element = ET.SubElement(piece, "CellData")
    for name, values in cell_data.items():
        _xml_data_array(cell_element, values, name=name)
    points = ET.SubElement(piece, "Points")
    _xml_data_array(points, vertices, components=3)
    polys = ET.SubElement(piece, "Polys")
    _xml_data_array(polys, faces.ravel(), name="connectivity")
    _xml_data_array(
        polys,
        3 * np.arange(1, len(faces) + 1, dtype=np.int64),
        name="offsets",
    )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)


def write_tecplot_fields(
    mesh: trimesh.Trimesh,
    local_fields: dict[str, np.ndarray],
    out_path: str | Path,
    *,
    associations: dict[str, str] | None = None,
) -> None:
    """Write nodal and cell-centered fields to Tecplot FE-triangle ASCII."""
    out_path = Path(out_path)
    point_data, cell_data = _split_fields(mesh, local_fields, associations)

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    point_names = list(point_data)
    cell_names = list(cell_data)

    with out_path.open("w") as f:
        f.write('TITLE = "HullProd local fields"\n')
        variables = (
            ['"x"', '"y"', '"z"']
            + [f'"{name}"' for name in point_names]
            + [f'"{name}"' for name in cell_names]
        )
        f.write("VARIABLES = " + " ".join(variables) + "\n")
        zone = (
            f'ZONE T="HullProd", N={len(vertices)}, E={len(faces)}, '
            "DATAPACKING=BLOCK, ZONETYPE=FETRIANGLE"
        )
        if cell_names:
            first_cell_variable = 4 + len(point_names)
            last_cell_variable = first_cell_variable + len(cell_names) - 1
            zone += f", VARLOCATION=([{first_cell_variable}-{last_cell_variable}]=CELLCENTERED)"
        f.write(zone + "\n")

        blocks = [vertices[:, axis] for axis in range(3)]
        blocks.extend(point_data[name] for name in point_names)
        blocks.extend(cell_data[name] for name in cell_names)
        for block in blocks:
            values = [_format_scalar(value) for value in block]
            for start in range(0, len(values), 6):
                f.write(" ".join(values[start : start + 6]) + "\n")

        for i, j, k in faces:
            f.write(f"{i + 1:d} {j + 1:d} {k + 1:d}\n")


def public_surface_fields(result) -> dict[str, np.ndarray]:
    """Return the stable field names exported by a default 1.0 assessment."""
    fields: dict[str, np.ndarray] = {}
    for public_name, internal_name in _PUBLIC_FIELD_ALIASES.items():
        if internal_name in result.local_fields:
            fields[public_name] = np.asarray(result.local_fields[internal_name])
    return fields


def write_surface_fields_csv(
    mesh: trimesh.Trimesh,
    local_fields: dict[str, np.ndarray],
    out_path: str | Path,
) -> None:
    """Write point-associated fields with coordinates to a portable CSV."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    point_fields = {
        name: _numeric_array(values)
        for name, values in local_fields.items()
        if len(np.asarray(values).ravel()) == len(vertices)
    }
    columns = ["point_id", "x", "y", "z", *point_fields]
    with Path(out_path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for index, coordinates in enumerate(vertices):
            writer.writerow(
                [index, *(_format_scalar(value) for value in coordinates)]
                + [_format_scalar(point_fields[name][index]) for name in point_fields]
            )


def export_visualization_fields(mesh, result, out_dir: str | Path) -> dict[str, str]:
    """Export local HullProd fields for external visualization tools."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    local_fields = public_surface_fields(result)
    associations = {name: "point" for name in local_fields}

    required = {
        "developability_density",
        "developability_positive",
        "developability_negative",
        "curvature_class_id",
        "valid_mask",
    }
    missing = sorted(required - set(local_fields))
    if missing:
        raise ValueError(f"Required distributed field(s) unavailable: {', '.join(missing)}")

    write_vtp_fields(
        mesh,
        local_fields,
        out_dir / "surface_fields.vtp",
        associations=associations,
    )
    write_surface_fields_csv(mesh, local_fields, out_dir / "surface_fields.csv")
    return {
        "surface_fields_vtp": str((out_dir / "surface_fields.vtp").resolve()),
        "surface_fields_csv": str((out_dir / "surface_fields.csv").resolve()),
    }
