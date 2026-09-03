"""Parser-based round-trip gates for stable 1.0 field transports."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import trimesh
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from hullprod.export import write_tecplot_fields, write_vtk_fields, write_vtp_fields


@pytest.fixture
def field_case():
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    fields = {
        "point_float": np.array([0.0, np.nan, 2.0, 3.0]),
        "K_valid": np.array([1, 0, 1, 1], dtype=np.uint8),
        "cell_float": np.array([10.0, np.nan]),
        "curvature_class_id": np.array([-1, 3], dtype=np.int8),
    }
    return mesh, fields


def _vtk_arrays(polydata, association: str) -> dict[str, np.ndarray]:
    attributes = polydata.GetPointData() if association == "point" else polydata.GetCellData()
    return {
        attributes.GetArrayName(index): vtk_to_numpy(attributes.GetArray(index))
        for index in range(attributes.GetNumberOfArrays())
    }


def _assert_polydata(polydata) -> None:
    assert polydata.GetNumberOfPoints() == 4
    assert polydata.GetNumberOfPolys() == 2
    point = _vtk_arrays(polydata, "point")
    cell = _vtk_arrays(polydata, "cell")
    assert set(point) == {"point_float", "K_valid"}
    assert set(cell) == {"cell_float", "curvature_class_id"}
    np.testing.assert_allclose(point["point_float"][[0, 2, 3]], [0.0, 2.0, 3.0])
    assert np.isnan(point["point_float"][1])
    np.testing.assert_array_equal(point["K_valid"], [1, 0, 1, 1])
    assert np.isnan(cell["cell_float"][1])
    np.testing.assert_array_equal(cell["curvature_class_id"], [-1, 3])
    connectivity = vtk_to_numpy(polydata.GetPolys().GetConnectivityArray())
    np.testing.assert_array_equal(connectivity, [0, 1, 2, 0, 2, 3])


def test_vtp_parser_round_trip_preserves_fields_types_and_connectivity(
    tmp_path: Path, field_case
) -> None:
    mesh, fields = field_case
    output = tmp_path / "fields.vtp"
    write_vtp_fields(mesh, fields, output)
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(output))
    reader.Update()
    assert reader.GetErrorCode() == 0
    _assert_polydata(reader.GetOutput())
    assert reader.GetOutput().GetPointData().GetArray("K_valid").GetDataTypeAsString() == (
        "unsigned char"
    )
    assert (
        reader.GetOutput().GetCellData().GetArray("curvature_class_id").GetDataTypeAsString()
        == "signed char"
    )


def test_legacy_vtk_parser_round_trip_preserves_association_and_connectivity(
    tmp_path: Path, field_case
) -> None:
    mesh, fields = field_case
    output = tmp_path / "fields.vtk"
    write_vtk_fields(mesh, fields, output)
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(output))
    reader.ReadAllScalarsOn()
    reader.Update()
    assert reader.GetErrorCode() == 0
    _assert_polydata(reader.GetOutput())


def _parse_tecplot(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    variables = re.findall(r'"([^"]+)"', lines[1])
    zone = lines[2]
    node_count = int(re.search(r"\bN=(\d+)", zone).group(1))
    element_count = int(re.search(r"\bE=(\d+)", zone).group(1))
    match = re.search(r"VARLOCATION=\(\[(\d+)-(\d+)\]=CELLCENTERED\)", zone)
    assert match is not None
    first_cell, last_cell = (int(value) for value in match.groups())
    counts = [
        element_count if first_cell <= index <= last_cell else node_count
        for index in range(1, len(variables) + 1)
    ]
    tokens = " ".join(lines[3:]).split()
    values = {}
    cursor = 0
    for name, count in zip(variables, counts, strict=True):
        values[name] = np.asarray(tokens[cursor : cursor + count], dtype=float)
        cursor += count
    connectivity = np.asarray(tokens[cursor:], dtype=int).reshape(element_count, 3)
    return zone, variables, values, connectivity


def test_tecplot_block_export_uses_explicit_cell_centering(tmp_path: Path, field_case) -> None:
    mesh, fields = field_case
    output = tmp_path / "fields.dat"
    write_tecplot_fields(mesh, fields, output)
    zone, variables, values, connectivity = _parse_tecplot(output)
    assert "DATAPACKING=BLOCK" in zone
    assert "VARLOCATION=([6-7]=CELLCENTERED)" in zone
    assert variables == [
        "x",
        "y",
        "z",
        "point_float",
        "K_valid",
        "cell_float",
        "curvature_class_id",
    ]
    assert len(values["point_float"]) == 4
    assert len(values["cell_float"]) == 2
    assert np.isnan(values["point_float"][1])
    assert np.isnan(values["cell_float"][1])
    np.testing.assert_array_equal(values["curvature_class_id"], [-1, 3])
    np.testing.assert_array_equal(connectivity, [[1, 2, 3], [1, 3, 4]])
    assert not any(name.endswith("_vertex_average") for name in variables)
