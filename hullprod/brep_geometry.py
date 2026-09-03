"""Native OpenCascade BRep import and differential geometry.

This module is imported only when the native CAD backend is selected. Canonical BRep
metrics are evaluated from parametric surface derivatives; triangulations are
never used by the routines in this module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np


class BRepDependencyError(ImportError):
    """Raised when native CAD support is requested without OpenCascade."""


class BRepEvaluationError(RuntimeError):
    """Raised when a CAD entity cannot be evaluated consistently."""


@dataclass(frozen=True)
class BRepModel:
    """An unchanged source BRep plus import and topology provenance."""

    path: Path | None
    shape: Any
    faces: tuple[Any, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SurfaceDifferential:
    """Differential properties evaluated directly on one BRep face."""

    position: np.ndarray
    normal: np.ndarray
    mean_curvature: float
    gaussian_curvature: float
    mean_u: float | None
    mean_v: float | None
    gradient_mean_squared: float | None
    jacobian: float
    metric_determinant: float
    metric_condition: float
    lprop_mean_curvature: float
    lprop_gaussian_curvature: float
    orientation_reversed: bool


def require_ocp() -> None:
    """Fail clearly if a damaged/incomplete installation lacks OpenCascade."""
    try:
        import OCP  # noqa: F401
    except ImportError as error:  # pragma: no cover - environment dependent
        raise BRepDependencyError(
            "Native IGES/STEP support is part of the standard HullProd installation, "
            "but the OCP module is unavailable. Reinstall HullProd in a supported "
            "binary-wheel environment."
        ) from error


def ocp_runtime_metadata() -> dict[str, str | None]:
    """Return installed OpenCascade wrapper provenance without importing it eagerly."""
    try:
        wrapper_version = version("cadquery-ocp")
    except PackageNotFoundError:  # pragma: no cover - damaged installation
        wrapper_version = None
    return {
        "python_module": "OCP",
        "distribution": "cadquery-ocp",
        "distribution_version": wrapper_version,
    }


def sha256_file(path: str | Path) -> str:
    """Return a source-file digest without modifying the CAD master."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _topology_count(shape: Any, shape_type: Any) -> int:
    from OCP.TopExp import TopExp_Explorer

    explorer = TopExp_Explorer(shape, shape_type)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _faces(shape: Any) -> tuple[Any, ...]:
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    values = []
    while explorer.More():
        values.append(TopoDS.Face_s(explorer.Current()))
        explorer.Next()
    return tuple(values)


def _compound_from_roots(reader: Any, selected: tuple[int, ...]) -> tuple[Any, list[int]]:
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    transferred = []
    for root_index in selected:
        before = reader.NbShapes()
        if not reader.TransferOneRoot(root_index):
            continue
        after = reader.NbShapes()
        if after <= before:
            continue
        builder.Add(compound, reader.Shape(after))
        transferred.append(root_index)
    return compound, transferred


def _iges_unit_metadata(reader: Any) -> dict[str, Any]:
    try:
        section = reader.IGESModel().GlobalSection()
        return {
            "declared_name": section.UnitName().ToCString(),
            "declared_flag": int(section.UnitFlag()),
            "declared_value_mm": float(section.UnitValue()),
            "model_scale": float(section.Scale()),
            "working_unit": "millimetre",
        }
    except Exception as error:  # pragma: no cover - malformed file metadata
        return {
            "declared_name": None,
            "working_unit": "millimetre",
            "warning": f"IGES unit metadata unavailable: {error}",
        }


def _shape_bounds(shape: Any) -> list[list[float]]:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box, True)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return [[xmin, ymin, zmin], [xmax, ymax, zmax]]


def independent_surface_area(shape: Any, tolerance: float = 1.0e-9) -> tuple[float, float]:
    """Return exact-geometry OpenCascade area and its adaptive error estimate."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    properties = GProp_GProps()
    estimated_error = BRepGProp.SurfaceProperties_s(
        shape,
        properties,
        float(tolerance),
        False,
    )
    return float(properties.Mass()), float(estimated_error)


def model_from_shape(shape: Any, *, label: str = "generated") -> BRepModel:
    """Wrap a programmatically generated OpenCascade shape for testing."""
    require_ocp()
    from OCP.TopAbs import TopAbs_COMPOUND, TopAbs_SHELL, TopAbs_SOLID

    faces = _faces(shape)
    if not faces:
        raise BRepEvaluationError("BRep shape contains no faces")
    area, area_error = independent_surface_area(shape)
    return BRepModel(
        path=None,
        shape=shape,
        faces=faces,
        metadata={
            "source_path": label,
            "source_sha256": None,
            "source_format": "generated_brep",
            "root_count": 1,
            "selected_root_indices": [1],
            "transferred_root_indices": [1],
            "face_count": len(faces),
            "shell_count": _topology_count(shape, TopAbs_SHELL),
            "solid_count": _topology_count(shape, TopAbs_SOLID),
            "compound_count": _topology_count(shape, TopAbs_COMPOUND),
            "bounds": _shape_bounds(shape),
            "source_units": {"working_unit": "model_coordinate_unit"},
            "independent_surface_area": area,
            "independent_surface_area_estimated_error": area_error,
            "import_warnings": [],
            "cad_kernel": ocp_runtime_metadata(),
        },
    )


def load_brep(
    path: str | Path,
    *,
    root_indices: tuple[int, ...] | list[int] | None = None,
) -> BRepModel:
    """Read IGES/STEP without healing, sewing, tessellating, or editing it."""
    require_ocp()
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from OCP.TopAbs import TopAbs_COMPOUND, TopAbs_SHELL, TopAbs_SOLID

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Geometry file not found: {path}")
    extension = path.suffix.lower()
    Interface_Static.SetCVal_s("xstep.cascade.unit", "MM")
    warnings: list[str] = []

    if extension in {".igs", ".iges"}:
        from OCP.IGESControl import IGESControl_Reader

        reader = IGESControl_Reader()
        source_format = "iges"
    elif extension in {".stp", ".step"}:
        from OCP.STEPControl import STEPControl_Reader

        reader = STEPControl_Reader()
        source_format = "step"
    else:
        raise ValueError(f"Unsupported BRep format: {extension}")

    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise BRepEvaluationError(f"OpenCascade could not read {path}")
    root_count = int(reader.NbRootsForTransfer())
    if root_count <= 0:
        raise BRepEvaluationError(f"OpenCascade found no transferable roots in {path}")
    selected = (
        tuple(range(1, root_count + 1))
        if root_indices is None
        else tuple(int(value) for value in root_indices)
    )
    invalid = [value for value in selected if value < 1 or value > root_count]
    if invalid:
        raise ValueError(f"BRep root indices out of range 1..{root_count}: {invalid}")
    shape, transferred = _compound_from_roots(reader, selected)
    if not transferred or shape.IsNull():
        raise BRepEvaluationError(f"No selected roots transferred from {path}")
    if len(transferred) != len(selected):
        warnings.append(f"Transferred {len(transferred)} of {len(selected)} requested roots")
    faces = _faces(shape)
    if not faces:
        raise BRepEvaluationError(f"Transferred BRep contains no faces: {path}")
    area, area_error = independent_surface_area(shape)
    units = (
        _iges_unit_metadata(reader)
        if source_format == "iges"
        else {
            "working_unit": "millimetre",
            "note": "STEP declarations are converted to the OpenCascade millimetre system unit",
        }
    )
    metadata = {
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "source_format": source_format,
        "root_count": root_count,
        "selected_root_indices": list(selected),
        "transferred_root_indices": transferred,
        "face_count": len(faces),
        "shell_count": _topology_count(shape, TopAbs_SHELL),
        "solid_count": _topology_count(shape, TopAbs_SOLID),
        "compound_count": _topology_count(shape, TopAbs_COMPOUND),
        "bounds": _shape_bounds(shape),
        "source_units": units,
        "independent_surface_area": area,
        "independent_surface_area_estimated_error": area_error,
        "import_warnings": warnings,
        "cad_kernel": ocp_runtime_metadata(),
        "generic_selection_policy": (
            "all_transferable_roots" if root_indices is None else "explicit_root_indices"
        ),
    }
    return BRepModel(path=path, shape=shape, faces=faces, metadata=metadata)


def _array(vector: Any) -> np.ndarray:
    return np.array([vector.X(), vector.Y(), vector.Z()], dtype=float)


def evaluate_surface_jacobian(
    face: Any,
    u: float,
    v: float,
    *,
    surface: Any | None = None,
) -> tuple[np.ndarray, float]:
    """Return position and exact parametric area Jacobian from BRep D1."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.gp import gp_Pnt, gp_Vec

    if surface is None:
        surface = BRepAdaptor_Surface(face, True)
    point = gp_Pnt()
    su = gp_Vec()
    sv = gp_Vec()
    surface.D1(float(u), float(v), point, su, sv)
    jacobian = float(np.linalg.norm(np.cross(_array(su), _array(sv))))
    if not np.isfinite(jacobian):
        raise BRepEvaluationError("non-finite surface Jacobian")
    return np.array([point.X(), point.Y(), point.Z()], dtype=float), jacobian


def evaluate_surface_differential(
    face: Any,
    u: float,
    v: float,
    *,
    need_third: bool = True,
    resolution: float = 1.0e-10,
    surface: Any | None = None,
    crosscheck_lprop: bool = True,
) -> SurfaceDifferential:
    """Evaluate H, K, and optionally ``|grad_s H|^2`` from BRep derivatives.

    ``BRepAdaptor_Surface.D2/D3`` supplies derivatives in the face's parameter
    coordinates.  The formula normal agrees with ``BRepLProp_SLProps`` before
    applying the topological face-orientation sign.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepLProp import BRepLProp_SLProps
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.TopAbs import TopAbs_REVERSED

    if surface is None:
        surface = BRepAdaptor_Surface(face, True)
    point = gp_Pnt()
    derivatives = [gp_Vec() for _ in range(9)]
    if need_third:
        surface.D3(float(u), float(v), point, *derivatives)
    else:
        surface.D2(float(u), float(v), point, *derivatives[:5])
    su, sv, suu, svv, suv = map(_array, derivatives[:5])
    q = np.cross(su, sv)
    w = float(np.linalg.norm(q))
    if not np.isfinite(w) or w <= resolution:
        raise BRepEvaluationError("singular surface parameterization")
    base_normal = q / w
    orientation_reversed = face.Orientation() == TopAbs_REVERSED
    orientation = -1.0 if orientation_reversed else 1.0
    normal = orientation * base_normal

    e_first = float(su @ su)
    f_first = float(su @ sv)
    g_first = float(sv @ sv)
    determinant = e_first * g_first - f_first * f_first
    if not np.isfinite(determinant) or determinant <= resolution * resolution:
        raise BRepEvaluationError("singular first fundamental form")
    trace = e_first + g_first
    discriminant = max(trace * trace - 4.0 * determinant, 0.0) ** 0.5
    lambda_max = 0.5 * (trace + discriminant)
    lambda_min = 0.5 * (trace - discriminant)
    metric_condition = lambda_max / max(lambda_min, resolution * resolution)
    e_second = float(normal @ suu)
    f_second = float(normal @ suv)
    g_second = float(normal @ svv)
    numerator = e_second * g_first - 2.0 * f_second * f_first + g_second * e_first
    mean = numerator / (2.0 * determinant)
    gaussian = (e_second * g_second - f_second * f_second) / determinant

    lprop_mean = lprop_gaussian = float("nan")
    if crosscheck_lprop:
        properties = BRepLProp_SLProps(
            surface,
            float(u),
            float(v),
            2,
            float(resolution),
        )
        if not properties.IsCurvatureDefined():
            raise BRepEvaluationError("OpenCascade curvature is undefined")
        lprop_mean = orientation * float(properties.MeanCurvature())
        lprop_gaussian = float(properties.GaussianCurvature())

    mean_u = mean_v = gradient_squared = None
    if need_third:
        suuu, svvv, suuv, suvv = map(_array, derivatives[5:])
        e_u = 2.0 * float(su @ suu)
        f_u = float(suu @ sv + su @ suv)
        g_u = 2.0 * float(sv @ suv)
        e_v = 2.0 * float(su @ suv)
        f_v = float(suv @ sv + su @ svv)
        g_v = 2.0 * float(sv @ svv)

        q_u = np.cross(suu, sv) + np.cross(su, suv)
        q_v = np.cross(suv, sv) + np.cross(su, svv)
        base_normal_u = (q_u - base_normal * float(base_normal @ q_u)) / w
        base_normal_v = (q_v - base_normal * float(base_normal @ q_v)) / w
        normal_u = orientation * base_normal_u
        normal_v = orientation * base_normal_v

        e_second_u = float(normal_u @ suu + normal @ suuu)
        f_second_u = float(normal_u @ suv + normal @ suuv)
        g_second_u = float(normal_u @ svv + normal @ suvv)
        e_second_v = float(normal_v @ suu + normal @ suuv)
        f_second_v = float(normal_v @ suv + normal @ suvv)
        g_second_v = float(normal_v @ svv + normal @ svvv)

        numerator_u = (
            e_second_u * g_first
            + e_second * g_u
            - 2.0 * (f_second_u * f_first + f_second * f_u)
            + g_second_u * e_first
            + g_second * e_u
        )
        numerator_v = (
            e_second_v * g_first
            + e_second * g_v
            - 2.0 * (f_second_v * f_first + f_second * f_v)
            + g_second_v * e_first
            + g_second * e_v
        )
        determinant_u = e_u * g_first + e_first * g_u - 2.0 * f_first * f_u
        determinant_v = e_v * g_first + e_first * g_v - 2.0 * f_first * f_v
        mean_u = (numerator_u * determinant - numerator * determinant_u) / (
            2.0 * determinant * determinant
        )
        mean_v = (numerator_v * determinant - numerator * determinant_v) / (
            2.0 * determinant * determinant
        )
        gradient_squared = (
            g_first * mean_u * mean_u - 2.0 * f_first * mean_u * mean_v + e_first * mean_v * mean_v
        ) / determinant
        if gradient_squared < 0.0 and abs(gradient_squared) < 1.0e-12:
            gradient_squared = 0.0

    return SurfaceDifferential(
        position=np.array([point.X(), point.Y(), point.Z()], dtype=float),
        normal=normal,
        mean_curvature=float(mean),
        gaussian_curvature=float(gaussian),
        mean_u=None if mean_u is None else float(mean_u),
        mean_v=None if mean_v is None else float(mean_v),
        gradient_mean_squared=(None if gradient_squared is None else float(gradient_squared)),
        jacobian=w,
        metric_determinant=float(determinant),
        metric_condition=float(metric_condition),
        lprop_mean_curvature=lprop_mean,
        lprop_gaussian_curvature=lprop_gaussian,
        orientation_reversed=orientation_reversed,
    )


def face_continuity_intervals(face: Any, *, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return trimmed parameter intervals split at requested continuity."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_C2, GeomAbs_C3
    from OCP.TColStd import TColStd_Array1OfReal

    continuity = GeomAbs_C3 if order >= 3 else GeomAbs_C2
    surface = BRepAdaptor_Surface(face, True)

    def intervals(axis: str) -> np.ndarray:
        count = int(getattr(surface, f"Nb{axis}Intervals")(continuity))
        values = TColStd_Array1OfReal(1, count + 1)
        getattr(surface, f"{axis}Intervals")(values, continuity)
        result = np.array(
            [values.Value(index) for index in range(1, count + 2)],
            dtype=float,
        )
        lower = float(getattr(surface, f"First{axis}Parameter")())
        upper = float(getattr(surface, f"Last{axis}Parameter")())
        result = np.clip(result, min(lower, upper), max(lower, upper))
        return np.unique(np.concatenate(([lower], result, [upper])))

    return intervals("U"), intervals("V")
