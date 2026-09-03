"""Automatic reference-length invariance and provenance gates."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from hullprod.metrics import compute_metrics
from hullprod.reference_length import resolve_reference_length


def _automatic(points: np.ndarray) -> tuple[float, dict]:
    return resolve_reference_length(points, None, source="test_geometry")


def _controlled_samples() -> dict[str, np.ndarray]:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    return {
        "elongated_hull_like": sphere.vertices * np.array([5.0, 1.2, 0.8]),
        "sphere": sphere.vertices,
        "asymmetric": np.array(
            [
                [0.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [4.0, 2.0, 0.0],
                [1.0, -0.7, 1.0],
                [2.0, 0.2, -1.4],
                [5.0, 0.5, 0.2],
                [3.0, -1.2, 0.4],
            ]
        ),
    }


def test_auto_principal_span_is_rigid_motion_and_scale_invariant() -> None:
    rotations = Rotation.from_euler(
        "xyz",
        [[0.0, 0.0, 0.0], [17.0, 31.0, 73.0], [123.0, -28.0, 11.0]],
        degrees=True,
    ).as_matrix()
    translation = np.array([3.25, -7.5, 11.75])
    for name, points in _controlled_samples().items():
        baseline, metadata = _automatic(points)
        assert metadata["mode"] == "auto_principal_span"
        if name == "sphere":
            assert metadata["method"] == "centroid_radial_diameter_isotropic_fallback"
        else:
            assert metadata["method"] == "principal_axis_projected_span"
        for matrix in rotations:
            moved = points @ matrix.T + translation
            value, _ = _automatic(moved)
            assert value == pytest.approx(baseline, rel=2.0e-13, abs=2.0e-13)
        for scale in (0.1, 1.0, 10.0):
            value, _ = _automatic(points * scale)
            assert value == pytest.approx(scale * baseline, rel=2.0e-13, abs=2.0e-13)


def test_axis_aligned_diagonal_candidate_is_rotation_dependent() -> None:
    points = _controlled_samples()["elongated_hull_like"]
    rotated = points @ Rotation.from_euler("xyz", [17.0, 31.0, 73.0], degrees=True).as_matrix().T
    first = np.linalg.norm(np.ptp(points, axis=0))
    second = np.linalg.norm(np.ptp(rotated, axis=0))
    assert abs(first - second) / first > 1.0e-3


def test_float32_round_trip_does_not_create_a_spherical_major_axis() -> None:
    points = _controlled_samples()["sphere"]
    matrix = Rotation.from_euler("xyz", [17.0, 31.0, 73.0], degrees=True).as_matrix()
    rotated_and_quantized = (points @ matrix.T + [3.0, -4.0, 7.0]).astype(np.float32)
    baseline, _ = _automatic(points)
    value, metadata = _automatic(rotated_and_quantized)
    assert metadata["method"] == "centroid_radial_diameter_isotropic_fallback"
    assert value == pytest.approx(baseline, rel=3.0e-7)


def test_mesh_signature_is_uniform_scale_invariant_with_automatic_lref() -> None:
    source = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    vertices = source.vertices * np.array([5.0, 1.2, 0.8])
    reference = None
    reference_lref = None
    for scale in (0.1, 1.0, 10.0):
        mesh = trimesh.Trimesh(vertices=vertices * scale, faces=source.faces, process=False)
        result = compute_metrics(mesh)
        if reference is None:
            reference = result.signature
            reference_lref = result.metrics["length_ref"] / scale
            continue
        assert result.metrics["length_ref"] == pytest.approx(
            reference_lref * scale, rel=2.0e-12
        )
        for key in ("I_D", "I_D_plus", "I_D_minus"):
            assert result.signature[key] == pytest.approx(reference[key], rel=2.0e-10)
        for key in ("flat", "single", "elliptic", "saddle"):
            assert result.signature["a_C"][key] == pytest.approx(
                reference["a_C"][key], abs=2.0e-12
            )


def test_explicit_reference_length_is_exact_and_skips_samples() -> None:
    value, metadata = resolve_reference_length(
        None,
        142.0,
        source="ProducibilityConfig.length_ref_or_cli_lref",
    )
    assert value == 142.0
    assert metadata["value"] == 142.0
    assert metadata["mode"] == "explicit_user"
    assert metadata["method"] == "explicit_user_value"
    assert metadata["is_lpp"] is False


def test_brep_reference_sampling_is_rigid_motion_and_scale_invariant() -> None:
    pytest.importorskip("OCP")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

    from hullprod.brep_display import sample_brep_reference_points
    from hullprod.brep_geometry import model_from_shape
    from hullprod.brep_metrics import compute_brep_metrics
    from hullprod.types import ProducibilityConfig

    def transformed(shape, *, scale: float = 1.0, angle: float = 0.0):
        if scale != 1.0:
            operation = gp_Trsf()
            operation.SetScaleFactor(scale)
            shape = BRepBuilderAPI_Transform(shape, operation, True).Shape()
        if angle:
            operation = gp_Trsf()
            operation.SetRotation(
                gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.3, 0.5, 0.8)),
                angle,
            )
            shape = BRepBuilderAPI_Transform(shape, operation, True).Shape()
            operation = gp_Trsf()
            operation.SetTranslation(gp_Vec(3.0, -4.0, 7.0))
            shape = BRepBuilderAPI_Transform(shape, operation, True).Shape()
        return shape

    shape = BRepPrimAPI_MakeSphere(1.0).Shape()
    points, sampling = sample_brep_reference_points(model_from_shape(shape))
    baseline, metadata = resolve_reference_length(
        points,
        None,
        source="controlled_BRep_reference_tessellation",
        sampling=sampling,
    )
    assert metadata["sampling"]["canonical_metrics_depend_on_sampling"] is False
    moved_shape = transformed(shape, angle=0.73)
    scaled_shape = transformed(shape, scale=10.0)
    moved_points, moved_sampling = sample_brep_reference_points(model_from_shape(moved_shape))
    moved, _ = resolve_reference_length(
        moved_points,
        None,
        source="controlled_BRep_reference_tessellation",
        sampling=moved_sampling,
    )
    scaled_points, scaled_sampling = sample_brep_reference_points(model_from_shape(scaled_shape))
    scaled, _ = resolve_reference_length(
        scaled_points,
        None,
        source="controlled_BRep_reference_tessellation",
        sampling=scaled_sampling,
    )
    assert moved == pytest.approx(baseline, rel=2.0e-10)
    assert scaled == pytest.approx(10.0 * baseline, rel=2.0e-10)

    config = ProducibilityConfig(
        length_ref=None,
        n_stations=0,
        brep_display_mesh=False,
        brep_quadrature_order=5,
        brep_quadrature_max_depth=0,
        brep_quadrature_base_subdivisions=2,
    )
    results = [
        compute_brep_metrics(model_from_shape(candidate), config).result
        for candidate in (shape, moved_shape, scaled_shape)
    ]
    for candidate in results[1:]:
        for key in ("I_D", "I_D_plus", "I_D_minus"):
            assert candidate.signature[key] == pytest.approx(results[0].signature[key], rel=2.0e-10)
        assert candidate.signature["a_C"] == pytest.approx(results[0].signature["a_C"])
