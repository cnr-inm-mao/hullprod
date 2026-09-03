"""Focused analytical evidence for current open-boundary estimator behavior."""

import numpy as np
import pytest
import trimesh

from hullprod.experimental_curvature import (
    gaussian_curvature_boundary_interpolated,
    gaussian_curvature_interior_only,
    mean_curvature_legacy_v012,
    mean_curvature_normal_vector,
    mean_curvature_projected,
)
from hullprod.mesh_ops import boundary_vertices, gaussian_curvature, mean_curvature, vertex_areas


def make_flat_patch(n=11, nonuniform=False):
    x = np.linspace(0.0, 1.0, n)
    if nonuniform:
        x = x**3
    y = np.linspace(0.0, 1.0, n)
    vertices = np.array([[xi, yi, 0.0] for xi in x for yi in y])

    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = (i + 1) * n + j
            c = (i + 1) * n + j + 1
            d = i * n + j + 1
            faces.extend([[a, b, c], [a, c, d]])

    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def make_open_cylinder(radius=1.0, height=5.0, sections=64, axial_points=9):
    theta = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    x_values = np.linspace(-0.5 * height, 0.5 * height, axial_points)
    vertices = np.array(
        [[x, radius * np.cos(t), radius * np.sin(t)] for x in x_values for t in theta]
    )

    faces = []
    for axial in range(axial_points - 1):
        offset = axial * sections
        next_offset = (axial + 1) * sections
        for i in range(sections):
            j = (i + 1) % sections
            faces.extend(
                [
                    [offset + i, offset + j, next_offset + j],
                    [offset + i, next_offset + j, next_offset + i],
                ]
            )
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def test_flat_open_patch_exposes_boundary_angle_defect_contamination():
    """Scientific invariant versus current implementation behavior.

    True Gaussian curvature is zero everywhere on this planar patch.  Interior
    defects are roundoff-sized, while the boundary rule contributes the
    polygon's turning angle (2*pi) to the absolute-curvature integral.
    """
    mesh = make_flat_patch()
    areas = vertex_areas(mesh)
    boundary = boundary_vertices(mesh)
    K = gaussian_curvature(mesh, areas=areas)

    assert np.max(np.abs(K[~boundary])) < 1.0e-10
    assert np.sum(areas * np.abs(K)) == pytest.approx(2.0 * np.pi)
    assert np.sum(areas[boundary] * np.abs(K[boundary])) == pytest.approx(np.sum(areas * np.abs(K)))


def test_flat_open_patch_mean_curvature_boundary_term_is_spacing_sensitive():
    """The unmodified boundary cotangent Laplacian is not a surface-H estimate."""
    uniform = make_flat_patch()
    nonuniform = make_flat_patch(nonuniform=True)

    def summarize(mesh):
        areas = vertex_areas(mesh)
        boundary = boundary_vertices(mesh)
        H = mean_curvature(mesh, areas=areas)
        return np.max(H[~boundary]), np.max(H[boundary])

    uniform_interior, uniform_boundary = summarize(uniform)
    nonuniform_interior, nonuniform_boundary = summarize(nonuniform)

    assert uniform_interior < 1.0e-10
    assert nonuniform_interior < 1.0e-8
    assert uniform_boundary > 1.0
    assert nonuniform_boundary > 50.0 * uniform_boundary


@pytest.mark.parametrize("nonuniform", [False, True])
def test_projected_experimental_mean_curvature_is_zero_on_open_plane(nonuniform):
    """Normal projection removes tangential open-boundary contamination."""
    mesh = make_flat_patch(n=17, nonuniform=nonuniform)
    boundary = boundary_vertices(mesh)
    vector = mean_curvature_normal_vector(mesh)
    production = mean_curvature(mesh)
    projected = mean_curvature_projected(mesh)

    assert np.max(np.linalg.norm(vector[~boundary], axis=1)) < 1.0e-8
    np.testing.assert_allclose(production, np.linalg.norm(vector, axis=1))
    assert np.max(production[boundary]) > 1.0
    assert np.max(projected) < 1.0e-10


def test_experimental_gaussian_variants_do_not_assign_boundary_turning_to_plane():
    mesh = make_flat_patch(n=17, nonuniform=True)
    boundary = boundary_vertices(mesh)
    interior = gaussian_curvature_interior_only(mesh)
    extrapolated = gaussian_curvature_boundary_interpolated(mesh)

    assert np.all(np.isnan(interior[boundary]))
    assert np.max(np.abs(interior[~boundary])) < 1.0e-7
    assert np.all(np.isfinite(extrapolated))
    assert np.max(np.abs(extrapolated)) < 1.0e-7


@pytest.mark.parametrize("subdivisions", [2, 3, 4])
def test_production_sphere_mean_curvature_converges_to_analytic_value(subdivisions):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=2.0)
    production = mean_curvature(mesh)
    projected = mean_curvature_projected(mesh)

    assert np.median(production) == pytest.approx(0.5, rel=0.04)
    assert np.median(projected) == pytest.approx(0.5, rel=0.04)


def test_production_mean_curvature_is_twice_legacy_v012_behavior():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    production = mean_curvature(mesh)
    legacy = mean_curvature_legacy_v012(mesh)

    np.testing.assert_allclose(production, 2.0 * legacy, rtol=1.0e-14, atol=0.0)


def test_signed_production_curvature_preserves_existing_orientation_semantics():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    unsigned = mean_curvature(mesh)
    signed = mean_curvature(mesh, signed=True)
    legacy_signed = mean_curvature_legacy_v012(mesh, signed=True)

    np.testing.assert_allclose(np.abs(signed), unsigned, rtol=1.0e-14, atol=0.0)
    np.testing.assert_allclose(signed, 2.0 * legacy_signed, rtol=1.0e-14, atol=0.0)


def test_closed_sphere_gaussian_curvature_converges_to_analytic_value():
    errors = []
    for subdivisions in (2, 3, 4):
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=2.0)
        K = gaussian_curvature(mesh)
        errors.append(abs(np.median(K) - 0.25))

    assert errors[2] < errors[1] < errors[0]
    assert np.median(K) == pytest.approx(0.25, rel=0.002)


@pytest.mark.parametrize("sections", [32, 64, 128])
def test_production_cylinder_mean_curvature_converges_to_analytic_value(sections):
    mesh = make_open_cylinder(radius=2.0, sections=sections)
    boundary = boundary_vertices(mesh)
    production = mean_curvature(mesh)
    projected = mean_curvature_projected(mesh)

    assert np.median(production[~boundary]) == pytest.approx(0.25, rel=0.03)
    assert np.median(projected) == pytest.approx(0.25, rel=0.03)


def test_experimental_cylinder_gaussian_curvature_is_zero_at_open_ends():
    mesh = make_open_cylinder(radius=2.0, sections=64)
    boundary = boundary_vertices(mesh)
    interior = gaussian_curvature_interior_only(mesh)
    extrapolated = gaussian_curvature_boundary_interpolated(mesh)

    assert np.all(np.isnan(interior[boundary]))
    assert np.max(np.abs(interior[~boundary])) < 1.0e-10
    assert np.max(np.abs(extrapolated)) < 1.0e-10


def test_experimental_saddle_center_matches_negative_analytic_gaussian_curvature():
    n = 81
    amplitude = 0.25
    axis = np.linspace(-1.0, 1.0, n)
    vertices = np.array([[x, y, amplitude * (x**2 - y**2)] for x in axis for y in axis])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = (i + 1) * n + j
            c = (i + 1) * n + j + 1
            d = i * n + j + 1
            faces.extend([[a, b, c], [a, c, d]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    center = (n // 2) * n + n // 2

    K = gaussian_curvature_interior_only(mesh)
    projected_signed_H = mean_curvature_projected(mesh, signed=True)

    assert K[center] == pytest.approx(-4.0 * amplitude**2, rel=0.01)
    assert abs(projected_signed_H[center]) < 1.0e-10


def test_unit_sphere_mean_curvature_matches_analytic_value():
    """Scientific invariant: H=(k1+k2)/2 is one on a unit sphere."""
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    H = mean_curvature(mesh)
    assert np.median(H) == pytest.approx(1.0, rel=0.03)


def test_unit_cylinder_interior_mean_curvature_matches_analytic_value():
    """Scientific invariant: H=(k1+k2)/2 is 0.5 on a unit cylinder."""
    mesh = make_open_cylinder()
    boundary = boundary_vertices(mesh)
    H = mean_curvature(mesh)
    assert np.median(H[~boundary]) == pytest.approx(0.5, rel=0.03)
