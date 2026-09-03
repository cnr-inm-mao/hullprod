from __future__ import annotations

import trimesh

from hullprod.metrics import compute_metrics
from hullprod.types import ProducibilityConfig
from hullprod.validity import VALIDITY_VOCABULARY, metric_validity_record


def test_validity_record_keeps_value_and_mathematical_status_independent() -> None:
    record = metric_validity_record(
        value=None,
        validity="geometric_singularity_nonintegrable",
        additional_validity=("insufficient_C3_for_fairness",),
        reason="analytical control",
        represented_area=2.0,
        valid_area=None,
        numerical_convergence={"bounded_refinement_levels": 2},
        representation="brep",
        backend="brep_native",
    )
    assert record["value"] is None
    assert record["computable"] is False
    assert record["status"] == "geometric_singularity_nonintegrable"
    assert record["display_status"] == "SINGULAR"
    assert set(record["validity_codes"]).issubset(VALIDITY_VOCABULARY)


def test_mesh_backend_reports_each_metric_without_changing_values() -> None:
    result = compute_metrics(
        trimesh.creation.icosphere(subdivisions=1, radius=1.0),
        ProducibilityConfig(length_ref=2.0, n_stations=0),
    )
    validity = result.metadata["metric_validity"]
    for name in (
        "surface_area",
        "curvature_energy",
        "developability_deviation",
        "developability_deviation_positive",
        "developability_deviation_negative",
        "developability_area_ratio",
        "curvature_classes",
        "curvature_fairness",
        "section_waviness",
        "section_waviness_fft",
        "local_plate_twist",
    ):
        assert name in validity
        assert validity[name]["representation"] == "mesh"
        assert validity[name]["backend"] == "mesh_rusinkiewicz"
    assert validity["curvature_energy"]["status"] == "not_evaluated"
    assert validity["curvature_fairness"]["status"] == "not_evaluated"
    assert validity["section_waviness"]["status"] == "not_evaluated"
    assert validity["section_waviness_fft"]["status"] == "not_evaluated"
    assert validity["local_plate_twist"]["status"] == "not_evaluated"
