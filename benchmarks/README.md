# HullProd verification benchmarks

This directory contains a compact, public-safe verification subset for the
HullProd 1.0 scientific contract. It does not contain downloaded hull masters,
private geometry, or generated visualization campaigns.

## Contents

- `config/retained_signature.json` records the frozen four-case comparison
  settings used by the retained-signature tests.
- `results/retained_signature_closing_audit/` contains machine-readable
  analytical, signed-developability, representation, refinement, and
  improper-integral evidence.
- `scripts/wigley_control.py` provides the deterministic project-owned Wigley
  control.
- `scripts/multi_hull_common.py` contains representation and tessellation
  utilities exercised without redistributing external hull geometry.

## Data policy

Externally sourced ship geometry is not redistributed. The test suite skips
geometry-dependent checks when a separately obtained master is absent. The
bundled package examples are project-owned analytical spheres and are not ship
benchmark substitutes.

The recommended HullProd signature is limited to total and signed
developability plus represented-valid-area curvature-class composition.
Historical higher-regularity and section-metric screening campaigns are
deliberately excluded from this public benchmark subset. Their optional runtime
status is documented separately in `docs/experimental.md`.
