# Changelog

All notable changes to HullProd will be documented in this file.

HullProd follows semantic versioning. Patch releases fix defects without
breaking the public contract, minor releases add backward-compatible features,
and major releases may change public schemas, formulas, CLI behavior, or field
and output-layout contracts.

## [1.0.0] - 2026-09-03

### Added

- A zero-configuration `hullprod INPUT` workflow for IGES, STEP, STL, OBJ, and
  PLY, with automatic native-BRep or triangle-mesh backend selection.
- The explicit recommended signature `I_D`, `I_D_plus`, `I_D_minus`, and the
  four represented-valid-area curvature-class fractions.
- Stable `signature.json`, one-row `signature.csv`, `validity.json`,
  `provenance.json`, and concise text/HTML summaries.
- ParaView-readable `surface_fields.vtp`, portable field CSV, and default
  developability-density and curvature-class plots.
- A shared `hullprod.assess(...)` Python API, deterministic automatic geometric
  reference length, explicit `--lref`, and safe nonempty-output handling.
- Small redistributable IGES and STL analytical examples for installed-wheel
  smoke testing.
- Clean-wheel validation on Linux with Python 3.10–3.12 and on macOS Apple
  Silicon with Python 3.12, including native BRep and mesh inputs.

### Changed

- Standard installation now includes the `cadquery-ocp` dependency used for
  native IGES/STEP evaluation; BRep global values do not come from display
  tessellation.
- Zero-configuration reference length now uses a rigid-motion-invariant
  principal-axis projected span, with a centroid-radial fallback for isotropic
  geometry; explicit user and configured benchmark reference lengths remain
  authoritative.
- Gaussian-curvature mesh results exclude invalid open-boundary vertices and
  report represented-valid-area normalization and mesh-quality provenance.
- The legacy `hullprod assess INPUT` command and raw metric keys remain
  available for compatibility.
- The default static HTML report is now a compact user view; full provenance,
  validity, and numerical internals remain in linked machine-readable files.
- Source/working units and the interpreted reference length are shown before
  integration. Explicit reference lengths are compared with the automatic span
  and receive a nonfatal warning below a 0.1 or above a 10 ratio.
- Human-facing output distinguishes scientifically valid results from local
  BRep quadrature cautions and mesh representation sensitivity.

### Experimental and non-recommended quantities

- Curvature energy, curvature fairness, section waviness, FFT section waviness,
  and robust research variants are outside the recommended signature and
  require explicit opt-in.
- `developability_area_ratio` is auxiliary/redundant; local plate twist and
  triangle/mesh-quality quantities are representation diagnostics.
- HullProd does not provide a fabrication-cost, man-hour, forming-route, or
  shipyard-calibrated composite score.
