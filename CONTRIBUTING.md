# Contributing to HullProd

HullProd is scientific software with a deliberately narrow scope: geometry-based
producibility signatures from triangulated and native-BRep external hull
surfaces. Contributions are welcome within that scope.

## Scope of contributions

Appropriate contributions include:

- bug fixes;
- improved mesh-quality diagnostics;
- improved numerical robustness of existing metrics;
- additional analytic tests;
- benchmark workflows;
- documentation improvements;
- export utilities for visualization tools;
- reproducible examples based on redistributable geometries.

Contributions that require shipyard-specific production data, structural
scantlings, cost models, man-hour models, or detailed fabrication planning
should be discussed first because they lie outside the mesh/native-CAD
geometry-assessment workflow.

## Development setup

A CPython 3.10, 3.11, or 3.12 environment is required.

Create and activate a development environment with a supported interpreter:

    python3.12 -m venv .venv
    source .venv/bin/activate

Install the package in editable mode with development dependencies:

    python -m pip install -U pip
    python -m pip install -e ".[dev]"

## Checks before committing

Before committing, please run:

    ruff check .
    pytest -q
    python -m build
    python -m twine check dist/*.whl dist/*.tar.gz
    git diff --check

## Coding style

The project uses Ruff for linting. Code should favor clarity, explicit variable
names, and reproducible numerical behavior over compactness. Pull requests
should include focused tests and documentation for user-visible changes.

## Scientific caution

HullProd metrics are geometric descriptors. They should not be presented as direct predictors of fabrication cost, man-hours, shipyard productivity, or production schedule unless additional calibrated production data are introduced and validated.
