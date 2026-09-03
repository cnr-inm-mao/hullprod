<p align="center">
  <img
    src="https://raw.githubusercontent.com/cnr-inm-mao/hullprod/main/assets/branding/hullprod-logo-horizontal.png"
    alt="HullProd logo"
    width="420"
  >
</p>

# HullProd

[![CI](https://github.com/cnr-inm-mao/hullprod/actions/workflows/ci.yml/badge.svg)](https://github.com/cnr-inm-mao/hullprod/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/hullprod.svg)](https://pypi.org/project/hullprod/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22287357.svg)](https://doi.org/10.5281/zenodo.22287357)
![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)

**Geometry-based producibility screening for ship hull surfaces.**

HullProd computes a compact signature of double-curvature intensity, sign, and
areal composition directly from native IGES/STEP BRep geometry or triangulated
STL/OBJ/PLY surfaces. It also exports distributed fields, plots, validity, and
representation provenance.

## Installation

HullProd requires Python 3.10–3.12. The intended public installation command is:

```bash
pip install hullprod
```

One installation includes both the native OpenCascade and triangle-mesh
backends; no separate CAD extra or system OpenCascade installation is required.

## Quick start

Assess native CAD directly:

```bash
hullprod myvessel.iges
```

The same command works for a triangulated surface:

```bash
hullprod myvessel.stl
```

HullProd selects the backend from the file extension and writes a predictable
`myvessel_hullprod/` result directory.

To supply the normalization length explicitly:

```bash
hullprod myvessel.iges --lref 200000
```

`--lref` uses the working length unit of the input geometry. For CAD imported
in millimetres, `200000` means 200000 mm, or 200 m. HullProd prints the source
unit, working unit, and interpreted value before integration and warns about a
large discrepancy from the automatic geometric span without changing the
supplied value.

See [Getting started](docs/getting-started.md) for output-directory controls
and other normal options.

## Recommended signature

HullProd 1.0 reports exactly:

```text
[I_D, I_D_plus, I_D_minus,
 a_C_flat, a_C_single, a_C_elliptic, a_C_saddle]
```

- `I_D` measures total normalized double-curvature intensity.
- `I_D_plus` and `I_D_minus` separate elliptic/synclastic and
  saddle/reverse/anticlastic intensity.
- The four `a_C` values are represented-valid-area fractions for flat, singly
  curved, elliptic, and saddle/reverse regions.

Together they describe **intensity**, **sign**, and **areal extent**. HullProd
does not combine them into a universal scalar producibility score. The default
quasi-zero factors are `h_f = k_f = 1e-4`; these are numerical classification
thresholds, not manufacturing limits. See [Metrics](docs/metrics.md) for the
equations and interpretation.

## What HullProd produces

A normal result directory contains:

- `signature.json` and `signature.csv` for the recommended global signature;
- `report.html` for a compact, static user report;
- `validity.json` and `provenance.json` for scientific interpretation and audit;
- `plots/` with developability-density and curvature-class maps; and
- `fields/` with ParaView-readable VTP and portable CSV surface fields.

The complete output contract and file roles are documented in
[Outputs](docs/outputs.md).

## Supported geometry

| Representation | Formats | Backend |
|---|---|---|
| Native BRep | `.iges`, `.igs`, `.step`, `.stp` | Direct CAD derivatives and trimmed-domain quadrature |
| Triangle mesh | `.stl`, `.obj`, `.ply` | Discrete curvature reconstruction on the supplied mesh |

Unsupported formats fail with an actionable error; HullProd does not silently
convert a BRep to a mesh for canonical global evaluation or fit CAD to a mesh.

## BRep and mesh realizations

Native BRep evaluation is the canonical realization when source CAD is
available. Its global metrics come from direct surface derivatives and
trimmed-domain integration, not from the display/export tessellation.

Mesh evaluation is supported when triangulated geometry is the available
representation. Its derivative-based values are explicitly
representation-sensitive and should be interpreted with the recorded mesh
quality and refinement provenance. This distinction does not make mesh results
invalid; it makes their represented geometry part of the result.

Without `--lref`, both backends use the same conceptual automatic convention: a
rigid-motion-invariant principal-axis projected span, with a deterministic
centroid-radial fallback for isotropic samples. It is recorded as
`auto_principal_span` and is never described as `L_pp`.

## Python API

The Python API uses the same assessment pipeline as the CLI:

```python
from hullprod import assess

result = assess("myvessel.iges")
print(result.signature)
```

See the [Python API guide](docs/python-api.md).

## Documentation

- [Getting started](docs/getting-started.md)
- [Recommended metrics](docs/metrics.md)
- [Result files and field exports](docs/outputs.md)
- [Validity and provenance](docs/validity-and-provenance.md)
- [Python API](docs/python-api.md)
- [Experimental quantities](docs/experimental.md)
- [Release history](CHANGELOG.md)

README plus versioned Markdown under `docs/` are the complete v1 documentation
system; no hosted documentation site is required.

## Scientific scope and limitations

HullProd is a geometry-based early-design screening tool, not a
fabrication-cost predictor. It does not directly predict labor, forming route,
production schedule, or shipyard-specific performance. It does not perform panelization, seam
placement, forming simulation, optimization, or geometry repair.

Auxiliary, mesh-diagnostic, and experimental quantities are kept separate from
the recommended signature. Their status is explained in
[Experimental quantities](docs/experimental.md).

## Funding

Development of HullProd was supported by the U.S. Office of Naval Research
(ONR), under Grant No. N00014-26-1-2164, as part of the BEAM project
(*Bayesian Exploration and Optimization for Hull-form Architecture and
Producibility Modeling*).

The views and conclusions expressed in this software and its documentation are
those of the authors and do not necessarily reflect the views of the Office of
Naval Research.

## Citation

If you use HullProd in scientific work, please cite the archived software
release:

> Serani, A. (2026). *HullProd: Geometry-Based Producibility Metrics for Ship
> Hull Forms* (Version 1.0.0) [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.22287357

Machine-readable citation metadata are provided in
[CITATION.cff](CITATION.cff). Please also cite the associated scientific
publication once its final journal metadata are available.

## Contributing

Bug reports, documentation corrections, reproducible analytical tests, and
carefully scoped numerical improvements are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) before proposing changes to scientific
definitions.

## License

HullProd is distributed under the [BSD-3-Clause license](LICENSE). The bundled
analytical sphere fixtures are project-owned; restricted benchmark geometry is
not redistributed.
