# Getting started

## Install

HullProd 1.0 supports Python 3.10--3.12. Install the standard package with both
native-CAD and mesh support:

```bash
python -m pip install hullprod
```

No configuration file or separate CAD extra is required.

## Assess geometry

Use the input file as the root positional argument:

```bash
hullprod vessel.iges
hullprod vessel.step
hullprod vessel.stl
```

`.iges`, `.igs`, `.step`, and `.stp` select the native BRep backend. `.stl`,
`.obj`, and `.ply` select the triangle-mesh backend. Unsupported extensions
produce an actionable error rather than an implicit conversion.

The default output directory is `<input-stem>_hullprod/` in the current working
directory. For example, `hullprod vessel.iges` writes `vessel_hullprod/`.

## Common options

Choose a result directory:

```bash
hullprod vessel.iges --out results
```

Supply an authoritative positive reference length in the file's working length
unit:

```bash
hullprod vessel.iges --lref 200000
```

For example, if a CAD file's working unit is millimetres, that command means
`L_ref = 200000 mm = 200 m`. It does **not** mean 200000 m, and `--lref 200`
would mean 200 mm rather than 200 m. HullProd prints the detected source unit,
working unit, and interpreted reference length before metric integration.

Even with an explicit value, HullProd estimates the automatic geometric span
for a plausibility check. If `L_user / L_auto < 0.1` or `> 10`, it warns before
integration and records the warning in provenance. The explicit value remains
authoritative; HullProd never guesses a conversion or silently rescales it.

Without `--lref`, HullProd estimates an automatic geometric extent. It projects
the represented geometry samples onto their dominant PCA axis and uses that
span. If the leading axis is numerically non-unique, as for a sphere, it uses
twice the maximum distance from the sample centroid. Numerically non-unique
means a relative leading covariance-eigenvalue gap no greater than `1e-5`; the
value and decision metadata are recorded in provenance. The result is recorded
as `auto_principal_span`, is rigid-motion invariant and scale covariant, and is
not an inferred `L_pp`.

STL, OBJ, and PLY do not declare physical length units, so HullProd reports
their working unit honestly as `input-length units`.

An existing nonempty result directory is protected. Reuse it deliberately with:

```bash
hullprod vessel.iges --out results --overwrite
```

Skip PNG generation while retaining machine-readable fields and reports:

```bash
hullprod vessel.iges --no-plots
```

Assessment completes with exit code 0 when artifacts are written, including
when individual validity records contain cautionary statuses. Fatal input,
parser, backend, I/O, or runtime failures return nonzero. The older
`hullprod assess vessel.iges` spelling remains a compatibility alias.
