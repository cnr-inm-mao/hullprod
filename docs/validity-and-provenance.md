# Validity and provenance

A finite number is not automatically a valid canonical metric. A quadrature
calculation can stop at finite depth without converging, a geometric
singularity can make one integral non-integrable, and a mesh estimate can be
sensitive to the supplied tessellation. HullProd therefore reports validity
separately for each metric; one unavailable metric does not invalidate the
entire geometry.

## Important statuses

| Machine-readable status | Meaning |
|---|---|
| `valid` | The value satisfies the applicable numerical and regularity contract. |
| `valid_improper_integral_convergent` | A singular/degenerate set exists, but nested evidence establishes a finite improper integral. This is cautionary, not fatal. |
| `quadrature_unconverged` | The configured refinement did not establish a stable integral; do not treat the finite-depth sample as canonical. |
| `geometric_singularity_nonintegrable` | The continuous metric diverges for the represented geometry; a finite display sample is not a replacement. |
| `mesh_representation_sensitive` | The value is valid for the supplied mesh representation, but representation-independent interpretation needs refinement evidence. |
| `not_evaluated` | The optional or experimental calculation was not requested. |
| `not_applicable` | The metric does not apply to this backend or representation. |

Additional detailed codes may describe parameter degeneracy or insufficient
surface regularity. `validity.json` gives the reason, represented and valid
areas, convergence metadata, representation, and backend for each result.

The console and HTML report render `mesh_representation_sensitive` as
`MESH-SENSITIVE`, followed by a short explanation. This is more specific than
a generic caution: the value is valid for the supplied triangulation, while
derivative estimates may depend on resolution, connectivity, and mesh quality.

Scientific validity and local numerical notes are distinct. A BRep result can
remain `VALID` when its global retained integral satisfies the established
stability criterion while some local cells reach the bounded quadrature depth.
In that case the console and report show a secondary `BRep quadrature: CAUTION`
note with the available cell count and final global relative change. This note
does not relabel the integral as divergent or unconverged; full evidence remains
in `validity.json` and `provenance.json`.

## BRep and mesh are distinct realizations

When source IGES/STEP CAD is available, native BRep evaluation is the canonical
realization: HullProd uses direct surface derivatives and trimmed-domain
quadrature. A controlled tessellation may estimate the automatic convenience
reference length and carry fields for visualization, but never supplies the
canonical BRep curvature integral.

When STL/OBJ/PLY is the available geometry, HullProd evaluates the represented
mesh with its validated discrete estimator and reports mesh-quality and
boundary provenance. A difference from a BRep result does not make the mesh
result useless or automatically invalid; it means the representation and its
resolution are part of the interpretation.

## Provenance to retain

Keep `provenance.json` with any reported signature. It records the input name,
extension and hash; HullProd/backend versions; reference-length value, mode and
sampling method; quasi-zero thresholds; represented valid/invalid area;
quadrature or mesh settings; BRep topology or mesh quality; source/working
units where available; warnings; and run timing.

An explicit `--lref` is stored as `explicit_user` in the geometry working unit.
The provenance also retains the automatic comparison span, their ratio, and a
warning when the ratio is below 0.1 or above 10. HullProd reports these units
before integration and never changes the explicit value.
