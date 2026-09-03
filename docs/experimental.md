# Auxiliary, diagnostic, and experimental quantities

The recommended HullProd 1.0 signature is limited to total and signed
developability intensity plus the four curvature-class area fractions:

```text
I_D, I_D_plus, I_D_minus,
a_C_flat, a_C_single, a_C_elliptic, a_C_saddle
```

Other quantities have deliberately different roles.

## Auxiliary

`developability_area_ratio` is the valid-area fraction above the shared
Gaussian-curvature threshold. Under the common threshold it equals
`a_C_elliptic + a_C_saddle`, so it is retained for compatibility but is not an
independent signature component.

## Mesh diagnostics

Mesh-quality statistics, boundary information, triangle/edge statistics, and
local plate twist describe the assessed representation. They help interpret
mesh curvature reliability; they are not geometry-invariant signature
components or manufacturing acceptance limits.

## Screened experimental metrics

`curvature_energy`, `curvature_fairness`, `section_waviness`,
`section_waviness_fft`, and robust research variants were investigated but are
not part of the recommended v1 signature. They are not computed or plotted by
default.

Request the screened set deliberately with:

```bash
hullprod myvessel.iges --experimental
```

Experimental values appear in a separately labelled JSON/report section and
never enter `recommended_signature`. Their mathematical meanings and existing
raw keys are preserved for research compatibility.
