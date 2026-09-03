# Result files

A normal run writes this compact tree:

```text
vessel_hullprod/
├── signature.json
├── signature.csv
├── validity.json
├── provenance.json
├── summary.txt
├── report.html
├── metrics.json
├── field_manifest.json
├── output_manifest.json
├── geometry/display_mesh.stl
├── plots/developability_density.png
├── plots/curvature_classes.png
├── fields/surface_fields.vtp
└── fields/surface_fields.csv
```

## Primary results

- `signature.json` is the stable structured result. Use it in software that
  needs the recommended signature, validity, auxiliary values, diagnostics,
  and their hierarchy.
- `signature.csv` is a compact one-row table for pandas, spreadsheets,
  optimization pipelines, and surrogate-model inputs.
- `validity.json` contains every metric-specific validity record. Inspect this
  before treating a finite value as canonical.
- `provenance.json` records input identity, backend, representation, reference
  length, thresholds, numerical settings, valid area, geometry quality, units,
  warnings, and timing.
- `summary.txt` is a concise plain-text signature and run-location summary.
- `report.html` is a compact, self-contained user report with input and
  representation, recommended signature, curvature-class composition, the two
  plots, validity and numerical notes, concise provenance, output links, and
  scope/limitations. It does not duplicate raw provenance or per-face
  quadrature internals; use the linked JSON records for the complete audit
  trail. Open it directly in a browser; no server or external JavaScript is
  required.

## Plots and distributed fields

- `plots/developability_density.png` maps `q_D = |K| L_ref^2`; invalid regions
  are visually distinct.
- `plots/curvature_classes.png` maps flat, single, elliptic, saddle/reverse, and
  invalid categories with stable labels.
- `fields/surface_fields.vtp` is VTK PolyData for interactive visualization.
  It includes developability density and positive/negative components,
  curvature class, validity, and available `H`/`K` samples.
- `fields/surface_fields.csv` provides coordinates and the same practical field
  columns for generic tabular tools.
- `geometry/display_mesh.stl` is the surface carrying output fields. For BRep
  input it is visualization transport only and is not the source of canonical
  global integration.

To open the VTP field in ParaView, start ParaView, choose **File > Open**, select
`fields/surface_fields.vtp`, click **Apply**, and select a point-data array from
the coloring menu.

## Manifests and compatibility view

- `field_manifest.json` documents each exported array, association, formula,
  interpretation, and validity.
- `output_manifest.json` inventories the files produced by the run.
- `metrics.json` retains the raw/compatibility key view for established users;
  new integrations should start with `signature.json`.

`--no-plots` omits only the two PNGs. Field exports and machine-readable
results remain available.
