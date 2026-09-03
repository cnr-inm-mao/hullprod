# Python API

The stable `assess` entry point uses the same backend dispatch, numerical
pipeline, and output writers as the command line:

```python
from hullprod import assess

result = assess("myvessel.iges")

print(result.signature)
print(result.validity)
print(result.provenance)
print(result.output_paths)
```

Write the complete result directory and plots:

```python
result = assess(
    "myvessel.iges",
    out_dir="myvessel_hullprod",
    plots=True,
)
```

Supply an authoritative reference length, or disable plots while still writing
fields and reports:

```python
result = assess(
    "myvessel.iges",
    lref=142.0,
    out_dir="results",
    plots=False,
)
```

Use `overwrite=True` only when intentionally writing into an existing nonempty
directory. `assess_hull(...)` and `ProducibilityConfig` remain available for
advanced and compatibility use.
