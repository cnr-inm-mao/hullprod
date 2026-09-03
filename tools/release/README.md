# Release verification tools

Run the distribution-content audit after building:

```bash
python tools/release/audit_artifacts.py dist/*.whl dist/*.tar.gz
```

To exercise an installed wheel, create a clean supported Python environment,
install the wheel, and run:

```bash
python tools/release/fresh_wheel_smoke.py
```

The smoke test creates temporary analytical mesh and CAD inputs, checks all
supported input families, validates the recommended signature and field
exports, and removes its temporary working directory.
