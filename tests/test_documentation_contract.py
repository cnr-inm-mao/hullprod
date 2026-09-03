"""Keep the concise public documentation aligned with executable v1 behavior."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from hullprod.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = (
    "getting-started.md",
    "metrics.md",
    "outputs.md",
    "validity-and-provenance.md",
    "python-api.md",
    "experimental.md",
)


def test_readme_links_every_public_guide_and_leads_with_root_cli() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.index("pip install hullprod") < readme.index("hullprod myvessel.iges")
    assert "hullprod myvessel.stl" in readme
    for name in PUBLIC_DOCS:
        assert f"docs/{name}" in readme
        assert (ROOT / "docs" / name).is_file()


def test_documented_primary_options_are_real_parser_options() -> None:
    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions  # argparse exposes actions for introspection.
        for option in action.option_strings
    }
    assert {"--out", "--lref", "--overwrite", "--no-plots", "--experimental"} <= (
        option_strings
    )
    parsed = parser.parse_args(
        [
            "vessel.iges",
            "--out",
            "results",
            "--lref",
            "142.0",
            "--overwrite",
            "--no-plots",
        ]
    )
    assert parsed.geometry == Path("vessel.iges")
    assert parsed.out == Path("results")
    assert parsed.lref == 142.0
    assert parsed.overwrite and parsed.no_plots


def test_public_docs_have_current_reference_and_metric_contract() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8") + "\n" + "\n".join(
        (ROOT / "docs" / name).read_text(encoding="utf-8") for name in PUBLIC_DOCS
    )
    assert "axis_aligned_bounding_box_diagonal" not in text
    assert "auto_principal_span" in text
    assert "universal scalar" in text
    assert "fabrication-cost" in text
    for key in (
        "I_D",
        "I_D_plus",
        "I_D_minus",
        "a_C_flat",
        "a_C_single",
        "a_C_elliptic",
        "a_C_saddle",
    ):
        assert key in text


def test_bundled_examples_are_installed_package_resources() -> None:
    examples = files("hullprod.examples")
    assert examples.joinpath("simple_sphere.iges").is_file()
    assert examples.joinpath("simple_sphere.stl").is_file()
