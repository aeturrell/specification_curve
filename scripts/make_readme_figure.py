"""Rebuild the specification curve shown in the README quickstart.

Runs the same code as the README's quickstart and writes the figure to
``docs/quickstart.svg``. Run it after changing the plot styling or the example
data so the README image stays in step with the code:

    uv run python scripts/make_readme_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import specification_curve as sc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "quickstart.svg"


def main() -> None:
    """Fit the quickstart specification curve and save it as an SVG."""
    # Keep this in step with the Quickstart code block in README.md.
    df = sc.load_example_panel()
    sco = sc.SpecificationCurve(
        df,
        y="log(sales)",
        x="treated",
        controls=["size", "leverage", "rd"],
        fe=[None, "firm", "firm + year"],
        vcov=["hetero", {"CRV1": "firm"}],
    )
    sco.fit()

    # A fixed hash salt and no date metadata make the SVG byte-for-byte
    # reproducible, so it only shows up in a diff when the figure changes.
    with plt.rc_context({"svg.hashsalt": "specification_curve"}):
        fig, _ = sco.plot()
        fig.savefig(OUTPUT, bbox_inches="tight", metadata={"Date": None})
    plt.close("all")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
