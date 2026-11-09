"""Drawing a specification curve from a results frame."""

from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

STYLE: dict[str, object] = {
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
}

COLOR_POSITIVE = "blue"
COLOR_NEGATIVE = "red"
COLOR_INSIGNIFICANT = "black"
COLOR_OFF = "#E6E6E6"
DENSE_THRESHOLD = 160

# Dimensions shown under the curve, in order, with their panel titles.
_PANELS: dict[str, str] = {
    "y": "y",
    "x": "x",
    "controls": "controls",
    "fe": "fixed effects",
    "iv": "instruments",
    "vcov": "std. errors",
    "model": "model",
    "weights": "weights",
    "subset": "sample",
}


def _model_labels(results: pd.DataFrame) -> pd.Series:
    """Labels each row's estimator, including its family or quantile."""
    labels = results["model"].astype(str)
    family = results["family"].map(lambda v: "" if pd.isna(v) else f" ({v})")
    quantile = results["quantile"].map(lambda v: "" if pd.isna(v) else f" (q = {v:g})")
    return labels + family + quantile


def _indicators(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """For each dimension that varies, a table of which values each row uses."""
    panels: dict[str, pd.DataFrame] = {}
    for name in _PANELS:
        if name == "controls":
            terms = sorted({term for row in results["controls"] for term in row})
            table = pd.DataFrame(
                {term: [term in row for row in results["controls"]] for term in terms},
                index=results.index,
            )
            if terms:
                panels[name] = table
            continue
        values = _model_labels(results) if name == "model" else results[name]
        values = values.map(lambda v: "none" if pd.isna(v) else str(v))
        levels = list(dict.fromkeys(values))
        if len(levels) > 1:
            panels[name] = pd.DataFrame(
                {level: values == level for level in levels}, index=results.index
            )
    return panels


def _ci_columns(results: pd.DataFrame) -> list[tuple[str, str]]:
    """Pairs of interval columns in the results, widest level last."""
    lows = [c for c in results.columns if str(c).startswith("ci_low_")]
    lows.sort(key=lambda c: float(str(c).removeprefix("ci_low_")))
    return [(low, str(low).replace("ci_low_", "ci_high_")) for low in lows]


def plot_curve(
    results: pd.DataFrame,
    null_curves: Optional[pd.DataFrame] = None,
    *,
    alpha: float = 0.05,
    style: bool = True,
) -> tuple[Figure, list[Axes]]:
    """Draws a specification curve.

    The top panel shows the focal estimate of every specification, sorted from
    smallest to largest, with confidence intervals. Estimates that are
    significant at ``alpha`` are coloured by sign. Below it, one panel per
    dimension that varies shows which choice each specification makes.

    Args:
        results (pd.DataFrame): Results as produced by ``SpecificationCurve.fit``.
        null_curves (Optional[pd.DataFrame]): Quantiles of the curve under the
            null, as produced by ``SpecificationCurve.fit_null``.
        alpha (float): Significance level used for colouring.
        style (bool): Use this package's fonts and sizes for the figure.

    Returns:
        tuple[Figure, list[Axes]]: The figure and its axes, top panel first.

    Raises:
        ValueError: If no specification was estimated successfully.
    """
    ok = results[results["error"].isna()].sort_values("estimate", kind="stable")
    ok = ok.reset_index(drop=True)
    if ok.empty:
        raise ValueError("Nothing to plot: no specification was estimated successfully")
    with mpl.rc_context(STYLE if style else {}):
        return _draw(ok, null_curves, alpha, n_failed=len(results) - len(ok))


def _draw(
    ok: pd.DataFrame, null_curves: Optional[pd.DataFrame], alpha: float, n_failed: int
) -> tuple[Figure, list[Axes]]:
    """Draws the figure for the successful, sorted specifications."""
    panels = _indicators(ok)
    n_specs = len(ok)
    positions = np.arange(n_specs)
    dense = n_specs > DENSE_THRESHOLD
    heights = [5.0] + [0.35 * table.shape[1] + 0.3 for table in panels.values()]
    fig, axes_grid = plt.subplots(
        nrows=len(heights),
        ncols=1,
        sharex=True,
        squeeze=False,
        figsize=(12, 4.5 + sum(heights[1:])),
        gridspec_kw={"height_ratios": heights, "hspace": 0.12},
    )
    axes: list[Axes] = list(axes_grid[:, 0])
    top = axes[0]

    estimates = ok["estimate"].to_numpy(dtype=float)
    significant = ok["pvalue"].to_numpy(dtype=float) < alpha
    colors = np.where(
        significant & (estimates > 0),
        COLOR_POSITIVE,
        np.where(significant & (estimates < 0), COLOR_NEGATIVE, COLOR_INSIGNIFICANT),
    )
    intervals = _ci_columns(ok)
    for layer, (low, high) in enumerate(reversed(intervals)):
        top.vlines(
            positions,
            ok[low].to_numpy(dtype=float),
            ok[high].to_numpy(dtype=float),
            colors=list(colors),
            alpha=0.2 + 0.2 * layer,
            linewidth=1 if dense else 2,
            zorder=1,
        )
    top.scatter(
        positions, estimates, c=list(colors), s=4 if dense else 16, alpha=0.7, zorder=2
    )
    median = float(np.median(estimates))
    top.axhline(median, color="k", linewidth=0.5, dashes=(12, 5))
    top.annotate(
        "Median",
        xy=(0.02, median),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=10,
        alpha=0.7,
    )
    if null_curves is not None:
        ranks = np.linspace(0, n_specs - 1, len(null_curves))
        columns = list(null_curves.columns)
        top.fill_between(
            ranks,
            null_curves[columns[0]].to_numpy(dtype=float),
            null_curves[columns[-1]].to_numpy(dtype=float),
            color="gray",
            alpha=0.15,
            linewidth=0,
            zorder=0,
        )
        middle = null_curves[columns[len(columns) // 2]].to_numpy(dtype=float)
        top.plot(ranks, middle, color="gray", linestyle="dashdot", linewidth=1)
        top.annotate(
            "Median under null",
            xy=(0.6, float(np.interp(0.6 * (n_specs - 1), ranks, middle))),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=10,
            color="gray",
        )
    top.set_ylabel("Coefficient")
    if n_failed:
        top.set_title(
            f"{n_failed} specification(s) failed and are not shown",
            fontsize=9,
            loc="right",
        )

    marker = "|" if dense else "s"
    size = 30 if dense else float(np.clip((450 / n_specs) ** 2, 12, 120))
    for ax, (name, table) in zip(axes[1:], panels.items()):
        on = table.to_numpy(dtype=bool).T
        rows, cols = np.nonzero(on)
        if not dense:
            off_rows, off_cols = np.nonzero(~on)
            ax.scatter(
                off_cols, off_rows, marker=marker, s=size, c=COLOR_OFF, linewidths=0
            )
        ax.scatter(
            cols,
            rows,
            marker=marker,
            s=size,
            c=list(colors[cols]),
            alpha=0.5,
            linewidths=1 if dense else 0,
        )
        ax.set_yticks(range(table.shape[1]))
        ax.set_yticklabels([str(label) for label in table.columns])
        ax.set_ylim(table.shape[1] - 0.5, -0.5)
        ax.set_ylabel(_PANELS[name], rotation=-90, labelpad=14, fontsize=10)
        ax.yaxis.set_label_position("right")
    for ax in axes:
        ax.set_xticks([])
        ax.set_xlim(-1, n_specs)
        for side in ("right", "top", "bottom"):
            ax.spines[side].set_visible(False)
    return fig, axes
