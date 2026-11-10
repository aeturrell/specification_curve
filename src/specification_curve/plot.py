"""Drawing a specification curve from a results frame."""

from typing import Any, Mapping, Optional, Sequence, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D

from ._pyfixest import normalise

STYLE: dict[str, Any] = {
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
}

SIGNIFICANCE_COLORS: dict[str, str] = {
    "positive": "blue",
    "negative": "red",
    "insignificant": "black",
}
COLOR_OFF = "#E6E6E6"
DENSE_THRESHOLD = 160

# Dimensions that can be shown under the curve, in order, with their titles.
PANELS: dict[str, str] = {
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

Selector = Union[int, str, Mapping[str, Any]]
ColorSpec = Union[Mapping[str, Any], Sequence[Any], None]


def _model_labels(results: pd.DataFrame) -> pd.Series:
    """Labels each row's estimator, including its family or quantile."""
    labels = results["model"].astype(str)
    family = results["family"].map(lambda v: "" if pd.isna(v) else f" ({v})")
    quantile = results["quantile"].map(lambda v: "" if pd.isna(v) else f" (q = {v:g})")
    return labels + family + quantile


def _dimension_values(results: pd.DataFrame, name: str) -> pd.Series:
    """The value each row takes on a single-valued dimension, as text."""
    values = _model_labels(results) if name == "model" else results[name]
    return values.map(lambda v: "none" if pd.isna(v) else str(v))


def _indicators(
    results: pd.DataFrame, panels: Optional[Sequence[str]]
) -> dict[str, pd.DataFrame]:
    """For each panel, a table of which values each row uses."""
    names = list(PANELS) if panels is None else list(panels)
    unknown = [name for name in names if name not in PANELS]
    if unknown:
        raise ValueError(f"Unknown panel(s) {unknown}; choose from {list(PANELS)}")
    tables: dict[str, pd.DataFrame] = {}
    for name in names:
        if name == "controls":
            terms = sorted({term for row in results["controls"] for term in row})
            table = pd.DataFrame(
                {term: [term in row for row in results["controls"]] for term in terms},
                index=results.index,
            )
            if terms:
                tables[name] = table
            continue
        values = _dimension_values(results, name)
        levels = list(dict.fromkeys(values))
        if len(levels) > 1 or panels is not None:
            tables[name] = pd.DataFrame(
                {level: values == level for level in levels}, index=results.index
            )
    return tables


def _ci_columns(
    results: pd.DataFrame, ci_levels: Optional[Sequence[float]]
) -> list[tuple[str, str]]:
    """Pairs of interval columns to draw, narrowest level first."""
    stored = {
        float(str(c).removeprefix("ci_low_")) / 100: str(c)
        for c in results.columns
        if str(c).startswith("ci_low_")
    }
    levels = sorted(stored) if ci_levels is None else sorted(ci_levels)
    missing = [
        level for level in levels if not any(np.isclose(level, s) for s in stored)
    ]
    if missing:
        raise ValueError(
            f"Confidence level(s) {missing} are not in the results; stored levels "
            f"are {sorted(stored)}. Choose them with fit(ci_levels=...)."
        )
    lows = [
        next(col for s, col in stored.items() if np.isclose(level, s))
        for level in levels
    ]
    return [(low, low.replace("ci_low_", "ci_high_")) for low in lows]


def _sort(ok: pd.DataFrame, sort_by: Union[str, Sequence[str]]) -> pd.DataFrame:
    """Orders the specifications for plotting."""
    columns = [sort_by] if isinstance(sort_by, str) else list(sort_by)
    missing = [column for column in columns if column not in ok.columns]
    if missing:
        raise ValueError(f"Cannot sort by {missing}: not a column of the results")
    if "estimate" not in columns:
        columns.append("estimate")
    keys = pd.DataFrame(index=ok.index)
    for position, column in enumerate(columns):
        values = ok[column]
        numeric = pd.api.types.is_numeric_dtype(values)
        keys[position] = values if numeric else values.map(str)
    order = keys.sort_values(list(keys.columns), kind="stable").index
    return ok.loc[order].reset_index(drop=True)


def _significance_colors(
    ok: pd.DataFrame, alpha: float, colors: ColorSpec
) -> tuple[np.ndarray, dict[str, Any]]:
    """Colours each specification by the sign and significance of its estimate."""
    palette: dict[str, Any] = dict(SIGNIFICANCE_COLORS)
    if colors is not None:
        if not isinstance(colors, Mapping) or not set(colors) <= set(palette):
            raise ValueError(
                'With color_by="significance", colors must be a dict with keys from '
                f"{list(palette)}"
            )
        palette.update(colors)
    estimates = ok["estimate"].to_numpy(dtype=float)
    significant = ok["pvalue"].to_numpy(dtype=float) < alpha
    kinds = np.where(
        significant & (estimates > 0),
        "positive",
        np.where(significant & (estimates < 0), "negative", "insignificant"),
    )
    out = np.empty(len(ok), dtype=object)
    out[:] = [palette[kind] for kind in kinds]
    return out, {}


def _dimension_colors(
    ok: pd.DataFrame, color_by: str, colors: ColorSpec
) -> tuple[np.ndarray, dict[str, Any]]:
    """Colours each specification by the value it takes on one dimension."""
    allowed = [name for name in PANELS if name != "controls"]
    if color_by not in allowed:
        raise ValueError(f'color_by must be "significance" or one of {allowed}')
    values = _dimension_values(ok, color_by)
    levels = list(dict.fromkeys(values))
    if colors is None:
        cmap = mpl.colormaps["tab10"]
        legend = {level: cmap(i % 10) for i, level in enumerate(levels)}
    elif isinstance(colors, Mapping):
        missing = [level for level in levels if level not in colors]
        if missing:
            raise ValueError(f"colors has no entry for {missing}")
        legend = {level: colors[level] for level in levels}
    else:
        if len(colors) < len(levels):
            raise ValueError(
                f"colors has {len(colors)} entries but {color_by!r} takes "
                f"{len(levels)} values"
            )
        legend = dict(zip(levels, colors))
    out = np.empty(len(ok), dtype=object)
    out[:] = [legend[value] for value in values]
    return out, legend


def _term_set(value: Any) -> set[str]:
    """Normalised terms of a single term or a collection of terms."""
    terms = [value] if isinstance(value, str) else list(value)
    return {normalise(str(term)) for term in terms}


def _matches(ok: pd.DataFrame, selector: Selector) -> np.ndarray:
    """Positions of the specifications that a highlight selector picks out."""
    if isinstance(selector, str):
        try:
            return ok.index.get_indexer(ok.query(selector).index)
        except Exception as err:
            raise ValueError(f"The highlight query {selector!r} failed: {err}") from err
    if isinstance(selector, Mapping):
        unknown = [key for key in selector if key not in ok.columns]
        if unknown:
            raise ValueError(f"highlight refers to {unknown}: not in the results")
        keep = np.ones(len(ok), dtype=bool)
        for key, wanted in selector.items():
            if key in ("controls", "always"):
                target = _term_set(wanted)
                keep &= np.array(
                    [{normalise(t) for t in row} == target for row in ok[key]]
                )
            elif wanted is None:
                keep &= ok[key].isna().to_numpy()
            else:
                keep &= (ok[key] == wanted).to_numpy(dtype=bool, na_value=False)
        return np.flatnonzero(keep)
    return np.flatnonzero((ok["spec_id"] == selector).to_numpy())


def _highlighted(
    ok: pd.DataFrame, highlight: Union[Selector, Sequence[Selector], None]
) -> list[int]:
    """Position of each highlighted specification; each selector must match one."""
    if highlight is None:
        return []
    single = isinstance(highlight, (int, np.integer, str, Mapping))
    selectors: list[Any] = [highlight] if single else list(highlight)  # type: ignore[arg-type]
    positions = []
    for selector in selectors:
        found = _matches(ok, selector)
        if len(found) != 1:
            raise ValueError(
                f"highlight={selector!r} matches {len(found)} successfully estimated "
                "specifications; each highlight must match exactly one"
            )
        positions.append(int(found[0]))
    return positions


def _label(labels: Optional[Mapping[str, str]], text: str) -> str:
    """Display text for a dimension or value."""
    return text if labels is None else labels.get(text, text)


def _title(labels: Optional[Mapping[str, str]], dimension: str) -> str:
    """Display title for a dimension's panel or legend."""
    return (
        PANELS[dimension]
        if labels is None
        else labels.get(dimension, PANELS[dimension])
    )


def plot_curve(
    results: pd.DataFrame,
    null_curves: Optional[pd.DataFrame] = None,
    *,
    alpha: float = 0.05,
    ci_levels: Optional[Sequence[float]] = None,
    sort_by: Union[str, Sequence[str]] = "estimate",
    color_by: str = "significance",
    panels: Optional[Sequence[str]] = None,
    labels: Optional[Mapping[str, str]] = None,
    highlight: Union[Selector, Sequence[Selector], None] = None,
    show_median: bool = True,
    show_nobs: bool = False,
    colors: ColorSpec = None,
    figsize: Optional[tuple[float, float]] = None,
    fig: Union[Figure, SubFigure, None] = None,
    style: bool = True,
) -> tuple[Union[Figure, SubFigure], dict[str, Axes]]:
    """Draws a specification curve.

    The top panel shows the focal estimate of every specification with its
    confidence interval(s). Below it, one panel per dimension that varies shows
    which choice each specification makes. Specifications that could not be
    estimated are left out.

    The function works on any data frame with the columns of
    ``SpecificationCurve.results``, so results that were saved, filtered or
    combined from several curves can be drawn too. It does not call
    ``plt.show()`` and does not change matplotlib's global settings.

    Args:
        results (pd.DataFrame): Results as produced by ``SpecificationCurve.fit``.
        null_curves (Optional[pd.DataFrame]): Quantiles of the curve under the
            null, as produced by ``SpecificationCurve.fit_null``. Can only be
            shown when sorting by estimate.
        alpha (float): Significance level used when colouring by significance.
        ci_levels (Optional[Sequence[float]]): Which of the stored confidence
            levels to draw. By default, all of them.
        sort_by (Union[str, Sequence[str]]): Order of the specifications:
            "estimate", another column of the results, or a list of columns.
            Ties are ordered by estimate.
        color_by (str): "significance" colours estimates by sign and
            significance. A dimension ("y", "x", "fe", "iv", "vcov", "model",
            "weights" or "subset") colours by the value of that dimension and
            adds a legend.
        panels (Optional[Sequence[str]]): Which dimensions to show under the
            curve, in order. By default, every dimension that varies.
        labels (Optional[Mapping[str, str]]): Replacement display text for panel
            titles (keyed by dimension, e.g. ``{"fe": "Fixed effects"}``) and for
            values (keyed by the value, e.g. ``{"firm + year": "Firm and year"}``).
        highlight (Union[Selector, Sequence[Selector], None]): A preferred
            specification to mark and annotate, given as a ``spec_id``, a dict of
            column values (e.g. ``{"y": "log(wage)", "controls": ["age"], "fe":
            None}``) or a query string. It must match exactly one specification.
            Pass a list to mark several.
        show_median (bool): Draw a line at the median estimate.
        show_nobs (bool): Add a panel with the number of observations used by
            each specification.
        colors (ColorSpec): Colours to use. When colouring by significance, a
            dict with any of the keys "positive", "negative", "insignificant".
            When colouring by a dimension, a list of colours or a dict keyed by
            the dimension's values.
        figsize (Optional[tuple[float, float]]): Figure size in inches. By
            default, chosen from the number of panels.
        fig (Union[Figure, SubFigure, None]): An existing figure or subfigure to
            draw into. By default a new figure is created.
        style (bool): Use this package's fonts and sizes for the figure.

    Returns:
        tuple[Union[Figure, SubFigure], dict[str, Axes]]: The figure and its
            axes keyed by panel name: "curve", then the dimensions shown, then
            "nobs" if requested.

    Raises:
        ValueError: If no specification was estimated successfully, or an
            argument does not fit the results.
    """
    ok = results[results["error"].isna()]
    if ok.empty:
        raise ValueError("Nothing to plot: no specification was estimated successfully")
    by_estimate = sort_by == "estimate" or list(sort_by) == ["estimate"]
    if null_curves is not None and not by_estimate:
        raise ValueError(
            "The curve under the null can only be shown with sort_by='estimate'"
        )
    ok = _sort(ok, sort_by)
    intervals = _ci_columns(ok, ci_levels)
    tables = _indicators(ok, panels)
    if color_by == "significance":
        point_colors, legend = _significance_colors(ok, alpha, colors)
    else:
        point_colors, legend = _dimension_colors(ok, color_by, colors)
    marked = _highlighted(ok, highlight)

    heights = [5.0] + [0.35 * table.shape[1] + 0.3 for table in tables.values()]
    names = ["curve", *tables]
    if show_nobs:
        heights.append(1.0)
        names.append("nobs")
    with mpl.rc_context(STYLE if style else {}):
        if fig is None:
            size = (12.0, 4.5 + sum(heights[1:])) if figsize is None else figsize
            fig = plt.figure(figsize=size)
        elif figsize is not None:
            raise ValueError("Give figsize or fig, not both")
        axes_grid = fig.subplots(
            nrows=len(heights),
            ncols=1,
            sharex=True,
            squeeze=False,
            gridspec_kw={"height_ratios": heights, "hspace": 0.12},
        )
        axes = dict(zip(names, axes_grid[:, 0]))
        _draw_curve(
            axes["curve"], ok, intervals, point_colors, null_curves, show_median, marked
        )
        if legend:
            handles = [
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    color=color,
                    label=_label(labels, level),
                )
                for level, color in legend.items()
            ]
            axes["curve"].legend(
                handles=handles,
                title=_title(labels, color_by),
                frameon=False,
                fontsize=9,
                title_fontsize=9,
                loc="best",
            )
        n_failed = len(results) - len(ok)
        if n_failed:
            axes["curve"].set_title(
                f"{n_failed} specification(s) failed and are not shown",
                fontsize=9,
                loc="right",
            )
        for name, table in tables.items():
            _draw_panel(axes[name], table, point_colors, _title(labels, name), labels)
        if show_nobs:
            axes["nobs"].bar(
                np.arange(len(ok)),
                ok["nobs"].to_numpy(dtype=float),
                color="gray",
                width=0.8,
            )
            axes["nobs"].set_ylabel("N", fontsize=10)
        for ax in axes.values():
            ax.set_xticks([])
            ax.set_xlim(-1, len(ok))
            for side in ("right", "top", "bottom"):
                ax.spines[side].set_visible(False)
    return fig, axes


def _draw_curve(
    ax: Axes,
    ok: pd.DataFrame,
    intervals: Sequence[tuple[str, str]],
    colors: np.ndarray,
    null_curves: Optional[pd.DataFrame],
    show_median: bool,
    marked: Sequence[int],
) -> None:
    """Draws the estimates, intervals, median, null curve and highlights."""
    n_specs = len(ok)
    dense = n_specs > DENSE_THRESHOLD
    positions = np.arange(n_specs)
    estimates = ok["estimate"].to_numpy(dtype=float)
    # widest interval first and lightest, so narrower ones are drawn over it
    for layer, (low, high) in enumerate(reversed(intervals)):
        ax.vlines(
            positions,
            ok[low].to_numpy(dtype=float),
            ok[high].to_numpy(dtype=float),
            colors=list(colors),
            alpha=min(0.2 + 0.25 * layer, 0.8),
            linewidth=(1 if dense else 2) + 0.8 * layer,
            zorder=1,
            gid="interval",
        )
    ax.scatter(
        positions,
        estimates,
        c=list(colors),
        s=4 if dense else 16,
        alpha=0.7,
        zorder=2,
        gid="estimate",
    )
    if show_median:
        median = float(np.median(estimates))
        ax.axhline(median, color="k", linewidth=0.5, dashes=(12, 5))
        ax.annotate(
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
        ax.fill_between(
            ranks,
            null_curves[columns[0]].to_numpy(dtype=float),
            null_curves[columns[-1]].to_numpy(dtype=float),
            color="gray",
            alpha=0.15,
            linewidth=0,
            zorder=0,
        )
        middle = null_curves[columns[len(columns) // 2]].to_numpy(dtype=float)
        ax.plot(ranks, middle, color="gray", linestyle="dashdot", linewidth=1)
        ax.annotate(
            "Median under null",
            xy=(0.6, float(np.interp(0.6 * (n_specs - 1), ranks, middle))),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=10,
            color="gray",
        )
    for position in marked:
        value = estimates[position]
        text = f"{value:+.3g}"
        if len(marked) == 1:
            text = f"Preferred specification: {text}"
        ax.scatter(
            [position],
            [value],
            s=90,
            facecolors="none",
            edgecolors="k",
            linewidths=1.2,
            zorder=3,
            gid="highlight",
        )
        ax.annotate(
            text,
            xy=(position, value),
            xytext=(-30, 45) if position > n_specs / 2 else (30, 45),
            textcoords="offset points",
            ha="right" if position > n_specs / 2 else "left",
            fontsize=10,
            arrowprops={"arrowstyle": "->", "color": "0.3"},
            zorder=4,
        )
    ax.set_ylabel("Coefficient")


def _draw_panel(
    ax: Axes,
    table: pd.DataFrame,
    colors: np.ndarray,
    title: str,
    labels: Optional[Mapping[str, str]],
) -> None:
    """Draws the markers showing which choice each specification makes."""
    n_specs = len(table)
    dense = n_specs > DENSE_THRESHOLD
    marker = "|" if dense else "s"
    size = 30 if dense else float(np.clip((450 / n_specs) ** 2, 12, 120))
    on = table.to_numpy(dtype=bool).T
    rows, cols = np.nonzero(on)
    if not dense:
        off_rows, off_cols = np.nonzero(~on)
        ax.scatter(off_cols, off_rows, marker=marker, s=size, c=COLOR_OFF, linewidths=0)
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
    ax.set_yticklabels([_label(labels, str(column)) for column in table.columns])
    ax.set_ylim(table.shape[1] - 0.5, -0.5)
    ax.set_ylabel(title, rotation=-90, labelpad=14, fontsize=10)
    ax.yaxis.set_label_position("right")
