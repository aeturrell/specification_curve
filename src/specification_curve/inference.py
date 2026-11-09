"""Inference on a specification curve under the null of no effect."""

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from ._pyfixest import VcovSpec, evaluate_term
from .engine import run_grid
from .formula import render
from .grid import GRID_COLUMNS

NULL_QUANTILES: tuple[float, ...] = (0.025, 0.5, 0.975)


def _null_grid(
    results: pd.DataFrame, data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adds one outcome column per model with the estimated effect removed."""
    y_values = {term: evaluate_term(term, data) for term in results["y"].unique()}
    x_values = {term: evaluate_term(term, data) for term in results["x"].unique()}
    grid = results.copy()
    new_columns: dict[str, pd.Series] = {}
    names: dict[tuple[Any, ...], str] = {}
    outcome_names = []
    for row in results.to_dict("records"):
        key = (row["y"], row["x"], row["estimate"])
        if key not in names:
            names[key] = f"sc_null_y{len(names)}"
            new_columns[names[key]] = (
                y_values[row["y"]] - row["estimate"] * x_values[row["x"]]
            )
        outcome_names.append(names[key])
    grid["y"] = outcome_names
    grid["formula"] = [
        render(row["y"], row["x"], row["controls"], row["always"], row["fe"], row["iv"])
        for row in grid.to_dict("records")
    ]
    augmented = pd.concat([data, pd.DataFrame(new_columns, index=data.index)], axis=1)
    return grid, augmented


def _share_p(null: np.ndarray, observed: float) -> float:
    """Share of resamples at least as large as observed, with the +1 correction."""
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def fit_null(
    results: pd.DataFrame,
    data: pd.DataFrame,
    *,
    vcov_specs: Mapping[str, VcovSpec],
    subsets: Mapping[str, Optional[str]],
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    common_sample: bool = True,
    fit_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstraps the specification curve under the null of no effect.

    For each specification the estimated effect is removed from the outcome
    (``y* = y - b * x``), which imposes the null. Rows are then resampled with
    replacement, using the same draw for every specification, and all
    specifications are re-estimated on ``y*``.

    Args:
        results (pd.DataFrame): Results of the fitted curve.
        data (pd.DataFrame): The data the curve was fitted on.
        vcov_specs (Mapping[str, VcovSpec]): Maps ``vcov`` labels to pyfixest values.
        subsets (Mapping[str, Optional[str]]): Maps subset names to queries.
        n_boot (int): Number of bootstrap resamples.
        alpha (float): Significance level used for the shares of significant
            estimates.
        seed (Optional[int]): Seed for the resampling.
        common_sample (bool): As used when fitting the curve.
        fit_kwargs (Optional[Mapping[str, Any]]): Passed to the pyfixest estimators.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The quantiles of the sorted curve under
            the null (one row per rank), and a summary of the observed statistics
            with their p-values.

    Raises:
        ValueError: If any specification uses an estimator other than "feols", or
            no specification was estimated successfully.
    """
    if (results["model"] != "feols").any():
        raise ValueError(
            "Inference under the null is only available when every specification "
            'uses model="feols": removing the estimated effect from the outcome '
            "does not impose the null in non-linear models."
        )
    ok = results[results["error"].isna()].reset_index(drop=True)
    if ok.empty:
        raise ValueError("No specification was estimated successfully")
    grid, augmented = _null_grid(ok, data)
    grid["vcov"] = ok["vcov"].map(
        lambda label: label if label in vcov_specs else "default"
    )
    specs = {**vcov_specs, "default": None}

    rng = np.random.default_rng(seed)
    n_rows = len(augmented)
    estimates = np.full((n_boot, len(grid)), np.nan)
    pvalues = np.full((n_boot, len(grid)), np.nan)
    for b in range(n_boot):
        draw = augmented.iloc[rng.integers(0, n_rows, size=n_rows)]
        boot, _ = run_grid(
            grid[GRID_COLUMNS],
            draw.reset_index(drop=True),
            vcov_specs=specs,
            subsets=subsets,
            common_sample=common_sample,
            ci_levels=(),
            progress=False,
            warn=False,
            fit_kwargs=fit_kwargs,
        )
        estimates[b] = boot["estimate"].to_numpy(dtype=float)
        pvalues[b] = boot["pvalue"].to_numpy(dtype=float)

    null_curves = pd.DataFrame(
        np.nanquantile(np.sort(estimates, axis=1), NULL_QUANTILES, axis=0).T,
        columns=list(NULL_QUANTILES),
    )
    null_curves.index.name = "rank"

    observed = ok["estimate"].to_numpy(dtype=float)
    observed_sig = ok["pvalue"].to_numpy(dtype=float) < alpha
    significant = pvalues < alpha
    obs_median = float(np.median(observed))
    obs_positive = float(np.mean(observed_sig & (observed > 0)))
    obs_negative = float(np.mean(observed_sig & (observed < 0)))
    null_median = np.nanmedian(estimates, axis=1)
    null_positive = np.mean(significant & (estimates > 0), axis=1)
    null_negative = np.mean(significant & (estimates < 0), axis=1)
    summary = pd.DataFrame(
        {
            "observed": [obs_median, obs_positive, obs_negative],
            "p-value": [
                _share_p(np.abs(null_median), abs(obs_median)),
                _share_p(null_positive, obs_positive),
                _share_p(null_negative, obs_negative),
            ],
        },
        index=[
            "median estimate",
            "share significant and positive",
            "share significant and negative",
        ],
    )
    return null_curves, summary
