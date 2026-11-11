"""Inference on a specification curve under the null of no effect.

The tests follow Simonsohn, Simmons and Nelson (2020), "Specification curve
analysis", Nature Human Behaviour 4, 1208-1214. The Bayesian bootstrap is due to
Rubin (1981), "The Bayesian bootstrap", Annals of Statistics 9(1), 130-134; see
also Andrews, Moon and Shapiro (2026), "In praise of the Bayesian bootstrap",
Journal of Economic Perspectives 40(4), 137-158.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from tqdm.auto import tqdm

from ._pyfixest import VcovSpec, evaluate_term
from .engine import run_grid
from .formula import render
from .grid import GRID_COLUMNS, variables_in

NULL_QUANTILES: tuple[float, ...] = (0.025, 0.5, 0.975)
STATISTICS: tuple[str, ...] = ("median", "share significant", "Stouffer Z")
METHODS: tuple[str, ...] = ("auto", "bootstrap", "bayesian-bootstrap", "permutation")
#: Methods that impose the null by removing the estimated effect from the outcome.
NULL_OUTCOME_METHODS: tuple[str, ...] = ("bootstrap", "bayesian-bootstrap")
#: Prefix of the weight columns added for the Bayesian bootstrap.
WEIGHT_PREFIX = "sc_bb_w"
ALTERNATIVES: tuple[str, ...] = ("two-sided", "greater", "less")


@dataclass(frozen=True)
class NullResult:
    """Results of inference under the null.

    Attributes:
        curves (pd.DataFrame): Quantiles of the sorted curve under the null, one
            row per rank.
        summary (pd.DataFrame): Observed test statistics and their p-values.
        draws (Optional[pd.DataFrame]): Estimates from every resample (rows) and
            specification (columns, by ``spec_id``), if they were kept.
        method (str): The resampling method used.
        n_boot (int): Number of resamples that were used.
        n_dropped (int): Number of resamples left out because too few
            specifications could be estimated.
    """

    curves: pd.DataFrame
    summary: pd.DataFrame
    draws: Optional[pd.DataFrame]
    method: str
    n_boot: int
    n_dropped: int


def curve_statistics(
    estimates: np.ndarray,
    pvalues: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> np.ndarray:
    """Computes the three test statistics of a specification curve.

    The statistics are the median estimate, the share of estimates that are
    significant, and Stouffer's Z (the sum of the estimates' z-values divided by
    the square root of their number). With ``alternative="two-sided"`` the share
    counts significant estimates of whichever sign has more of them; with
    "greater" or "less" it counts significant positive or negative estimates.

    Args:
        estimates (np.ndarray): Estimates, with specifications along the last axis.
        pvalues (np.ndarray): Two-sided p-values, the same shape as ``estimates``.
        alpha (float): Significance level for the share of significant estimates.
        alternative (str): "two-sided", "greater" or "less".

    Returns:
        np.ndarray: The three statistics along the last axis. Missing estimates
            are ignored.
    """
    estimates = np.asarray(estimates, dtype=float)
    pvalues = np.asarray(pvalues, dtype=float)
    valid = ~np.isnan(estimates) & ~np.isnan(pvalues)
    n_valid = np.maximum(valid.sum(axis=-1), 1)
    significant = valid & (pvalues < alpha)
    positive = (significant & (estimates > 0)).sum(axis=-1) / n_valid
    negative = (significant & (estimates < 0)).sum(axis=-1) / n_valid
    if alternative == "greater":
        share = positive
    elif alternative == "less":
        share = negative
    else:
        share = np.maximum(positive, negative)
    z_values = np.sign(estimates) * norm.isf(np.clip(pvalues, 1e-300, 1) / 2)
    stouffer = np.where(valid, z_values, 0.0).sum(axis=-1) / np.sqrt(n_valid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(np.where(valid, estimates, np.nan), axis=-1)
    return np.stack([median, share, stouffer], axis=-1)


def resampling_pvalues(
    observed: np.ndarray, null: np.ndarray, alternative: str = "two-sided"
) -> np.ndarray:
    """Share of resampled statistics at least as extreme as those observed.

    Args:
        observed (np.ndarray): The observed statistics.
        null (np.ndarray): The statistics in each resample, one row per resample.
        alternative (str): "two-sided", "greater" or "less".

    Returns:
        np.ndarray: One p-value per statistic, computed as
            ``(1 + count) / (1 + number of resamples)``.
    """
    if alternative == "greater":
        extreme = null >= observed
    elif alternative == "less":
        extreme = null <= observed
        # the share of significant negative estimates is large when extreme
        extreme[:, 1] = null[:, 1] >= observed[1]
    else:
        extreme = np.abs(null) >= np.abs(observed)
    return (1 + extreme.sum(axis=0)) / (1 + len(null))


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


def _units(
    data: pd.DataFrame, cluster: Optional[str], strata: Optional[str]
) -> list[list[np.ndarray]]:
    """Row positions of each resampling unit, grouped by stratum."""
    positions = np.arange(len(data))
    stratum_ids = (
        np.zeros(len(data), dtype=int)
        if strata is None
        else pd.factorize(data[strata])[0]
    )
    unit_ids = positions if cluster is None else pd.factorize(data[cluster])[0]
    out: list[list[np.ndarray]] = []
    for stratum in np.unique(stratum_ids):
        in_stratum = positions[stratum_ids == stratum]
        if cluster is None:
            out.append([np.array([p]) for p in in_stratum])
            continue
        frame = pd.Series(in_stratum).groupby(unit_ids[in_stratum], sort=False)
        out.append([group.to_numpy() for _, group in frame])
    return out


def bootstrap_rows(
    units: Sequence[Sequence[np.ndarray]], rng: np.random.Generator
) -> np.ndarray:
    """Draws row positions for one bootstrap resample.

    Within each stratum, units (rows or whole clusters) are drawn with
    replacement, as many as the stratum has.

    Args:
        units (Sequence[Sequence[np.ndarray]]): Row positions of each unit,
            grouped by stratum.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: Row positions of the resample.
    """
    drawn = []
    for stratum in units:
        picks = rng.integers(0, len(stratum), size=len(stratum))
        drawn.extend(stratum[pick] for pick in picks)
    return np.concatenate(drawn)


def bayesian_weights(
    units: Sequence[Sequence[np.ndarray]], n_rows: int, rng: np.random.Generator
) -> np.ndarray:
    """Draws row weights for one Bayesian bootstrap resample.

    Within each stratum, the units (rows or whole clusters) receive weights
    drawn from a flat Dirichlet distribution (Rubin, 1981), multiplied by the
    stratum's number of units so that each averages one. Every row of a cluster
    gets that cluster's weight. This is the smooth counterpart of
    ``bootstrap_rows``: a unit's weight plays the role of the number of times it
    is drawn, but no unit is ever left out entirely.

    Args:
        units (Sequence[Sequence[np.ndarray]]): Row positions of each unit,
            grouped by stratum.
        n_rows (int): Number of rows in the data.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: One positive weight per row.
    """
    weights = np.ones(n_rows)
    for stratum in units:
        sizes = [len(unit) for unit in stratum]
        # normalised standard exponentials are a flat Dirichlet draw
        draw = rng.standard_exponential(len(stratum))
        multipliers = draw * len(stratum) / draw.sum()
        weights[np.concatenate(stratum)] = np.repeat(multipliers, sizes)
    return weights


def _weighted_grid(
    grid: pd.DataFrame, data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Optional[str]]]:
    """Points each specification at a Bayesian bootstrap weight column.

    Returns the new grid, the data with placeholder weight columns, and a map
    from each new weight column to the user's weight column it multiplies (or
    None).
    """
    grid = grid.copy()
    sources: dict[str, Optional[str]] = {}
    names: dict[Optional[str], str] = {}
    for weight in grid["weights"].unique():
        source = None if pd.isna(weight) else str(weight)
        names[source] = f"{WEIGHT_PREFIX}{len(names)}"
        sources[names[source]] = source
    grid["weights"] = [
        names[None if pd.isna(weight) else str(weight)] for weight in grid["weights"]
    ]
    placeholders = pd.DataFrame(1.0, index=data.index, columns=list(sources))
    return grid, pd.concat([data, placeholders], axis=1), sources


def permute_rows(
    units: Sequence[Sequence[np.ndarray]], n_rows: int, rng: np.random.Generator
) -> np.ndarray:
    """Draws the source row of each row's focal values for one permutation.

    Within each stratum, the focal values are shuffled between units. With
    clusters, the clusters swap their (constant) focal values.

    Args:
        units (Sequence[Sequence[np.ndarray]]): Row positions of each unit,
            grouped by stratum.
        n_rows (int): Number of rows in the data.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: For each row, the position of the row whose focal values it
            receives.
    """
    source = np.arange(n_rows)
    for stratum in units:
        order = rng.permutation(len(stratum))
        for unit, donor in zip(stratum, order):
            # focal values are constant within a unit, so any donor row will do
            source[unit] = stratum[int(donor)][0]
    return source


def _one_resample(
    seed: np.random.SeedSequence,
    method: str,
    grid: pd.DataFrame,
    data: pd.DataFrame,
    units: Sequence[Sequence[np.ndarray]],
    focal_columns: Sequence[str],
    engine_kwargs: Mapping[str, Any],
    weight_sources: Mapping[str, Optional[str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Re-estimates the curve on one resample; returns estimates and p-values."""
    rng = np.random.default_rng(seed)
    if method == "bootstrap":
        draw = data.iloc[bootstrap_rows(units, rng)].reset_index(drop=True)
    elif method == "bayesian-bootstrap":
        weights = bayesian_weights(units, len(data), rng)
        draw = data.copy()
        for column, source in weight_sources.items():
            draw[column] = (
                weights if source is None else weights * data[source].to_numpy()
            )
    else:
        source = permute_rows(units, len(data), rng)
        draw = data.copy()
        for column in focal_columns:
            draw[column] = data[column].to_numpy()[source]
    results, _ = run_grid(
        grid, draw, ci_levels=(), progress=False, warn=False, **engine_kwargs
    )
    return (
        results["estimate"].to_numpy(dtype=float),
        results["pvalue"].to_numpy(dtype=float),
    )


def _check_design(
    data: pd.DataFrame,
    method: str,
    cluster: Optional[str],
    strata: Optional[str],
    focal_columns: Sequence[str],
) -> None:
    """Raises if the resampling design does not fit the data."""
    for name, column in (("cluster", cluster), ("strata", strata)):
        if column is None:
            continue
        if column not in data.columns:
            raise ValueError(f"{name}={column!r} is not a column of the data")
        if data[column].isna().any():
            raise ValueError(f"{name}={column!r} has missing values")
    if cluster is not None and strata is not None:
        if (data.groupby(cluster)[strata].nunique() > 1).any():
            raise ValueError(
                f"Each cluster ({cluster!r}) must lie within one stratum ({strata!r})"
            )
    if method == "permutation" and cluster is not None:
        varying = [
            column
            for column in focal_columns
            if (data.groupby(cluster)[column].nunique(dropna=False) > 1).any()
        ]
        if varying:
            raise ValueError(
                f"To permute by cluster, the focal variable(s) {varying} must be "
                f"constant within each {cluster!r}"
            )


def fit_null(
    results: pd.DataFrame,
    data: pd.DataFrame,
    *,
    vcov_specs: Mapping[str, VcovSpec],
    subsets: Mapping[str, Optional[str]],
    n_boot: int = 500,
    method: str = "auto",
    cluster: Optional[str] = None,
    strata: Optional[str] = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    seed: Optional[int] = None,
    n_jobs: int = 1,
    keep_draws: bool = False,
    min_valid: float = 0.5,
    progress: Optional[bool] = None,
    common_sample: bool = True,
    fit_kwargs: Optional[Mapping[str, Any]] = None,
) -> NullResult:
    """Tests a specification curve against the null hypothesis of no effect.

    The curve is re-estimated on many resamples in which the null holds, and
    three statistics of the observed curve (its median estimate, the share of
    significant estimates, and Stouffer's Z) are compared with their
    distribution across resamples.

    Two ways of making the null hold are available. ``"bootstrap"`` removes each
    specification's estimated effect from its outcome (``y* = y - b * x``) and
    then resamples rows with replacement; it applies to linear models.
    ``"bayesian-bootstrap"`` imposes the null in the same way but, instead of
    resampling rows, gives them random weights from a flat Dirichlet
    distribution (Rubin, 1981; Andrews, Moon and Shapiro, 2026). No row is ever
    left out, so resamples rarely fail to estimate. With ``"permutation"`` the
    focal variable is shuffled between rows, which breaks any link with the
    outcome; it applies to any estimator but is only valid if the focal variable
    is as good as randomly assigned (within ``strata``, if given). Permutation
    does not apply to instrumental-variable specifications, since shuffling the
    endogenous regressor alone breaks the first stage.

    Args:
        results (pd.DataFrame): Results of the fitted curve.
        data (pd.DataFrame): The data the curve was fitted on.
        vcov_specs (Mapping[str, VcovSpec]): Maps ``vcov`` labels to pyfixest values.
        subsets (Mapping[str, Optional[str]]): Maps subset names to queries.
        n_boot (int): Number of resamples.
        method (str): "bootstrap", "bayesian-bootstrap", "permutation" or
            "auto". "auto" uses the bootstrap if every specification uses
            "feols".
        cluster (Optional[str]): Column identifying clusters. Whole clusters are
            resampled (bootstrap), share a weight (Bayesian bootstrap) or swap
            their focal values (permutation).
        strata (Optional[str]): Column identifying strata. Resampling is done
            within each stratum.
        alpha (float): Significance level for the share of significant estimates.
        alternative (str): "two-sided", "greater" (the effect is positive) or
            "less" (the effect is negative).
        seed (Optional[int]): Seed for the resampling. Results do not depend on
            ``n_jobs``.
        n_jobs (int): Number of parallel workers.
        keep_draws (bool): Keep the estimates from every resample.
        min_valid (float): A resample is left out if fewer than this share of
            the specifications could be estimated in it.
        progress (Optional[bool]): Show a progress bar. By default one is shown
            when there are at least 20 resamples.
        common_sample (bool): As used when fitting the curve.
        fit_kwargs (Optional[Mapping[str, Any]]): Passed to the pyfixest estimators.

    Returns:
        NullResult: The curve under the null and the tests.

    Raises:
        ValueError: If the arguments are not valid, the method does not apply to
            the estimators used, no specification was estimated successfully, or
            no resample could be used.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}")
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {list(ALTERNATIVES)}")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    all_linear = bool((results["model"] == "feols").all())
    if method == "auto":
        if not all_linear:
            raise ValueError(
                'Not every specification uses model="feols", so the bootstrap '
                "under the null does not apply. If the focal variable is as good as "
                'randomly assigned, use method="permutation".'
            )
        method = "bootstrap"
    if method in NULL_OUTCOME_METHODS and not all_linear:
        raise ValueError(
            f'method="{method}" needs every specification to use model="feols": '
            "removing the estimated effect from the outcome does not impose the "
            "null in non-linear models."
        )
    if (
        method == "bayesian-bootstrap"
        and (fit_kwargs or {}).get("weights_type", "aweights") != "aweights"
    ):
        raise ValueError(
            'method="bayesian-bootstrap" needs weights_type="aweights": its '
            "weights are not whole numbers."
        )
    ok = results[results["error"].isna()].reset_index(drop=True)
    if ok.empty:
        raise ValueError("No specification was estimated successfully")
    if method == "permutation" and ok["iv"].notna().any():
        raise ValueError(
            'method="permutation" does not apply to instrumental-variable '
            "specifications: shuffling the endogenous regressor but not its "
            'instruments breaks the first stage. Use method="bootstrap".'
        )

    focal_columns: list[str] = []
    for term in ok["x"].unique():
        focal_columns.extend(
            v for v in variables_in(term, list(data.columns)) if v not in focal_columns
        )
    _check_design(data, method, cluster, strata, focal_columns)

    grid = ok
    frame = data
    weight_sources: dict[str, Optional[str]] = {}
    if method in NULL_OUTCOME_METHODS:
        grid, frame = _null_grid(ok, data)
    if method == "bayesian-bootstrap":
        grid, frame, weight_sources = _weighted_grid(grid, frame)
    grid = grid[GRID_COLUMNS].copy()
    grid["vcov"] = ok["vcov"].map(lambda v: v if v in vcov_specs else "default")
    engine_kwargs = {
        "vcov_specs": {**vcov_specs, "default": None},
        "subsets": subsets,
        "common_sample": common_sample,
        "fit_kwargs": fit_kwargs,
    }
    units = _units(frame, cluster, strata)
    seeds = np.random.SeedSequence(seed).spawn(n_boot)
    show = n_boot >= 20 if progress is None else progress
    iterator = tqdm(seeds, desc="Resampling under the null", disable=not show)
    args = (method, grid, frame, units, focal_columns, engine_kwargs, weight_sources)
    if n_jobs == 1:
        outputs = [_one_resample(child, *args) for child in iterator]
    else:
        outputs = Parallel(n_jobs=n_jobs)(
            delayed(_one_resample)(child, *args) for child in iterator
        )
    estimates = np.vstack([output[0] for output in outputs])
    pvalues = np.vstack([output[1] for output in outputs])

    usable = (~np.isnan(estimates)).mean(axis=1) >= min_valid
    n_dropped = int((~usable).sum())
    if not usable.any():
        raise ValueError(
            "No resample could be used: too few specifications could be estimated "
            "in each of them"
        )
    if n_dropped:
        warnings.warn(
            f"{n_dropped} of {n_boot} resamples were left out because fewer than "
            f"{min_valid:.0%} of the specifications could be estimated in them.",
            stacklevel=3,
        )
    null_estimates = estimates[usable]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        quantiles = np.nanquantile(
            np.sort(null_estimates, axis=1), NULL_QUANTILES, axis=0
        )
    curves = pd.DataFrame(quantiles.T, columns=list(NULL_QUANTILES))
    curves.index.name = "rank"

    observed = curve_statistics(
        ok["estimate"].to_numpy(dtype=float),
        ok["pvalue"].to_numpy(dtype=float),
        alpha,
        alternative,
    )
    null = curve_statistics(null_estimates, pvalues[usable], alpha, alternative)
    summary = pd.DataFrame(
        {
            "observed": observed,
            "p-value": resampling_pvalues(observed, null, alternative),
        },
        index=list(STATISTICS),
    )
    draws = None
    if keep_draws:
        draws = pd.DataFrame(estimates, columns=ok["spec_id"].to_numpy())
        draws.index.name = "resample"
        draws.columns.name = "spec_id"
    return NullResult(curves, summary, draws, method, int(usable.sum()), n_dropped)
