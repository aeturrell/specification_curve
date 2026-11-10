"""Runs a specification grid through pyfixest and collects tidy results."""

import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from tqdm.auto import tqdm

from . import _pyfixest as backend
from ._pyfixest import VcovSpec, ci_suffix, normalise
from .grid import cluster_variables, missing_to_none, row_variables

STAT_COLUMNS: list[str] = ["estimate", "se", "statistic", "pvalue"]

BATCH_SIZE = 64
PROGRESS_THRESHOLD = 500


@dataclass
class _Unit:
    """One model to fit, and the grid rows (one per vcov) that read from it."""

    y: str
    x: str
    controls: tuple[str, ...]
    formula: str
    rows: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class _Call:
    """One pyfixest call covering one or more units that share a sample."""

    frame_key: tuple[Any, ...]
    model: str
    family: Optional[str]
    quantile: Optional[float]
    weights: Optional[str]
    units: list[_Unit]
    formula: Optional[str] = None


def result_columns(ci_levels: Sequence[float]) -> list[str]:
    """Names of the estimate columns that the engine adds to the grid.

    Args:
        ci_levels (Sequence[float]): Confidence levels, e.g. (0.95,).

    Returns:
        list[str]: Column names, in order.
    """
    intervals = [
        f"ci_{side}_{ci_suffix(level)}"
        for level in ci_levels
        for side in ("low", "high")
    ]
    return [*STAT_COLUMNS, *intervals, "nobs", "error"]


def _prepare_frames(
    grid: pd.DataFrame,
    data: pd.DataFrame,
    vcov_specs: Mapping[str, VcovSpec],
    subsets: Mapping[str, Optional[str]],
    common_sample: bool,
) -> tuple[dict[tuple[Any, ...], pd.DataFrame], dict[int, tuple[Any, ...]]]:
    """Builds the estimation sample(s) and maps each grid row to its sample."""
    columns = list(data.columns)
    frames: dict[tuple[Any, ...], pd.DataFrame] = {}
    frame_of_row: dict[int, tuple[Any, ...]] = {}
    for subset, rows in grid.groupby("subset", sort=False):
        used: list[str] = []
        for row in rows.to_dict("records"):
            for name in row_variables(row, columns, vcov_specs[row["vcov"]]):
                if name not in used:
                    used.append(name)
        query = subsets[str(subset)]
        base = data if query is None else data.query(query)
        base = base[[column for column in columns if str(column) in used]]
        if common_sample:
            base = base.dropna()
        frames[(subset,)] = base.copy()
        has_missing = {str(c) for c in base.columns[base.isna().any()]}
        for row in rows.to_dict("records"):
            clusters = cluster_variables(vcov_specs[row["vcov"]], columns)
            missing = tuple(c for c in clusters if c in has_missing)
            key = (subset, *missing)
            if key not in frames:
                frames[key] = base.dropna(subset=list(missing)).copy()
            frame_of_row[row["spec_id"]] = key
    return frames, frame_of_row


def _build_calls(
    grid: pd.DataFrame,
    frame_of_row: Mapping[int, tuple[Any, ...]],
    batch: bool,
    n_workers: int = 1,
) -> list[_Call]:
    """Groups grid rows into units (one per model) and units into pyfixest calls."""
    buckets: dict[tuple[Any, ...], dict[tuple[Any, ...], _Unit]] = {}
    for row in grid.to_dict("records"):
        bucket_key = (
            frame_of_row[row["spec_id"]],
            row["model"],
            row["family"],
            row["quantile"],
            row["weights"],
            row["iv"],
            row["x"],
            row["always"],
            row["fe"],
        )
        units = buckets.setdefault(bucket_key, {})
        unit = units.setdefault(
            (row["y"], row["controls"]),
            _Unit(row["y"], row["x"], row["controls"], row["formula"]),
        )
        unit.rows.append((row["spec_id"], row["vcov"]))

    calls: list[_Call] = []
    for bucket_key, units in buckets.items():
        frame_key, model, family, quantile, weights, iv, x, always, fe = bucket_key
        settings = (frame_key, model, family, quantile, weights)
        if not batch or iv is not None or len(units) == 1:
            calls.extend(_Call(*settings, [unit]) for unit in units.values())
            continue
        # Outcomes that are paired with the same list of control sets share a call.
        by_outcome: dict[str, list[_Unit]] = {}
        for unit in units.values():
            by_outcome.setdefault(unit.y, []).append(unit)
        by_controls: dict[tuple[tuple[str, ...], ...], list[str]] = {}
        for outcome, outcome_units in by_outcome.items():
            control_sets = tuple(unit.controls for unit in outcome_units)
            by_controls.setdefault(control_sets, []).append(outcome)
        for control_sets, outcomes in by_controls.items():
            # With several workers, use smaller batches so each has work to do.
            size = min(BATCH_SIZE, -(-len(control_sets) // n_workers))
            for start in range(0, len(control_sets), size):
                chunk = control_sets[start : start + size]
                chunk_units = [
                    units[(y, controls)] for y in outcomes for controls in chunk
                ]
                formula = _batched_formula(outcomes, x, always, chunk, fe)
                calls.append(_Call(*settings, chunk_units, formula))
    return calls


def _batched_formula(
    outcomes: Sequence[str],
    x: str,
    always: Sequence[str],
    control_sets: Sequence[tuple[str, ...]],
    fe: Optional[str],
) -> Optional[str]:
    """Writes one pyfixest formula that covers several outcomes and control sets."""
    if len(outcomes) * len(control_sets) == 1:
        return None
    rhs = [x, *always]
    filled = [" + ".join(controls) for controls in control_sets if controls]
    has_empty = len(filled) < len(control_sets)
    if has_empty and filled:
        rhs.append(f"sw0({', '.join(filled)})")
    elif len(filled) > 1:
        rhs.append(f"sw({', '.join(filled)})")
    elif filled:
        rhs.append(filled[0])
    formula = f"{' + '.join(outcomes)} ~ {' + '.join(rhs)}"
    return formula if fe is None else f"{formula} | {fe}"


def _model_key(y: str, terms: Sequence[str], fe: Optional[str]) -> tuple[Any, ...]:
    """A spelling-independent key identifying a model by its parts."""
    fe_terms: frozenset[str] = frozenset()
    if fe is not None:
        fe_terms = frozenset(normalise(part) for part in fe.split("+"))
    return (normalise(y), frozenset(normalise(term) for term in terms), fe_terms)


def _formula_key(formula: str) -> tuple[Any, ...]:
    """The model key of a single-model formula, as pyfixest reads it."""
    parsed = backend.parse_formula(formula)[0]
    return _model_key(parsed.y, parsed.exog, parsed.fe)


def _fit_batched(
    call: _Call, frame: pd.DataFrame, fit_kwargs: Mapping[str, Any]
) -> Optional[list[Any]]:
    """Fits a batched call; returns one model per unit, or None if that fails."""
    try:
        models = backend.estimate(
            str(call.formula),
            frame,
            model=call.model,
            family=call.family,
            quantile=call.quantile,
            weights=call.weights,
            **fit_kwargs,
        )
        by_key = {_formula_key(str(model._fml)): model for model in models}
        matched = [by_key.get(_formula_key(unit.formula)) for unit in call.units]
    except Exception:
        return None
    if len(models) != len(call.units) or any(model is None for model in matched):
        return None
    return matched


def _extract(
    unit: _Unit,
    model: Any,
    vcov_specs: Mapping[str, VcovSpec],
    ci_levels: tuple[float, ...],
    on_error: str,
) -> list[dict[str, Any]]:
    """Reads the focal coefficient off a fitted model for each of its vcov rows."""
    records = []
    # Rows using pyfixest's default are read first, before the model is changed.
    for spec_id, label in sorted(unit.rows, key=lambda r: vcov_specs[r[1]] is not None):
        record: dict[str, Any] = {"spec_id": spec_id, "error": None}
        spec = vcov_specs[label]
        try:
            if spec is not None:
                backend.set_vcov(model, spec)
            record["vcov"] = (
                label if spec is not None else backend.model_vcov_label(model)
            )
            record.update(backend.focal_stats(model, unit.x, ci_levels))
            record["nobs"] = backend.model_nobs(model)
        except Exception as err:
            if on_error == "raise":
                raise RuntimeError(
                    f"Specification {spec_id} ({unit.formula}, vcov={label}) failed: {err}"
                ) from err
            record["error"] = f"{type(err).__name__}: {err}"
        records.append(record)
    return records


def _run_call(
    call: _Call,
    frame: pd.DataFrame,
    vcov_specs: Mapping[str, VcovSpec],
    ci_levels: tuple[float, ...],
    on_error: str,
    keep_models: bool,
    fit_kwargs: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, Any]]:
    """Fits the models of one call and returns their result records."""
    fitted: Optional[list[Any]] = None
    if call.formula is not None:
        fitted = _fit_batched(call, frame, fit_kwargs)
    records: list[dict[str, Any]] = []
    models: dict[int, Any] = {}
    for position, unit in enumerate(call.units):
        try:
            if len(frame) == 0:
                raise ValueError("the estimation sample has no rows")
            model = (
                fitted[position]
                if fitted is not None
                else backend.estimate(
                    unit.formula,
                    frame,
                    model=call.model,
                    family=call.family,
                    quantile=call.quantile,
                    weights=call.weights,
                    **fit_kwargs,
                )[0]
            )
        except Exception as err:
            if on_error == "raise":
                raise RuntimeError(
                    f"Specification {unit.rows[0][0]} ({unit.formula}) failed: {err}"
                ) from err
            message = f"{type(err).__name__}: {err}"
            records.extend(
                {"spec_id": spec_id, "error": message} for spec_id, _ in unit.rows
            )
            continue
        records.extend(_extract(unit, model, vcov_specs, ci_levels, on_error))
        if keep_models:
            models.update({spec_id: model for spec_id, _ in unit.rows})
    return records, models


def _run_call_quietly(
    call: _Call, frame: pd.DataFrame, *args: Any
) -> tuple[list[dict[str, Any]], dict[int, Any], list[str]]:
    """Runs one call, returning any warnings it raised instead of showing them."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        records, models = _run_call(call, frame, *args)
    internal = (DeprecationWarning, PendingDeprecationWarning, FutureWarning)
    messages = [
        " ".join(str(warning.message).split())
        for warning in caught
        if not issubclass(warning.category, internal)
    ]
    return records, models, messages


def run_grid(
    grid: pd.DataFrame,
    data: pd.DataFrame,
    *,
    vcov_specs: Mapping[str, VcovSpec],
    subsets: Mapping[str, Optional[str]],
    common_sample: bool = True,
    ci_levels: Sequence[float] = (0.95,),
    n_jobs: int = 1,
    on_error: str = "record",
    keep_models: bool = False,
    batch: bool = True,
    progress: Optional[bool] = None,
    warn: bool = True,
    fit_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[pd.DataFrame, dict[int, Any]]:
    """Estimates every specification in a grid.

    Specifications that differ only in their variance-covariance type share one
    fit. With ``batch=True``, specifications that share a sample, estimator,
    focal regressor and fixed effects are sent to pyfixest in one call so that it
    can reuse work between them.

    Args:
        grid (pd.DataFrame): Grid rows, as built by ``build_grid``.
        data (pd.DataFrame): The data. It is not modified.
        vcov_specs (Mapping[str, VcovSpec]): Maps the grid's ``vcov`` labels to the
            values passed to pyfixest.
        subsets (Mapping[str, Optional[str]]): Maps subset names to queries.
        common_sample (bool): Use one complete-case sample for all specifications
            within a subset.
        ci_levels (Sequence[float]): Confidence levels to store.
        n_jobs (int): Number of parallel workers.
        on_error (str): "record" keeps failed specifications in the results with
            their error message; "raise" stops at the first failure.
        keep_models (bool): Also return the fitted pyfixest models.
        batch (bool): Combine specifications into multi-model pyfixest calls.
        progress (Optional[bool]): Show a progress bar. By default one is shown
            when there
            are many pyfixest calls to make.
        warn (bool): Summarise the warnings raised during estimation in a single
            warning.
        fit_kwargs (Optional[Mapping[str, Any]]): Passed to the pyfixest estimators.

    Returns:
        tuple[pd.DataFrame, dict[int, Any]]: The results, one row per
            specification in ``spec_id`` order, and the fitted models keyed by
            ``spec_id`` (empty unless ``keep_models`` is set).

    Raises:
        ValueError: If ``on_error`` is not "record" or "raise".
    """
    if on_error not in ("record", "raise"):
        raise ValueError('on_error must be "record" or "raise"')
    levels = tuple(float(level) for level in ci_levels)
    kwargs = dict(fit_kwargs or {})
    frames, frame_of_row = _prepare_frames(
        grid, data, vcov_specs, subsets, common_sample
    )
    calls = _build_calls(grid, frame_of_row, batch, effective_n_jobs(n_jobs))
    show = len(calls) >= PROGRESS_THRESHOLD if progress is None else progress
    args = (vcov_specs, levels, on_error, keep_models, kwargs)
    iterator = tqdm(calls, desc="Fitting specifications", disable=not show)
    if n_jobs == 1:
        outputs = [
            _run_call_quietly(call, frames[call.frame_key], *args) for call in iterator
        ]
    else:
        outputs = Parallel(n_jobs=n_jobs)(
            delayed(_run_call_quietly)(call, frames[call.frame_key], *args)
            for call in iterator
        )
    records = [record for output in outputs for record in output[0]]
    models = {
        spec_id: model for output in outputs for spec_id, model in output[1].items()
    }
    messages = Counter(message for output in outputs for message in output[2])
    if warn and messages:
        common = "; ".join(f"{text} ({n}x)" for text, n in messages.most_common(3))
        warnings.warn(
            f"{sum(messages.values())} warning(s) were raised during estimation. "
            f"Most common: {common}",
            stacklevel=3,
        )

    columns = result_columns(levels)
    stats = pd.DataFrame.from_records(records).set_index("spec_id")
    results = grid.set_index("spec_id")
    if "vcov" in stats:
        results["vcov"] = stats["vcov"].where(stats["vcov"].notna(), results["vcov"])
    for column in columns:
        results[column] = stats[column] if column in stats else None
    results["error"] = missing_to_none(results["error"])
    results["nobs"] = results["nobs"].astype("Int64")
    return results.reset_index().sort_values("spec_id", ignore_index=True), models
