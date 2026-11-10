"""The specification grid: one row per specification, as plain data."""

import re
import warnings
from dataclasses import dataclass, replace
from itertools import combinations, product
from typing import Any, Hashable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._pyfixest import MODELS, VcovSpec, normalise, vcov_label
from .formula import expand, render, split_top_level

StrOrList = Union[str, Sequence[str]]

GRID_COLUMNS: list[str] = [
    "spec_id",
    "model",
    "family",
    "quantile",
    "y",
    "x",
    "controls",
    "always",
    "fe",
    "iv",
    "vcov",
    "weights",
    "subset",
    "formula",
]

LARGE_GRID = 10_000

_IDENTIFIER = re.compile(r"`([^`]+)`|([A-Za-z_][A-Za-z0-9_.]*)")


@dataclass(frozen=True)
class Grid:
    """A specification grid and the settings needed to run it.

    Attributes:
        frame (pd.DataFrame): One row per specification.
        vcov_specs (dict[str, VcovSpec]): Maps each label in the ``vcov`` column to
            the value passed to pyfixest.
        subsets (dict[str, Optional[str]]): Maps each subset name to its query.
        n_skipped (int): Number of invalid combinations left out.
    """

    frame: pd.DataFrame
    vcov_specs: dict[str, VcovSpec]
    subsets: dict[str, Optional[str]]
    n_skipped: int = 0

    def sample(self, n: int, seed: Optional[int] = None) -> "Grid":
        """Draws a random sample of specifications.

        Args:
            n (int): Number of specifications to keep.
            seed (Optional[int]): Seed for the random draw.

        Returns:
            Grid: A grid with at most ``n`` rows, in ``spec_id`` order.
        """
        if n >= len(self.frame):
            return self
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(self.frame), size=n, replace=False))
        return replace(self, frame=self.frame.iloc[keep].reset_index(drop=True))


def as_list(value: Any) -> list[Any]:
    """Wraps a scalar in a list; turns other sequences into lists."""
    if value is None or isinstance(value, (str, dict, int, float)):
        return [value]
    return list(value)


def missing_to_none(values: pd.Series) -> pd.Series:
    """Returns an object column in which missing values are None."""
    cleaned = [None if pd.isna(value) else value for value in values]
    return pd.Series(cleaned, index=values.index, dtype=object)


def variables_in(term: str, columns: Sequence[Hashable]) -> list[str]:
    """Names in a formula term that are columns of the data.

    Args:
        term (str): A formula term, fixed-effects string or similar.
        columns (Sequence[Hashable]): Column names of the data.

    Returns:
        list[str]: Column names the term refers to, in order of appearance.
    """
    available = {str(column) for column in columns}
    found: list[str] = []
    for quoted, bare in _IDENTIFIER.findall(term):
        name = quoted or bare
        if name in available and name not in found:
            found.append(name)
    return found


def cluster_variables(vcov: VcovSpec, columns: Sequence[Hashable]) -> list[str]:
    """Cluster columns named in a variance-covariance specification."""
    if not isinstance(vcov, dict):
        return []
    found: list[str] = []
    for value in vcov.values():
        found.extend(v for v in variables_in(str(value), columns) if v not in found)
    return found


def row_variables(
    row: Mapping[Any, Any], columns: Sequence[Hashable], vcov: VcovSpec
) -> list[str]:
    """Every data column that a specification uses.

    Args:
        row (Mapping[Any, Any]): A row of the grid.
        columns (Sequence[Hashable]): Column names of the data.
        vcov (VcovSpec): The row's variance-covariance specification.

    Returns:
        list[str]: Column names, without duplicates.
    """
    pieces = [row["formula"]]
    if row["weights"] is not None:
        pieces.append(f"`{row['weights']}`")
    found = variables_in(" ".join(pieces), columns)
    found.extend(v for v in cluster_variables(vcov, columns) if v not in found)
    return found


def _structural_from_keywords(
    y: StrOrList,
    x: StrOrList,
    controls: Sequence[str],
    always: Sequence[str],
    fe: Sequence[Optional[str]],
    iv: Sequence[Optional[str]],
    min_controls: int,
    max_controls: Optional[int],
) -> list[dict[str, Any]]:
    """Builds the structural part of the grid from keyword arguments."""
    largest = (
        len(controls) if max_controls is None else min(max_controls, len(controls))
    )
    subsets = [
        subset
        for size in range(min_controls, largest + 1)
        for subset in combinations(controls, size)
    ]
    rows = []
    for y_term, x_term, fe_term, iv_term, subset in product(
        as_list(y), as_list(x), fe, iv, subsets
    ):
        taken = {normalise(y_term), normalise(x_term)}
        rows.append(
            {
                "y": y_term,
                "x": x_term,
                "controls": tuple(c for c in subset if normalise(c) not in taken),
                "always": tuple(a for a in always if normalise(a) not in taken),
                "fe": fe_term,
                "iv": iv_term,
            }
        )
    return rows


def _structural_from_formula(formula: str, x: StrOrList) -> list[dict[str, Any]]:
    """Builds the structural part of the grid from a pyfixest formula."""
    parsed = expand(formula)
    focal_names = {normalise(term) for term in as_list(x)}
    in_every_model = set.intersection(
        *[{normalise(term) for term in item.exog} for item in parsed]
    )
    rows = []
    for item in parsed:
        if not item.intercept:
            raise ValueError(
                "Formulas without an intercept are not supported; pass "
                "drop_intercept=True to fit() instead."
            )
        focal = [
            term for term in (*item.exog, *item.endog) if normalise(term) in focal_names
        ]
        if len(focal) != 1:
            raise ValueError(
                "Each specification needs exactly one focal regressor from "
                f"x={as_list(x)}, but the model with outcome {item.y!r} and terms "
                f"{list(item.exog + item.endog)} has {len(focal)}."
            )
        others = [term for term in item.exog if term != focal[0]]
        iv_term = None
        if item.endog:
            iv_term = f"{' + '.join(item.endog)} ~ {' + '.join(item.instruments)}"
        rows.append(
            {
                "y": item.y,
                "x": focal[0],
                "controls": tuple(
                    t for t in others if normalise(t) not in in_every_model
                ),
                "always": tuple(t for t in others if normalise(t) in in_every_model),
                "fe": item.fe,
                "iv": iv_term,
            }
        )
    return rows


def _filter_structural(
    rows: list[dict[str, Any]],
    exclude: Sequence[Sequence[str]],
    min_controls: int,
    max_controls: Optional[int],
) -> list[dict[str, Any]]:
    """Applies exclusion groups and size limits, and removes duplicate rows."""
    groups = [{normalise(term) for term in group} for group in exclude if group]
    kept: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        included = {normalise(term) for term in row["controls"]}
        if any(group <= included for group in groups):
            continue
        if len(included) < min_controls:
            continue
        if max_controls is not None and len(included) > max_controls:
            continue
        key = tuple(row.values())
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def _model_variants(
    model: StrOrList,
    family: Union[str, Sequence[str], None],
    quantile: Union[float, Sequence[float], None],
) -> list[tuple[str, Optional[str], Optional[float]]]:
    """Expands the model, family and quantile arguments into estimator settings."""
    models = as_list(model)
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}; choose from {list(MODELS)}")
    families = [] if family is None else as_list(family)
    quantiles = [] if quantile is None else [float(q) for q in as_list(quantile)]
    if "feglm" in models and not families:
        raise ValueError('model "feglm" needs a family, e.g. family="logit"')
    if families and "feglm" not in models:
        raise ValueError('family is only used by model "feglm"')
    if quantiles and "quantreg" not in models:
        raise ValueError('quantile is only used by model "quantreg"')
    variants: list[tuple[str, Optional[str], Optional[float]]] = []
    for name in models:
        if name == "feglm":
            variants.extend((name, fam, None) for fam in families)
        elif name == "quantreg":
            variants.extend((name, None, q) for q in (quantiles or [0.5]))
        else:
            variants.append((name, None, None))
    return variants


def _invalid_reason(
    model: str, row: Mapping[str, Any], weights: Optional[str]
) -> Optional[str]:
    """Explains why an estimator cannot run a specification, or returns None."""
    if model == "quantreg":
        if row["fe"] is not None:
            return "quantreg does not support fixed effects"
        if weights is not None:
            return "quantreg does not support weights"
        if row["iv"] is not None:
            return "quantreg does not support instrumental variables"
    if model in ("fepois", "feglm") and row["iv"] is not None:
        return f"{model} does not support instrumental variables"
    return None


def _check_columns(
    frame: pd.DataFrame,
    columns: Sequence[Hashable],
    vcov_specs: Mapping[str, VcovSpec],
) -> None:
    """Raises if any term, fixed effect, cluster or weight refers to no column."""
    available = {str(column) for column in columns}
    loose: set[str] = set()
    strict: set[str] = set()
    rows: list[dict[Any, Any]] = frame.to_dict("records")
    for row in rows:
        loose.update([row["y"], row["x"], *row["controls"], *row["always"]])
        if row["fe"] is not None:
            strict.add(row["fe"])
        if row["iv"] is not None:
            for side in split_top_level(row["iv"], "~"):
                loose.update(split_top_level(side, "+"))
    for weight in frame["weights"].dropna().unique():
        strict.add(f"`{weight}`")
    for spec in vcov_specs.values():
        if isinstance(spec, dict):
            strict.update(str(value) for value in spec.values())
    for term in sorted(loose):
        if not variables_in(term, columns):
            raise ValueError(
                f"The term {term!r} does not refer to any column of the data"
            )
    for term in sorted(strict):
        names = [quoted or bare for quoted, bare in _IDENTIFIER.findall(term)]
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(f"Column(s) {missing} in {term!r} are not in the data")


def _check_subsets(data: pd.DataFrame, subsets: Mapping[str, Optional[str]]) -> None:
    """Raises if a subset query fails or selects no rows."""
    for name, query in subsets.items():
        if query is None:
            continue
        try:
            n_rows = len(data.query(query))
        except Exception as err:
            raise ValueError(
                f"The query for subset {name!r} failed: {query!r} ({err})"
            ) from err
        if n_rows == 0:
            raise ValueError(
                f"The query for subset {name!r} selects no rows: {query!r}"
            )


def build_grid(
    data: pd.DataFrame,
    formula: Optional[str] = None,
    *,
    y: Union[str, Sequence[str], None] = None,
    x: Union[str, Sequence[str], None] = None,
    controls: Union[str, Sequence[str], None] = None,
    always: Union[str, Sequence[str], None] = None,
    exclude: Optional[Sequence[Sequence[str]]] = None,
    min_controls: int = 0,
    max_controls: Optional[int] = None,
    fe: Union[str, Sequence[Optional[str]], None] = None,
    iv: Union[str, Sequence[Optional[str]], None] = None,
    model: StrOrList = "feols",
    family: Union[str, Sequence[str], None] = None,
    quantile: Union[float, Sequence[float], None] = None,
    vcov: Union[VcovSpec, Sequence[VcovSpec]] = None,
    weights: Union[str, Sequence[Optional[str]], None] = None,
    subsets: Optional[Mapping[str, Optional[str]]] = None,
    skip_invalid: bool = False,
) -> Grid:
    """Builds the grid of specifications.

    Args:
        data (pd.DataFrame): The data the specifications will be run on.
        formula (Optional[str]): pyfixest formula describing outcomes, regressors,
            fixed effects and instruments. Use this or ``y``.
        y (Union[str, Sequence[str], None]): Outcome term(s).
        x (Union[str, Sequence[str], None]): Focal regressor(s).
        controls (Union[str, Sequence[str], None]): Terms included in all combinations.
        always (Union[str, Sequence[str], None]): Terms in every specification.
        exclude (Optional[Sequence[Sequence[str]]]): Groups of controls that never
            appear together.
        min_controls (int): Smallest number of controls in a specification.
        max_controls (Optional[int]): Largest number of controls in a specification.
        fe (Union[str, Sequence[Optional[str]], None]): Fixed-effect sets.
        iv (Union[str, Sequence[Optional[str]], None]): First stage(s), e.g. "x ~ z".
        model (StrOrList): Estimator(s).
        family (Union[str, Sequence[str], None]): Family or families for "feglm".
        quantile (Union[float, Sequence[float], None]): Quantile(s) for "quantreg".
        vcov (Union[VcovSpec, Sequence[VcovSpec]]): Variance-covariance type(s).
        weights (Union[str, Sequence[Optional[str]], None]): Weight column(s).
        subsets (Optional[Mapping[str, Optional[str]]]): Named sample restrictions.
        skip_invalid (bool): Drop combinations an estimator cannot run instead of
            raising.

    Returns:
        Grid: The grid.

    Raises:
        ValueError: If the arguments are inconsistent, refer to columns that are not
            in the data, or describe combinations an estimator cannot run.
    """
    if x is None:
        raise ValueError("x (the focal regressor) is required")
    exclude = [] if exclude is None else [as_list(group) for group in exclude]
    if formula is not None:
        if any(arg is not None for arg in (y, controls, always, fe, iv)):
            raise ValueError(
                "Give either a formula or y (with controls, always, fe, iv), not both"
            )
        structural = _structural_from_formula(formula, x)
    else:
        if y is None:
            raise ValueError("Give either a formula or y")
        structural = _structural_from_keywords(
            y,
            x,
            [] if controls is None else as_list(controls),
            [] if always is None else as_list(always),
            as_list(fe),
            as_list(iv),
            min_controls,
            max_controls,
        )
    structural = _filter_structural(structural, exclude, min_controls, max_controls)
    if not structural:
        raise ValueError("No specifications are left after applying the restrictions")

    variants = _model_variants(model, family, quantile)
    vcov_specs = {vcov_label(spec): spec for spec in as_list(vcov)}
    subset_map: dict[str, Optional[str]] = (
        {"all": None} if subsets is None else dict(subsets)
    )
    if not subset_map:
        raise ValueError("subsets must name at least one sample")
    _check_subsets(data, subset_map)

    records: list[dict[str, Any]] = []
    reasons: list[str] = []
    for subset, (name, fam, q), weight, row, label in product(
        subset_map, variants, as_list(weights), structural, vcov_specs
    ):
        reason = _invalid_reason(name, row, weight)
        if reason is not None:
            reasons.append(reason)
            continue
        records.append(
            {
                "model": name,
                "family": fam,
                "quantile": q,
                **row,
                "vcov": label,
                "weights": weight,
                "subset": subset,
                "formula": render(
                    row["y"],
                    row["x"],
                    row["controls"],
                    row["always"],
                    row["fe"],
                    row["iv"],
                ),
            }
        )
    if reasons and not skip_invalid:
        raise ValueError(
            f"{len(reasons)} combination(s) cannot be estimated: "
            f"{'; '.join(sorted(set(reasons)))}. Change the grid, or pass "
            "skip_invalid=True to leave these combinations out."
        )
    if not records:
        raise ValueError("No valid specifications: " + "; ".join(sorted(set(reasons))))
    if reasons:
        warnings.warn(
            f"Left out {len(reasons)} combination(s) that cannot be estimated: "
            f"{'; '.join(sorted(set(reasons)))}",
            stacklevel=3,
        )

    frame = pd.DataFrame.from_records(records)
    frame.insert(0, "spec_id", np.arange(len(frame)))
    frame = frame[GRID_COLUMNS]
    for column in ("family", "quantile", "fe", "iv", "weights"):
        frame[column] = missing_to_none(frame[column])
    _check_columns(frame, list(data.columns), vcov_specs)
    if len(frame) > LARGE_GRID:
        warnings.warn(
            f"The grid has {len(frame):,} specifications. Consider "
            "fit(max_specs=...) to estimate a random sample of them.",
            stacklevel=3,
        )
    return Grid(frame, vcov_specs, subset_map, n_skipped=len(reasons))


def subsets_by(
    data: pd.DataFrame, column: str, include_all: bool = False
) -> dict[str, Optional[str]]:
    """Builds a ``subsets`` argument with one sample per level of a column.

    Args:
        data (pd.DataFrame): The data.
        column (str): Column whose levels define the samples.
        include_all (bool): Also include the full sample, named "all".

    Returns:
        dict[str, Optional[str]]: Maps a name for each sample to its query.

    Examples:
        >>> import pandas as pd
        >>> subsets_by(pd.DataFrame({"g": ["a", "b", "a"]}), "g")
        {'g = a': "`g` == 'a'", 'g = b': "`g` == 'b'"}
    """
    out: dict[str, Optional[str]] = {"all": None} if include_all else {}
    for level in sorted(data[column].dropna().unique()):
        value = level.item() if hasattr(level, "item") else level
        out[f"{column} = {value}"] = f"`{column}` == {value!r}"
    return out
