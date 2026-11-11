"""Adapter around pyfixest.

This is the only module that imports pyfixest. Everything the rest of the
package needs from it (formula parsing, estimation, and reading numbers off a
fitted model) goes through the functions here.
"""

import inspect
import re
from dataclasses import dataclass
from typing import Any, Optional, Union

import pandas as pd
import pyfixest as pf
from formulaic import Formula as _FormulaicFormula
from pyfixest.estimation.formula.parse import Formula as _PyfixestFormula

VcovSpec = Union[str, dict[str, str], None]

MODELS: tuple[str, ...] = ("feols", "fepois", "feglm", "quantreg")

_ESTIMATORS: dict[str, Any] = {
    "feols": pf.feols,
    "fepois": pf.fepois,
    "feglm": pf.feglm,
    "quantreg": pf.quantreg,
}

# Arguments that the engine sets itself and that cannot be passed through.
_RESERVED_KWARGS = frozenset(
    {"fml", "data", "vcov", "weights", "family", "quantile", "split", "fsplit"}
)


def normalise(term: str) -> str:
    """Removes all whitespace from a term so that spellings can be compared."""
    return re.sub(r"\s+", "", term)


@dataclass(frozen=True)
class ParsedFormula:
    """The parts of a single-model pyfixest formula.

    Attributes:
        y (str): Outcome term.
        exog (tuple[str, ...]): Exogenous right-hand-side terms, without the intercept.
        fe (Optional[str]): Fixed effects part, or None.
        endog (tuple[str, ...]): Endogenous terms (instrumental variables only).
        instruments (tuple[str, ...]): Instruments (instrumental variables only).
        intercept (bool): Whether the model has an intercept (or fixed effects that absorb it).
    """

    y: str
    exog: tuple[str, ...]
    fe: Optional[str]
    endog: tuple[str, ...]
    instruments: tuple[str, ...]
    intercept: bool


def _terms(formula_part: Any) -> tuple[str, ...]:
    """Lists the non-intercept terms of a parsed formula part as strings."""
    return tuple(str(term) for term in formula_part if str(term) != "1")


def parse_formula(formula: str) -> list[ParsedFormula]:
    """Parses a pyfixest formula, expanding any multiple-estimation syntax.

    Args:
        formula (str): A pyfixest formula, which may use ``sw``, ``sw0``, ``csw``,
            ``csw0`` and several outcomes.

    Returns:
        list[ParsedFormula]: One entry per model that the formula describes.

    Raises:
        ValueError: If pyfixest cannot parse the formula.
    """
    try:
        parsed = _PyfixestFormula.parse(formula)
    except Exception as err:
        raise ValueError(
            f"pyfixest could not parse the formula {formula!r}: {err}"
        ) from err
    out: list[ParsedFormula] = []
    for item in parsed:
        has_fe = bool(item.is_fixed_effects)
        is_iv = bool(item.is_instrumental_variable)
        exog_terms = [str(term) for term in item.exogenous]
        out.append(
            ParsedFormula(
                y=str(item.dependent).strip(),
                exog=tuple(term for term in exog_terms if term != "1"),
                fe=str(item.fixed_effects).strip() if has_fe else None,
                endog=_terms(item.endogenous) if is_iv else (),
                instruments=_terms(item.instruments) if is_iv else (),
                intercept=has_fe or "1" in exog_terms,
            )
        )
    return out


def accepted_kwargs(model: str) -> frozenset[str]:
    """Names of the pass-through keyword arguments that an estimator accepts."""
    params = inspect.signature(_ESTIMATORS[model]).parameters
    return frozenset(params) - _RESERVED_KWARGS


def estimate(
    formula: str,
    data: pd.DataFrame,
    *,
    model: str,
    family: Optional[str] = None,
    quantile: Optional[float] = None,
    weights: Optional[str] = None,
    **kwargs: Any,
) -> list[Any]:
    """Runs one pyfixest estimation call.

    Args:
        formula (str): pyfixest formula; may describe several models.
        data (pd.DataFrame): Estimation data. pyfixest works on its own copy.
        model (str): One of "feols", "fepois", "feglm", "quantreg".
        family (Optional[str]): Family, for "feglm".
        quantile (Optional[float]): Quantile, for "quantreg".
        weights (Optional[str]): Name of the weights column.
        **kwargs (Any): Passed to the pyfixest estimator if it accepts them.

    Returns:
        list[Any]: The fitted pyfixest models, in pyfixest's order.
    """
    allowed = accepted_kwargs(model)
    call_kwargs: dict[str, Any] = {k: v for k, v in kwargs.items() if k in allowed}
    if model == "feglm":
        call_kwargs["family"] = family
    if model == "quantreg":
        call_kwargs["quantile"] = quantile
    elif weights is not None:
        call_kwargs["weights"] = weights
    fit = _ESTIMATORS[model](formula, data, **call_kwargs)
    if hasattr(fit, "all_fitted_models"):
        return list(fit.all_fitted_models.values())
    return [fit]


def model_nobs(model: Any) -> int:
    """Number of observations a fitted model used."""
    return int(model._N)


def set_vcov(model: Any, vcov: Union[str, dict[str, str]]) -> None:
    """Recomputes a fitted model's variance-covariance matrix in place."""
    model.vcov(vcov)


def vcov_label(vcov: VcovSpec) -> str:
    """Readable label for a variance-covariance specification."""
    if vcov is None:
        return "default"
    if isinstance(vcov, dict):
        return ", ".join(f"{kind}: {cluster}" for kind, cluster in vcov.items())
    return vcov


def model_vcov_label(model: Any) -> str:
    """Readable label for the variance-covariance type a fitted model is using."""
    return str(model._vcov_type)


def focal_stats(
    model: Any, focal: str, ci_levels: tuple[float, ...]
) -> dict[str, float]:
    """Reads the focal coefficient's estimate and inference off a fitted model.

    Args:
        model (Any): Fitted pyfixest model.
        focal (str): Name of the focal coefficient.
        ci_levels (tuple[float, ...]): Confidence levels, e.g. (0.95,).

    Returns:
        dict[str, float]: Estimate, standard error, statistic, p-value and one
            pair of interval bounds per confidence level.

    Raises:
        ValueError: If the model has no coefficient with that name.
    """
    table = model.tidy()
    lookup = {normalise(str(name)): name for name in table.index}
    key = normalise(focal)
    if key not in lookup:
        raise ValueError(
            f"no coefficient named {focal!r} in the fitted model (it may have been "
            f"dropped as collinear); coefficients are {list(table.index)}"
        )
    name = lookup[key]
    row = table.loc[name]
    out = {
        "estimate": float(row.iloc[0]),
        "se": float(row.iloc[1]),
        "statistic": float(row.iloc[2]),
        "pvalue": float(row.iloc[3]),
    }
    for level in ci_levels:
        interval = model.confint(alpha=1 - level).loc[name]
        out[f"ci_low_{ci_suffix(level)}"] = float(interval.iloc[0])
        out[f"ci_high_{ci_suffix(level)}"] = float(interval.iloc[1])
    return out


def ci_suffix(level: float) -> str:
    """Column-name suffix for a confidence level, e.g. 0.95 -> "95"."""
    return f"{level * 100:g}"


def evaluate_term(term: str, data: pd.DataFrame) -> pd.Series:
    """Evaluates a single numeric formula term against the data.

    Args:
        term (str): A formula term such as "log(wage)" or "x1:x2".
        data (pd.DataFrame): Data to evaluate the term on.

    Returns:
        pd.Series: The term's values, aligned to the rows of ``data``.

    Raises:
        ValueError: If the term does not evaluate to exactly one numeric column.
    """
    try:
        matrix = _FormulaicFormula.from_spec(f"{term} - 1").get_model_matrix(
            data, na_action="ignore"
        )
    except Exception as err:
        raise ValueError(f"could not evaluate the term {term!r}: {err}") from err
    if matrix.shape[1] != 1:
        raise ValueError(
            f"the term {term!r} produces {matrix.shape[1]} columns; a single numeric "
            "column is needed"
        )
    values = pd.Series(matrix.iloc[:, 0].to_numpy(dtype=float), index=data.index)
    return values
