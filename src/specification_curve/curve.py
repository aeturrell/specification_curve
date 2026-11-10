"""The SpecificationCurve class."""

import os
from typing import Any, Mapping, Optional, Sequence, Union

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from . import inference
from ._pyfixest import VcovSpec, accepted_kwargs
from .engine import run_grid
from .grid import build_grid
from .inference import NullResult
from .plot import plot_curve


def _format_number(value: float) -> str:
    """Formats a number to three significant figures."""
    return f"{value:.3g}"


class SpecificationCurve:
    """Runs and plots every variant of a regression specification.

    A specification curve shows how the estimate of one coefficient (the focal
    regressor, ``x``) changes across many defensible ways of specifying a model.
    The choices that can be varied are the outcome, the focal regressor, the
    control variables, the fixed effects, instruments, the estimator, the
    standard errors, the weights and the sample. All estimation is done by
    pyfixest, and every term is written in pyfixest's formula syntax, so
    transformations such as ``log(wage)`` and interactions such as ``i(year)``
    can be used anywhere.

    Specifications are described either with keyword arguments or with a
    pyfixest formula. In a formula, the multiple-estimation operators define the
    choices: several outcomes (``y1 + y2 ~``), ``sw()``, ``sw0()``, ``csw()``,
    ``csw0()``, and two operators added by this package, ``comb()`` (every
    non-empty subset of its arguments) and ``comb0()`` (every subset, including
    none of them).

    Args:
        data (pd.DataFrame): The data.
        formula (Optional[str]): pyfixest formula describing the outcomes,
            regressors, fixed effects and instruments, for example
            ``"y1 + y2 ~ x + comb0(a, b) | sw0(firm, firm + year)"``. Use this or
            ``y``, not both.
        y (Union[str, Sequence[str], None]): Outcome term(s).
        x (Union[str, Sequence[str], None]): Focal regressor(s): the coefficient
            that is plotted, named as pyfixest reports it. Required. With several,
            each specification uses one of them. With instruments, this is usually
            the endogenous variable.
        controls (Union[str, Sequence[str], None]): Terms included in every
            combination.
        always (Union[str, Sequence[str], None]): Terms included in every
            specification.
        exclude (Optional[Sequence[Sequence[str]]]): Groups of controls that never
            appear together in a specification.
        min_controls (int): Smallest number of controls in a specification.
        max_controls (Optional[int]): Largest number of controls in a specification.
        fe (Union[str, Sequence[Optional[str]], None]): Fixed-effect sets, each
            written as in a pyfixest formula (``"firm + year"``). ``None`` in the
            list means no fixed effects.
        iv (Union[str, Sequence[Optional[str]], None]): First stage(s), for example
            ``"x ~ z"``. ``None`` in the list means no instruments.
        model (Union[str, Sequence[str]]): Estimator(s): "feols", "fepois", "feglm"
            or "quantreg".
        family (Union[str, Sequence[str], None]): Family or families for "feglm".
        quantile (Union[float, Sequence[float], None]): Quantile(s) for "quantreg".
            Defaults to 0.5.
        vcov (Union[VcovSpec, Sequence[VcovSpec]]): Variance-covariance type(s), as
            accepted by pyfixest, for example ``"hetero"`` or ``{"CRV1": "firm"}``.
            ``None`` uses pyfixest's default.
        weights (Union[str, Sequence[Optional[str]], None]): Weight column(s).
            ``None`` in a list means unweighted.
        subsets (Optional[Mapping[str, Optional[str]]]): Named samples, each a
            ``DataFrame.query`` string or ``None`` for all rows.
        common_sample (bool): Within each subset, estimate every specification on
            the rows that are complete for all variables used by any of them, so
            that estimates are comparable. If False, each specification uses all
            the rows it can.
        skip_invalid (bool): Leave out combinations that an estimator cannot run
            (for example quantile regression with fixed effects) instead of raising.

    Attributes:
        grid (pd.DataFrame): One row per specification, available before fitting.
        results (pd.DataFrame): The grid with estimates, available after ``fit``.
        models (dict[int, Any]): Fitted pyfixest models keyed by ``spec_id``, if
            ``fit(keep_models=True)`` was used.
        null_summary (pd.DataFrame): Tests of the curve against the null, after
            ``fit_null``.
        null_curves (pd.DataFrame): Quantiles of the curve under the null, after
            ``fit_null``.
        null_draws (pd.DataFrame): Estimates in every resample, after
            ``fit_null(keep_draws=True)``.

    Examples:
        >>> import specification_curve as sc
        >>> df = sc.load_example_data2()
        >>> sco = sc.SpecificationCurve(df, y="y", x="x_1", controls=["x_2", "x_3"])
        >>> sco.n_specs
        4
        >>> len(sco.fit().results)
        4
    """

    def __init__(
        self,
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
        model: Union[str, Sequence[str]] = "feols",
        family: Union[str, Sequence[str], None] = None,
        quantile: Union[float, Sequence[float], None] = None,
        vcov: Union[VcovSpec, Sequence[VcovSpec]] = None,
        weights: Union[str, Sequence[Optional[str]], None] = None,
        subsets: Optional[Mapping[str, Optional[str]]] = None,
        common_sample: bool = True,
        skip_invalid: bool = False,
    ) -> None:
        self.data = data
        self.common_sample = common_sample
        self._grid = build_grid(
            data,
            formula,
            y=y,
            x=x,
            controls=controls,
            always=always,
            exclude=exclude,
            min_controls=min_controls,
            max_controls=max_controls,
            fe=fe,
            iv=iv,
            model=model,
            family=family,
            quantile=quantile,
            vcov=vcov,
            weights=weights,
            subsets=subsets,
            skip_invalid=skip_invalid,
        )
        self._results: Optional[pd.DataFrame] = None
        self._null: Optional[NullResult] = None
        self._fit_kwargs: dict[str, Any] = {}
        self.models: dict[int, Any] = {}

    @property
    def grid(self) -> pd.DataFrame:
        """One row per specification."""
        return self._grid.frame.copy()

    @property
    def n_specs(self) -> int:
        """Number of specifications in the grid."""
        return len(self._grid.frame)

    @property
    def results(self) -> pd.DataFrame:
        """The grid with estimates; one row per fitted specification."""
        if self._results is None:
            raise ValueError("No results yet. Run .fit() first.")
        return self._results.copy()

    def _null_result(self) -> NullResult:
        """The inference results, if fit_null() has been run."""
        if self._null is None:
            raise ValueError("No inference yet. Run .fit_null() first.")
        return self._null

    @property
    def null_curves(self) -> pd.DataFrame:
        """Quantiles of the sorted curve under the null, one row per rank."""
        return self._null_result().curves.copy()

    @property
    def null_summary(self) -> pd.DataFrame:
        """Observed statistics of the curve and their p-values under the null."""
        return self._null_result().summary.copy()

    @property
    def null_draws(self) -> pd.DataFrame:
        """Estimates in every resample, if ``fit_null(keep_draws=True)`` was used."""
        draws = self._null_result().draws
        if draws is None:
            raise ValueError("Draws were not kept. Run .fit_null(keep_draws=True).")
        return draws.copy()

    def fit(
        self,
        *,
        ci_levels: Sequence[float] = (0.95,),
        n_jobs: int = 1,
        max_specs: Optional[int] = None,
        seed: Optional[int] = None,
        on_error: str = "record",
        keep_models: bool = False,
        batch: bool = True,
        progress: Optional[bool] = None,
        **kwargs: Any,
    ) -> "SpecificationCurve":
        """Estimates the specifications.

        Args:
            ci_levels (Sequence[float]): Confidence levels to store. Each gives a
                pair of columns such as ``ci_low_95`` and ``ci_high_95``.
            n_jobs (int): Number of parallel workers.
            max_specs (Optional[int]): Estimate a random sample of this many
                specifications instead of the whole grid.
            seed (Optional[int]): Seed for the random sample.
            on_error (str): "record" keeps specifications that fail in the results
                with their error message; "raise" stops at the first failure.
            keep_models (bool): Keep the fitted pyfixest models in ``.models``,
                keyed by ``spec_id``. Specifications that differ only in ``vcov``
                share one model object.
            batch (bool): Send specifications that share a sample, estimator, focal
                regressor and fixed effects to pyfixest in a single call. The
                results are the same either way.
            progress (Optional[bool]): Show a progress bar. By default one is shown
                when there are many pyfixest calls to make.
            **kwargs (Any): Passed to the pyfixest estimators, for example ``ssc``,
                ``fixef_rm``, ``demeaner`` or ``context``. Each estimator receives
                the arguments it accepts.

        Returns:
            SpecificationCurve: The object itself, so calls can be chained.

        Raises:
            TypeError: If a keyword argument is not accepted by any of the
                estimators in the grid.
        """
        models_used = self._grid.frame["model"].unique()
        accepted = frozenset().union(*(accepted_kwargs(m) for m in models_used))
        unknown = sorted(set(kwargs) - accepted)
        if unknown:
            raise TypeError(
                f"fit() got argument(s) {unknown} that the estimator(s) "
                f"{list(models_used)} do not accept"
            )
        grid = self._grid if max_specs is None else self._grid.sample(max_specs, seed)
        self._results, self.models = run_grid(
            grid.frame,
            self.data,
            vcov_specs=grid.vcov_specs,
            subsets=grid.subsets,
            common_sample=self.common_sample,
            ci_levels=ci_levels,
            n_jobs=n_jobs,
            on_error=on_error,
            keep_models=keep_models,
            batch=batch,
            progress=progress,
            fit_kwargs=kwargs,
        )
        self._fit_kwargs = dict(kwargs)
        self._null = None
        return self

    def fit_null(
        self,
        n_boot: int = 500,
        *,
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
    ) -> "SpecificationCurve":
        """Tests the curve against the null hypothesis of no effect.

        The curve is re-estimated on many resamples in which the null holds.
        Three statistics of the observed curve are then compared with their
        distribution across resamples: the median estimate, the share of
        significant estimates, and Stouffer's Z, which combines the evidence of
        all specifications. The results are in ``.null_summary`` and
        ``.null_curves``.

        With ``method="bootstrap"``, each specification's estimated effect is
        removed from its outcome and rows are resampled with replacement. This
        applies when every specification uses ``model="feols"``. With
        ``method="permutation"``, the focal variable is shuffled between rows.
        This applies to any estimator, but it is only valid if the focal
        variable is as good as randomly assigned (within ``strata``, if given),
        as in an experiment.

        Args:
            n_boot (int): Number of resamples. Use several hundred for results
                you rely on; every resample re-estimates the whole curve.
            method (str): "bootstrap", "permutation" or "auto". "auto" uses the
                bootstrap if every specification uses "feols", and otherwise
                asks you to choose.
            cluster (Optional[str]): Column identifying clusters. Whole clusters
                are resampled (bootstrap) or swap their focal values
                (permutation; the focal variable must be constant within them).
            strata (Optional[str]): Column identifying strata. Resampling is done
                within each stratum.
            alpha (float): Significance level for the share of significant
                estimates.
            alternative (str): "two-sided", "greater" (the effect is positive) or
                "less" (the effect is negative).
            seed (Optional[int]): Seed for the resampling. Results do not depend
                on ``n_jobs``.
            n_jobs (int): Number of parallel workers.
            keep_draws (bool): Keep the estimates from every resample in
                ``.null_draws``.
            min_valid (float): A resample is left out if fewer than this share of
                the specifications could be estimated in it.
            progress (Optional[bool]): Show a progress bar. By default one is
                shown when there are at least 20 resamples.

        Returns:
            SpecificationCurve: The object itself, so calls can be chained.
        """
        self._null = inference.fit_null(
            self.results,
            self.data,
            vcov_specs=self._grid.vcov_specs,
            subsets=self._grid.subsets,
            n_boot=n_boot,
            method=method,
            cluster=cluster,
            strata=strata,
            alpha=alpha,
            alternative=alternative,
            seed=seed,
            n_jobs=n_jobs,
            keep_draws=keep_draws,
            min_valid=min_valid,
            progress=progress,
            common_sample=self.common_sample,
            fit_kwargs=self._fit_kwargs,
        )
        return self

    def plot(
        self,
        save_path: Union[str, os.PathLike[str], None] = None,
        *,
        show_null_curve: bool = False,
        **kwargs: Any,
    ) -> tuple[Union[Figure, SubFigure], dict[str, Axes]]:
        """Plots the fitted specification curve.

        Args:
            save_path (Union[str, os.PathLike[str], None]): File to save the figure
                to; the format is taken from the extension.
            show_null_curve (bool): Overlay the curve expected under the null.
                Requires ``fit_null`` to have been run.
            **kwargs (Any): Passed to ``plot_curve``, for example ``sort_by``,
                ``color_by``, ``panels``, ``labels``, ``highlight`` or ``fig``.

        Returns:
            tuple[Union[Figure, SubFigure], dict[str, Axes]]: The figure and its
                axes keyed by panel name.
        """
        null_curves = self.null_curves if show_null_curve else None
        fig, axes = plot_curve(self.results, null_curves, **kwargs)
        if save_path is not None:
            root: Any = fig
            while isinstance(root, SubFigure):
                root = root.get_figure()
            root.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, axes

    def __repr__(self) -> str:
        """Summarises the grid and, once fitted, the estimates.

        Returns:
            str: Summary message.
        """
        frame = self._grid.frame
        lines = ["Specification curve", "-------------------"]
        lines.append(f"Specifications: {len(frame)}")
        controls = sorted({term for row in frame["controls"] for term in row})
        always = sorted({term for row in frame["always"] for term in row})
        described = {
            "y": list(frame["y"].unique()),
            "x": list(frame["x"].unique()),
            "controls": controls,
            "always": always,
            "fe": [v for v in frame["fe"].unique() if v is not None],
            "iv": [v for v in frame["iv"].unique() if v is not None],
            "model": list(frame["model"].unique()),
            "vcov": list(frame["vcov"].unique()),
            "weights": [v for v in frame["weights"].unique() if v is not None],
            "subset": list(frame["subset"].unique()),
        }
        for name, values in described.items():
            if values:
                lines.append(f"  {name}: {', '.join(str(v) for v in values)}")
        if self._results is not None:
            ok = self._results[self._results["error"].isna()]
            lines.append("")
            lines.append(f"Fitted: {len(ok)} of {len(self._results)} succeeded")
            if len(ok):
                estimate = ok["estimate"]
                lines.append(
                    "  estimate: "
                    f"median {_format_number(estimate.median())}, "
                    f"min {_format_number(estimate.min())}, "
                    f"max {_format_number(estimate.max())}"
                )
                share = (ok["pvalue"] < 0.05).mean()
                lines.append(f"  significant at 0.05: {share:.0%}")
        if self._null is not None:
            lines.append("")
            null = self._null
            lines.append(f"Under the null ({null.method}, {null.n_boot} resamples):")
            lines.append(null.summary.to_string())
        return "\n".join(lines)
