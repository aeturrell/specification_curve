## SpecificationCurve


Runs and plots every variant of a regression specification.


Usage

``` python
SpecificationCurve(
    data,
    formula=None,
    *,
    y=None,
    x=None,
    controls=None,
    always=None,
    exclude=None,
    min_controls=0,
    max_controls=None,
    fe=None,
    iv=None,
    model="feols",
    family=None,
    quantile=None,
    vcov=None,
    weights=None,
    subsets=None,
    common_sample=True,
    skip_invalid=False
)
```


A specification curve shows how the estimate of one coefficient (the focal regressor, `x`) changes across many defensible ways of specifying a model. The choices that can be varied are the outcome, the focal regressor, the control variables, the fixed effects, instruments, the estimator, the standard errors, the weights and the sample. All estimation is done by pyfixest, and every term is written in pyfixest's formula syntax, so transformations such as `log(wage)` and interactions such as `i(year)` can be used anywhere.

Specifications are described either with keyword arguments or with a pyfixest formula. In a formula, the multiple-estimation operators define the choices: several outcomes (`y1 + y2 ~`), `sw()`, `sw0()`, `csw()`, `csw0()`, and two operators added by this package, `comb()` (every non-empty subset of its arguments) and `comb0()` (every subset, including none of them).


## Parameters


`data: pd.DataFrame`  
The data.

`formula: Optional[str] = None`  
pyfixest formula describing the outcomes, regressors, fixed effects and instruments, for example `"y1 + y2 ~ x + comb0(a, b) | sw0(firm, firm + year)"`. Use this or `y`, not both.

`y: Union[str, Sequence[str], None] = None`  
Outcome term(s).

`x: Union[str, Sequence[str], None] = None`  
Focal regressor(s): the coefficient that is plotted, named as pyfixest reports it. Required. With several, each specification uses one of them. With instruments, this is usually the endogenous variable.

`controls: Union[str, Sequence[str], None] = None`  
Terms included in every combination.

`always: Union[str, Sequence[str], None] = None`  
Terms included in every specification.

`exclude: Optional[Sequence[Sequence[str]]] = None`  
Groups of controls that never appear together in a specification.

`min_controls: int = ``0`  
Smallest number of controls in a specification.

`max_controls: Optional[int] = None`  
Largest number of controls in a specification.

`fe: Union[str, Sequence[Optional[str]], None] = None`  
Fixed-effect sets, each written as in a pyfixest formula (`"firm + year"`). `None` in the list means no fixed effects.

`iv: Union[str, Sequence[Optional[str]], None] = None`  
First stage(s), for example `"x ~ z"`. `None` in the list means no instruments.

`model: Union[str, Sequence[str]] = ``"feols"`  
Estimator(s): "feols", "fepois", "feglm" or "quantreg".

`family: Union[str, Sequence[str], None] = None`  
Family or families for "feglm".

`quantile: Union[float, Sequence[float], None] = None`  
Quantile(s) for "quantreg". Defaults to 0.5.

`vcov: Union[VcovSpec, Sequence[VcovSpec]] = None`  
Variance-covariance type(s), as accepted by pyfixest, for example `"hetero"` or `{"CRV1": "firm"}`. `None` uses pyfixest's default.

`weights: Union[str, Sequence[Optional[str]], None] = None`  
Weight column(s). `None` in a list means unweighted.

`subsets: Optional[Mapping[str, Optional[str]]] = None`  
Named samples, each a `DataFrame.query` string or `None` for all rows.

`common_sample: bool = ``True`  
Within each subset, estimate every specification on the rows that are complete for all variables used by any of them, so that estimates are comparable. If False, each specification uses all the rows it can.

`skip_invalid: bool = ``False`  
Leave out combinations that an estimator cannot run (for example quantile regression with fixed effects) instead of raising.


## Attributes


`grid: pd.DataFrame`  
One row per specification, available before fitting.

`results: pd.DataFrame`  
The grid with estimates, available after [fit](SpecificationCurve.md#specification_curve.SpecificationCurve.fit).

`models: dict[int, Any]`  
Fitted pyfixest models keyed by `spec_id`, if `fit(keep_models=True)` was used.

`null_summary: pd.DataFrame`  
Tests of the curve against the null, after [fit_null](SpecificationCurve.md#specification_curve.SpecificationCurve.fit_null).

`null_curves: pd.DataFrame`  
Quantiles of the curve under the null, after [fit_null](SpecificationCurve.md#specification_curve.SpecificationCurve.fit_null).

`null_draws: pd.DataFrame`  
Estimates in every resample, after `fit_null(keep_draws=True)`.


## Examples

``` python
>>> import specification_curve as sc
>>> df = sc.load_example_data2()
>>> sco = sc.SpecificationCurve(df, y="y", x="x_1", controls=["x_2", "x_3"])
>>> sco.n_specs
4
>>> len(sco.fit().results)
4
```


## Attributes

| Name | Description |
|----|----|
| [results](#results) | The grid with estimates; one row per fitted specification. |
| [grid](#grid) | One row per specification. |
| [n_specs](#n_specs) | Number of specifications in the grid. |
| [null_curves](#null_curves) | Quantiles of the sorted curve under the null, one row per rank. |
| [null_summary](#null_summary) | Observed statistics of the curve and their p-values under the null. |
| [null_draws](#null_draws) | Estimates in every resample, if `fit_null(keep_draws=True)` was used. |

------------------------------------------------------------------------


#### results


The grid with estimates; one row per fitted specification.


`results: pd.DataFrame`


------------------------------------------------------------------------


#### grid


One row per specification.


`grid: pd.DataFrame`


------------------------------------------------------------------------


#### n_specs


Number of specifications in the grid.


`n_specs: int`


------------------------------------------------------------------------


#### null_curves


Quantiles of the sorted curve under the null, one row per rank.


`null_curves: pd.DataFrame`


------------------------------------------------------------------------


#### null_summary


Observed statistics of the curve and their p-values under the null.


`null_summary: pd.DataFrame`


------------------------------------------------------------------------


#### null_draws


Estimates in every resample, if `fit_null(keep_draws=True)` was used.


`null_draws: pd.DataFrame`


## Methods

| Name | Description |
|----|----|
| [fit()](#fit) | Estimates the specifications. |
| [plot()](#plot) | Plots the fitted specification curve. |
| [fit_null()](#fit_null) | Tests the curve against the null hypothesis of no effect. |

------------------------------------------------------------------------


#### fit()


Estimates the specifications.


Usage

``` python
fit(
    *,
    ci_levels=(0.95,),
    n_jobs=1,
    max_specs=None,
    seed=None,
    on_error="record",
    keep_models=False,
    batch=True,
    progress=None,
    **kwargs
)
```


##### Parameters


`ci_levels: Sequence[float] = (0.95,)`  
Confidence levels to store. Each gives a pair of columns such as `ci_low_95` and `ci_high_95`.

`n_jobs: int = ``1`  
Number of parallel workers.

`max_specs: Optional[int] = None`  
Estimate a random sample of this many specifications instead of the whole grid.

`seed: Optional[int] = None`  
Seed for the random sample.

`on_error: str = ``"record"`  
"record" keeps specifications that fail in the results with their error message; "raise" stops at the first failure.

`keep_models: bool = ``False`  
Keep the fitted pyfixest models in `.models`, keyed by `spec_id`. Specifications that differ only in `vcov` share one model object.

`batch: bool = ``True`  
Send specifications that share a sample, estimator, focal regressor and fixed effects to pyfixest in a single call. The results are the same either way.

`progress: Optional[bool] = None`  
Show a progress bar. By default one is shown when there are many pyfixest calls to make.

`**kwargs: Any`  
Passed to the pyfixest estimators, for example `ssc`, `fixef_rm`, `demeaner` or `context`. Each estimator receives the arguments it accepts.


##### Returns


`SpecificationCurve: SpecificationCurve`  
The object itself, so calls can be chained.


##### Raises


`TypeError`  
If a keyword argument is not accepted by any of the estimators in the grid.


------------------------------------------------------------------------


#### plot()


Plots the fitted specification curve.


Usage

``` python
plot(save_path=None, *, show_null_curve=False, **kwargs)
```


##### Parameters


`save_path: Union[str, os.PathLike[str], None] = None`  
File to save the figure to; the format is taken from the extension.

`show_null_curve: bool = ``False`  
Overlay the curve expected under the null. Requires [fit_null](SpecificationCurve.md#specification_curve.SpecificationCurve.fit_null) to have been run.

`**kwargs: Any`  
Passed to [plot_curve](plot_curve.md#specification_curve.plot_curve), for example `sort_by`, `color_by`, `panels`, `labels`, `highlight` or `fig`.


##### Returns


`CurvePlot: CurvePlot`  
The figure and its axes keyed by panel name. It unpacks as `fig, axes`.


------------------------------------------------------------------------


#### fit_null()


Tests the curve against the null hypothesis of no effect.


Usage

``` python
fit_null(
    n_boot=500,
    *,
    method="auto",
    cluster=None,
    strata=None,
    alpha=0.05,
    alternative="two-sided",
    seed=None,
    n_jobs=1,
    keep_draws=False,
    min_valid=0.5,
    progress=None
)
```


The curve is re-estimated on many resamples in which the null holds. Three statistics of the observed curve are then compared with their distribution across resamples: the median estimate, the share of significant estimates, and Stouffer's Z, which combines the evidence of all specifications. The results are in `.null_summary` and `.null_curves`.

With `method="bootstrap"`, each specification's estimated effect is removed from its outcome and rows are resampled with replacement. This applies when every specification uses `model="feols"`. `method="bayesian-bootstrap"` (the Bayesian bootstrap of Rubin, 1981; see Andrews, Moon and Shapiro, 2026) imposes the null in the same way and has the same requirement, but gives rows random Dirichlet weights instead of resampling them, so that no row (or cluster) is ever left out. With `method="permutation"`, the focal variable is shuffled between rows. This applies to any estimator, but it is only valid if the focal variable is as good as randomly assigned (within `strata`, if given), as in an experiment, and it does not apply to instrumental-variable specifications.


##### Parameters


`n_boot: int = ``500`  
Number of resamples. Use several hundred for results you rely on; every resample re-estimates the whole curve.

`method: str = ``"auto"`  
"bootstrap", "bayesian-bootstrap", "permutation" or "auto". "auto" uses the bootstrap if every specification uses "feols", and otherwise asks you to choose.

`cluster: Optional[str] = None`  
Column identifying clusters. Whole clusters are resampled (bootstrap), share one weight (Bayesian bootstrap) or swap their focal values (permutation; the focal variable must be constant within them).

`strata: Optional[str] = None`  
Column identifying strata. Resampling is done within each stratum.

`alpha: float = ``0.05`  
Significance level for the share of significant estimates.

`alternative: str = ``"two-sided"`  
"two-sided", "greater" (the effect is positive) or "less" (the effect is negative).

`seed: Optional[int] = None`  
Seed for the resampling. Results do not depend on `n_jobs`.

`n_jobs: int = ``1`  
Number of parallel workers.

`keep_draws: bool = ``False`  
Keep the estimates from every resample in `.null_draws`.

`min_valid: float = ``0.5`  
A resample is left out if fewer than this share of the specifications could be estimated in it.

`progress: Optional[bool] = None`  
Show a progress bar. By default one is shown when there are at least 20 resamples.


##### Returns


`SpecificationCurve: SpecificationCurve`  
The object itself, so calls can be chained.
