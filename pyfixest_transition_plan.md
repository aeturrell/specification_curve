# Plan: rebuild Specification Curve on pyfixest

## 1. Goal and ground rules

Make it quick and flexible to draw specification curves for a wide range of
econometric problems by using [pyfixest](https://github.com/py-econometrics/pyfixest)
as the only estimation back-end.

Ground rules agreed for this work:

- pyfixest is the single back-end. Its formula interpreter, its fixed-effect
  absorption (demeaning) and its four estimators (`feols`, `fepois`, `feglm`,
  `quantreg`) are used directly. There is no plug-in estimator interface and no
  statsmodels path.
- There is no backwards compatibility. The code, docstrings, docs, README and
  release notes describe the package as if it had always worked this way: no
  deprecation shims, no aliases for old names, no "previously" or "migration"
  text anywhere. This plan file is the only place the old design is referred
  to, and it is deleted in PR 3.
- Work lands as three PRs: (1) core, (2) inference and plotting, (3) docs,
  benchmark and release. Each PR keeps CI green (pre-commit, ty, tests at
  98% coverage, typeguard, xdoctest).

## 2. What was checked against pyfixest

Everything in this section was run against pyfixest 0.70.0 (requires
Python >= 3.11, pandas >= 2.3.3). The design below depends on these facts.

| Fact | Consequence for the design |
|---|---|
| `sw()`/`sw0()` accept sums (`sw0(a, b, a + b)`), several can appear on the right-hand side, and they work in the fixed-effects slot (`\| sw0(f1, f1 + f2)`) | A whole block of specifications compiles to one pyfixest call, which shares the demeaning work |
| Multiple outcomes work as `Y1 + Y2 ~ ...` | Outcomes join the same single call |
| `fit.vcov(...)` recomputes standard errors in place without refitting, for all four estimators | The standard-error dimension costs no extra fits |
| `fit.vcov(...)` fails if the model was fitted with `lean=True` or `store_data=False` | Fit with full storage, extract numbers, then drop the model object |
| `copy_data=False` mutates the data frame passed in | Always pass pyfixest a frame the engine owns |
| Default `vcov` is `"iid"` with or without fixed effects | Default is predictable; the label is always recorded in results |
| `fit.tidy(alpha=)` and `fit.confint(alpha=)` give intervals at any level | Several confidence levels can be stored per fit |
| `quantreg` rejects fixed effects, has no `weights` argument, and returns a multi-model object for a list of quantiles | Validated up front; quantile is its own dimension |
| IV (`y ~ exog \| fe \| endog ~ instr`) works in `feols` only; `fepois` and `feglm` raise `NotImplementedError` | Validated up front |
| `feglm` families are `logit`, `probit`, `gaussian`, `poisson`, and all accept fixed effects | Family is part of the model dimension |
| Each fit exposes `capabilities` (for example `crv3_inference`, `multiway_clustering`, `hac_inference`) | Standard-error validity is checked from pyfixest, not a table we maintain |
| Clustered standard errors fail when the cluster variable has missing values | The engine removes those rows before fitting (see sample handling) |
| Formula terms can be transforms and interactions (`log(x)`, `i(f)`, `C(f)`, `a:b`), and `context=` supplies user functions | No term parsing of our own is needed |
| `pyfixest.estimation.formula.parse.Formula.parse` gives outcome, exogenous terms, fixed effects, endogenous terms and instruments; the function that expands multiple-estimation operators is private (`_expand_all_multiple_estimation`) | All parser access goes through one adapter module, and pyfixest is pinned to a tested range |
| On 200,000 rows with two fixed effects, 12 separate fits took 1.43 s and the same 12 models as four `sw()` calls took 1.15 s | Batching helps but is not dramatic at this size; it must be benchmarked, not assumed |

Not yet checked, to confirm during PR 1: pyfixest wheels install cleanly on
the Windows and macOS CI runners; `weights` with `feglm`; the exact exception
types raised for each unsupported combination.

## 3. Target user interface

### 3.1 Constructing a curve

```python
import specification_curve as sc

sco = sc.SpecificationCurve(
    data,
    y=["wage", "log(wage)"],
    x="training",
    controls=["age", "tenure", "i(sector)"],
    always=["female"],
    exclude=[["age", "tenure"]],
    fe=[None, "firm", "firm + year"],
    vcov=["hetero", {"CRV1": "firm"}],
    subsets={"all": None, "post-2000": "year > 2000"},
    model=["feols", "fepois"],
)
sco.fit()
sco.results          # tidy frame, one row per specification
sco.plot()
```

Or with a pyfixest formula, where the multiple-estimation operators define the
choices:

```python
sco = sc.SpecificationCurve(
    data,
    formula="wage + log(wage) ~ training + female + comb0(age, tenure) | sw0(firm, firm + year)",
    x="training",
    vcov=["hetero", {"CRV1": "firm"}],
)
```

Arguments:

| Argument | Type | Meaning | Default |
|---|---|---|---|
| `data` | DataFrame | Any frame pyfixest accepts | required |
| `formula` | str | pyfixest formula; may use `sw`, `sw0`, `csw`, `csw0`, several outcomes, and `comb`/`comb0` | `None` |
| `y` | str or list | Outcome terms | `None` |
| `x` | str or list | Focal regressor(s): the coefficient plotted. Required in both forms | required |
| `controls` | list | Terms included in every subset combination | `[]` |
| `always` | list | Terms in every specification | `[]` |
| `exclude` | list of lists | Groups of controls that never appear together | `[]` |
| `max_controls` / `min_controls` | int | Bounds on subset size | all sizes |
| `fe` | list of str or None | Fixed-effect sets; `None` means none | `[None]` |
| `iv` | str or list | First-stage part, e.g. `"training ~ lottery"` | `None` |
| `model` | str or list | `"feols"`, `"fepois"`, `"feglm"`, `"quantreg"` | `"feols"` |
| `family` | str or list | For `feglm` | `None` |
| `quantile` | float or list | For `quantreg` | `0.5` |
| `vcov` | str, dict or list of those | Standard-error types | `"iid"` |
| `weights` | str, None or list | Weight column(s) | `None` |
| `subsets` | dict of name to query string | Sample restrictions; `None` means all rows | `{"all": None}` |
| `common_sample` | bool | Use one complete-case sample for all specifications within a subset | `True` |

Rules:

- Give either `formula` or `y` (with optional `controls`, `always`, `fe`,
  `iv`). Giving both raises `ValueError`. `x`, `model`, `family`, `quantile`,
  `vcov`, `weights`, `subsets` apply in both forms.
- `x` values are coefficient names as pyfixest reports them (`"training"`,
  `"log(training)"`, `"training:female"`). With several `x` values each
  specification uses exactly one of them. With IV, the focal coefficient is the
  endogenous variable.
- `comb(a, b, c)` means every non-empty subset and `comb0(a, b, c)` adds the
  empty subset. They are the only syntax this package adds. A small
  balanced-parenthesis pre-processor rewrites them into `sw(...)`/`sw0(...)`
  with enumerated sums before pyfixest sees the formula. The names are open to
  change (see section 8).
- `exclude`, `min_controls` and `max_controls` filter the grid after expansion,
  so they work with the formula form too.
- A helper `sc.subsets_by(data, "region")` returns the `subsets` dict with one
  query per level, for the common "run it on each group" case.

### 3.2 Fitting

```python
sco.fit(
    ci_levels=(0.95,),
    n_jobs=1,
    max_specs=None,
    seed=None,
    on_error="record",
    keep_models=False,
    demeaner=None,
    **pyfixest_kwargs,
)
```

- `ci_levels`: every level is stored as a `ci_low_XX` / `ci_high_XX` pair.
- `max_specs` with `seed`: fit a random sample of the grid when it is too
  large; the full grid size is still reported.
- `on_error="record"`: a specification that fails is kept in results with its
  error message and missing estimates; `"raise"` stops at the first failure.
- `keep_models=True`: keep the pyfixest model objects in `sco.models`
  (a dict keyed by `spec_id`) for users who want `etable`, `wildboottest`, etc.
- `demeaner`: passed to pyfixest (`MapDemeaner`, `LsmrDemeaner`).
- Other keyword arguments (`ssc`, `fixef_rm`, `collin_tol`, `context`, ...)
  pass straight through to the pyfixest estimator.
- `fit()` returns `self` and does not change constructor state, so it can be
  called repeatedly.

### 3.3 Results

`sco.results` is a tidy DataFrame, sorted by `spec_id` (grid order, stable
across runs), with these columns:

| Column | Content |
|---|---|
| `spec_id` | Integer id from the grid |
| `model`, `family`, `quantile` | Estimator settings |
| `y`, `x` | Outcome term and focal coefficient |
| `controls` | Tuple of varying control terms included |
| `always` | Tuple of always-included terms |
| `fe` | Fixed-effect string or `None` |
| `iv` | First-stage string or `None` |
| `vcov` | Readable label: `"iid"`, `"hetero"`, `"CRV1: firm"` |
| `weights`, `subset` | Weight column and subset name |
| `formula` | The exact single-model pyfixest formula |
| `estimate`, `se`, `statistic`, `pvalue` | For the focal coefficient |
| `ci_low_95`, `ci_high_95`, ... | One pair per requested level |
| `nobs` | Observations used |
| `error` | `None` or the exception text |

Also available: `sco.grid` (the same frame without estimate columns, available
before fitting), `sco.n_specs`, and a `__repr__` that reports the dimensions,
their sizes, the estimators actually used, and median / min / max estimate plus
the share significant once fitted.

## 4. Architecture

```
src/specification_curve/
    __init__.py      public names and __version__ only
    _pyfixest.py     the only module that touches pyfixest internals
    formula.py       comb()/comb0() pre-processor; formula <-> grid rows
    grid.py          dimensions, combination logic, validation, sampling
    engine.py        fit groups, batched and single execution, extraction
    curve.py         SpecificationCurve (thin orchestration)
    inference.py     tests under the null (PR 2)
    plot.py          plot_curve and helpers (PR 2)
    data.py          example data loaders
    data/            csv files
```

Design principles:

- The grid is plain data (a DataFrame). The engine is a pure function
  `run_grid(grid, data, options) -> results`. Inference and plotting depend
  only on the grid and results frames. This is what makes PR 2 independent of
  PR 1's internals, and lets inference re-run the engine on resampled data.
- `_pyfixest.py` wraps: formula expansion, `Formula.parse`, estimator
  dispatch, capability lookup, extraction of focal-coefficient numbers and
  `nobs`. If pyfixest changes an internal, one file changes.
- pyfixest is pinned to a tested minor range in `pyproject.toml` and the upper
  bound is raised by a test run, not by default.

## 5. PR 1: core

### 5.1 Dependencies and packaging

- Runtime dependencies become: `pyfixest`, `pandas>=2.3.3`, `numpy`,
  `matplotlib`, `joblib`, `scipy`, `tqdm`.
- Removed from runtime: `statsmodels`, `pingouin`, `typeguard`,
  `pandas-stubs`, `types-setuptools`, `types-tqdm`, `nbclient`, `ipywidgets`.
  Stub packages and notebook tooling move to the dev group where still needed.
- Remove `typeguard_ignore` imports from the package (the typeguard nox session
  installs typeguard itself).
- Regenerate `uv.lock`. Confirm the CI matrix (Linux, Windows, macOS; Python
  3.12 and 3.13) installs pyfixest.

### 5.2 `formula.py`

1. `expand_comb(formula) -> str`: find `comb(`/`comb0(` with a
   balanced-parenthesis scanner (terms can contain commas and parentheses, as
   in `i(f1, ref=1)`), enumerate subsets, and emit `sw(...)`/`sw0(...)` with
   sums. Reject nested `comb` and `comb` inside the outcome or first-stage
   parts with a clear error.
2. `expand(formula) -> list[str]`: via `_pyfixest.py`, expand all
   multiple-estimation operators into single-model formulas.
3. `parse_single(formula) -> SpecParts`: via `_pyfixest.py`, return outcome,
   right-hand-side terms, fixed effects, endogenous term and instruments.
4. `render(parts) -> str`: build a single-model pyfixest formula from parts.
   Round-trip property `parse_single(render(p)) == p` is tested.

### 5.3 `grid.py`

1. Normalise every argument to a list (scalars become one-element lists).
2. Build the structural rows (outcome, focal regressor, controls subset,
   always, fixed effects, IV):
   - keyword form: `itertools` product, with control subsets enumerated between
     `min_controls` and `max_controls`;
   - formula form: `expand` then `parse_single`; right-hand-side terms are split
     into focal (in `x`), always (present in every expanded model) and controls
     (present in some).
3. Apply `exclude`: drop rows whose controls contain every member of any
   excluded group. Matching is on exact term strings, never substrings.
4. Remove from `controls` anything that is also in `always` or equal to the
   row's focal regressor or outcome, then drop duplicate rows.
5. Cross with `model` (with `family` / `quantile` attached to the estimators
   they belong to), `vcov`, `weights`, `subsets`.
6. Validate, raising `ValueError` with the offending combination named:
   - no focal regressor, or more than one, in a specification;
   - `quantreg` with fixed effects, weights or IV;
   - `fepois` / `feglm` with IV;
   - `feglm` without `family`; `family` or `quantile` given with no estimator
     that uses it;
   - terms referring to columns not in `data` (using the variables formulaic
     reports for each term);
   - a subset query that selects no rows.
   Combinations that are invalid for some estimators but valid for others
   (for example `fe` with a model list containing `quantreg`) are an error by
   default, with an argument `skip_invalid=True` to drop them and report how
   many were dropped.
7. Assign `spec_id`. Provide `sample(n, seed)` returning a row subset.
8. Guard rail: warn above a threshold (for example 10,000 specifications)
   with the count and a pointer to `max_specs`.

### 5.4 `engine.py`

Sample handling, per subset:

- Apply the query. If `common_sample=True`, drop rows with a missing value in
  any variable used by any specification in that subset, including cluster and
  weight columns, so `nobs` is constant within the subset.
- If `common_sample=False`, pyfixest drops missing rows per model; the engine
  additionally drops rows with missing cluster variables for the specifications
  that cluster on them.
- The engine works on its own copy of each subset frame, made once.

Fit groups:

- A fit group is all specifications sharing subset, model, family, quantile,
  weights and IV part, and differing only in outcome, focal regressor,
  controls, fixed effects and `vcov`.
- `vcov` is never a reason to refit: fit once, then for each `vcov` in the
  group call `fit.vcov(v)` and extract immediately (the call mutates the
  model, so extraction happens before the next call).
- Standard-error types the fitted model does not support (per
  `fit.capabilities`) are recorded as errors for those rows only.

Two execution paths, built in this order:

1. Single path (reference implementation): one pyfixest call per unique
   structural row, using `formula` from the grid. Simple and obviously correct.
2. Batched path: within a fit group, and for each fixed-effect set, emit one
   call `y1 + y2 ~ always + sw(<x and control combos as sums>) | fe`. Results
   are mapped back to grid rows by position and then verified against each
   fitted model's outcome, fixed effects and coefficient names; any mismatch
   falls back to the single path for that group. Sampling with `max_specs`
   simply shortens the `sw(...)` list.

The batched path is enabled only if the benchmark in 5.7 shows a worthwhile
gain; correctness is guaranteed by a test that both paths give identical
results on every test grid.

Parallelism: `joblib.Parallel` over fit groups with `n_jobs`. Default is 1
because the demeaner may already be multithreaded; the benchmark decides
whether to recommend more.

Extraction: for each fit and `vcov`, read estimate, standard error, statistic
and p-value for the focal coefficient from `tidy()`, confidence intervals from
`confint(alpha=1 - level)` per level, and `nobs`. If the focal coefficient was
dropped for collinearity, record that as the row's error.

Progress: a `tqdm` bar over fit groups, shown only above a size threshold and
switchable with `progress=`.

### 5.5 `curve.py`

- `SpecificationCurve.__init__` validates arguments and builds the grid
  eagerly, so mistakes surface before any estimation.
- `fit()` calls the engine and stores `results` (and `models` if requested).
- `plot()` and `fit_null()` delegate to `plot.py` and `inference.py`. In PR 1
  these two modules are simple, working versions built on the results frame:
  the current chart layout redrawn from `results`, and the null-curve routine
  re-running the engine. PR 2 replaces their internals without changing how
  PR 1's classes call them.
- `__repr__` as described in 3.3.

### 5.6 Example data

- Keep the three existing loaders, moved to `data.py`.
- Add `load_example_panel()`: a seeded synthetic firm-year panel with a
  treatment, a count outcome, a binary outcome, firm and year identifiers, a
  cluster variable, an instrument, weights and a few missing values. It
  exercises every dimension and is used by the tests and docs.
- All loaders take or create their own seeded generator; no module-level
  random state.

### 5.7 Tests and benchmark for PR 1

Replace `tests/test_specification_curve.py` with one file per module.

- `test_formula.py`: `comb`/`comb0` expansion including nested parentheses and
  keyword arguments inside terms; error cases; render/parse round trip.
- `test_grid.py`: counts for each dimension and their product; `exclude`;
  `min_controls`/`max_controls`; overlap removal; formula form and keyword form
  produce the same grid for equivalent inputs; every validation error;
  `skip_invalid`; deterministic `sample`.
- `test_engine.py`:
  - every result row equals a direct pyfixest call with the row's `formula`
    and settings (estimate, standard error, p-value, interval, `nobs`), for
    each of the four estimators, each `feglm` family, IV, weights, fixed
    effects and several `vcov` types;
  - batched path equals single path on all of the above;
  - `vcov` variants come from one fit (assert on the number of estimator calls
    using a counting wrapper around the adapter);
  - `common_sample` true and false; missing cluster values;
  - `on_error` record and raise; collinear focal regressor;
  - `n_jobs=2` equals `n_jobs=1`; `max_specs` with a seed is reproducible;
  - the caller's data frame is unchanged after `fit()`.
- `test_curve.py`: constructor argument rules, `repr` before and after fit,
  repeated `fit()` gives identical results, `keep_models`.
- Smoke tests for `plot()` and `fit_null()` so coverage holds until PR 2.

`benchmarks/benchmark.py` is rewritten to time, on the panel data at two
sizes: single against batched execution, `n_jobs` 1 against 4, and the cost of
adding `vcov` variants. The numbers decide the engine defaults and are quoted
in the PR description.

### 5.8 Done when

- All four estimators run through every dimension they support, and results
  match direct pyfixest calls.
- CI is green on the full matrix with coverage at or above 98%.
- No statsmodels or pingouin import remains.

## 6. PR 2: inference and plotting

These two parts depend only on the grid and results frames and can be
reviewed independently. They are one PR to keep the chart's null-curve overlay
and the inference output in step.

### 6.1 Inference under the null (`inference.py`)

Follows Simonsohn, Simmons and Nelson (2020).

Test statistics, computed on the observed curve and on each resample:

1. Median estimate across specifications.
2. Share of specifications that are significant with the dominant sign
   (`alpha` configurable).
3. Stouffer's Z: the sum of per-specification z-values divided by the square
   root of the number of specifications.

p-values are the share of resamples at least as extreme as observed
(two-sided by default, `alternative=` for one-sided), computed as
`(1 + count) / (B + 1)`.

Resampling schemes:

- `method="bootstrap"` (forced null; `feols` only, including IV and fixed
  effects). For each specification k, build `y*_k = y_k - b_k * x_k` using the
  outcome and focal regressor as pyfixest evaluates them (so transformed terms
  are handled), then in each resample draw rows with replacement (whole
  clusters when `cluster=` is given), using the same draw for every
  specification, and refit each specification with `y*_k` as the outcome.
  Cost is `n_boot` times the number of specifications.
- `method="permutation"` (any estimator). In each resample, permute the
  underlying columns of all focal regressors jointly, optionally within
  `strata=` or as whole blocks by `cluster=`, and re-run the whole grid through
  the batched engine. Valid when the focal variable is as good as randomly
  assigned; the docstring says so plainly.
- `method="auto"` uses the bootstrap when every specification is `feols` and
  otherwise raises an error asking the user to choose the permutation method
  explicitly, because its validity is an assumption the user has to own.

Interface:

```python
sco.fit_null(n_boot=500, method="auto", cluster=None, strata=None,
             alpha=0.05, alternative="two-sided", seed=None, n_jobs=1)
sco.null_summary     # three rows: observed statistic, p-value
sco.null_curves      # quantiles (2.5%, 50%, 97.5%) of the sorted null curve, by rank
```

- Null curves are sorted within each resample and summarised by rank, which is
  what the expected-curve overlay in the paper shows.
- Resamples run in parallel with `joblib`; each gets a child generator spawned
  from `seed`, so results are reproducible for any `n_jobs`.
- Only quantiles and the three statistics per resample are kept unless
  `keep_draws=True`.
- Specifications that fail in a resample are counted and reported; a resample
  is dropped from a statistic only if too few specifications survive
  (threshold configurable).

Tests:

- Statistics computed by hand on a small fixed results frame.
- Calibration: on data simulated with no effect, p-values over repeated
  simulations are roughly uniform (loose tolerance, marked slow); with a strong
  effect all three p-values are small.
- Bootstrap construction: on the full sample with no resampling, refitting
  `y*` gives a focal estimate of zero to numerical precision.
- Cluster bootstrap keeps clusters intact; strata are respected.
- Reproducibility across `seed` and `n_jobs`.
- `auto` raises for mixed-estimator grids; bootstrap raises for non-`feols`.

### 6.2 Plotting (`plot.py`)

A standalone function that also backs `SpecificationCurve.plot`:

```python
fig, axes = sc.plot_curve(
    results,
    null_curves=None,
    alpha=0.05,
    ci_levels=None,
    sort_by="estimate",
    color_by="significance",
    panels=None,
    labels=None,
    highlight=None,
    show_median=True,
    show_nobs=False,
    colors=None,
    figsize=None,
    fig=None,
    style=True,
)
```

Behaviour:

- Works on any frame with the results schema, so curves from results saved
  earlier, filtered, or concatenated across runs can be drawn.
- Top panel: estimates in `sort_by` order (`"estimate"`, any results column,
  or a list of columns to group then sort), with one error-bar layer per stored
  confidence level, narrower levels drawn heavier. `ci_levels` selects among
  the stored ones.
- Lower panels: one per dimension that takes more than one value (outcome,
  focal regressor, each control, fixed effects, standard errors, subset, model,
  weights, quantile). `panels` selects and orders them; `labels` renames
  dimensions and values for display.
- Markers are drawn with one vectorised call per panel (`scatter` with square
  markers, switching to thin vertical ticks above a specification-count
  threshold), not one patch per cell.
- `color_by="significance"` colours by sign and significance at `alpha`;
  any dimension name colours by that dimension with a legend. `colors`
  overrides the palette.
- `highlight` marks a preferred specification, given as a `spec_id`, a dict of
  dimension values, or a query string; it is annotated with its estimate. More
  than one match raises unless `highlight` is a list.
- `null_curves` overlays the null median and its band.
- `show_nobs=True` adds a thin bar panel of observations per specification,
  useful when `common_sample=False`.
- Failed specifications are left out of the chart and the count is noted in a
  small caption.
- The function never calls `plt.show()`, never closes other figures and never
  changes global matplotlib settings; the package style is applied inside
  `matplotlib.rc_context` and turned off with `style=False`. It returns the
  figure and a dict of axes keyed by panel name. Passing `fig` draws into an
  existing figure or subfigure.
- `sco.plot(save_path=...)` is a thin wrapper that calls `plot_curve` with the
  object's results and null curves and optionally saves.

Tests (structural, no image comparison): number and names of axes for a given
grid; number of plotted points equals successful specifications; colours match
sign and significance at a chosen `alpha`; sort orders; `panels` and `labels`;
each form of `highlight` and its error cases; drawing into a supplied figure;
global `rcParams` unchanged after the call; large-grid marker switch; saving to
a temporary path.

### 6.3 Done when

- `fit_null` and `plot_curve` are implemented as above with tests, coverage is
  at or above 98%, and the simple PR 1 versions are gone.
- `pingouin` is not imported anywhere (it was removed from dependencies in
  PR 1; this confirms no code path wants it back).

## 7. PR 3: docs, benchmark and release

1. README: short description, install, a ten-line quickstart using the panel
   example with fixed effects and clustered standard errors, and the image it
   produces.
2. `docs/index.ipynb`: introduction (kept), quickstart in both keyword and
   formula forms, pointer to pyfixest's formula documentation.
3. `docs/features.ipynb`, rewritten as short sections, each with one runnable
   example:
   - choosing outcomes, focal regressors and controls; `always`, `exclude`,
     size limits;
   - formula form with `sw`, `csw`, `comb`;
   - fixed effects and how absorption works;
   - standard errors as a dimension;
   - subsets and `subsets_by`; `common_sample`;
   - Poisson, logit/probit and quantile regression curves;
   - instrumental variables;
   - weights;
   - reading and exporting `results`;
   - large grids: `max_specs`, `n_jobs`, the dense chart style;
   - customising the chart: sorting, colouring, panels, labels, highlight,
     drawing into your own figure;
   - inference under the null: both methods and when each is valid.
4. `docs/_quarto.yml` reference: `SpecificationCurve`, `plot_curve`,
   `subsets_by`, the data loaders. Rebuild `objects.json`.
5. `docs/contributing.md`: update the development commands if they changed.
6. Benchmark: final `benchmarks/benchmark.py` with the scenarios from 5.7 plus
   null inference, and a short results table in the docs' large-grids section.
7. Packaging: update the project description and classifiers; confirm the
   `Makefile` docs build and the release workflow run end to end; bump the
   version with `version_bumper.py` (number to be agreed, see section 8).
8. Sweep: search the repository for any wording that implies an earlier
   design (for example "previously", "deprecated", "no longer", "legacy",
   "statsmodels") and remove it. Delete this plan file.

Done when the docs site builds with every notebook executing, the release
workflow passes, and the sweep finds nothing.

## 8. Risks and open points

| Item | Risk or question | Handling |
|---|---|---|
| pyfixest internals | Formula expansion is a private function and the parsed-formula interface changed in 0.70.0 | Single adapter module, pinned version range, tests that fail loudly on a pyfixest upgrade |
| Batched execution | Mapping multi-model output back to grid rows could silently misalign | Position mapping is verified against each model's outcome, fixed effects and coefficient names, with fallback to single fits |
| Batching gain | Measured gain was modest in one quick test | Benchmark in PR 1 decides whether it is on by default |
| Windows/macOS wheels | Not yet verified in CI | First thing checked in PR 1 |
| Permutation inference | Easy to misuse on observational data | Never chosen automatically for non-`feols` grids; assumption stated in the docstring and docs |
| Grid explosion | Dimensions multiply quickly | Count reported before fitting, warning threshold, `max_specs` |
| `comb` / `comb0` names | Our only added syntax; names are a suggestion | Decide before PR 1 merges |
| Default `vcov` | pyfixest's default is `"iid"`; `"hetero"` may be the better default for this audience | Decide before PR 1 merges |
| Version number | 1.0.0 or a 0.x bump | Decide in PR 3 |
