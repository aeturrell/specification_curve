## plot_curve()


Draws a specification curve.


Usage

``` python
plot_curve(
    results,
    null_curves=None,
    *,
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
    style=True
)
```


The top panel shows the focal estimate of every specification with its confidence interval(s). Below it, one panel per dimension that varies shows which choice each specification makes. Specifications that could not be estimated are left out.

The function works on any data frame with the columns of [SpecificationCurve.results](SpecificationCurve.md#specification_curve.SpecificationCurve.results), so results that were saved, filtered or combined from several curves can be drawn too. It does not call `plt.show()` and does not change matplotlib's global settings.


## Parameters


`results: pd.DataFrame`  
Results as produced by [SpecificationCurve.fit](SpecificationCurve.md#specification_curve.SpecificationCurve.fit).

`null_curves: Optional[pd.DataFrame] = None`  
Quantiles of the curve under the null, as produced by [SpecificationCurve.fit_null](SpecificationCurve.md#specification_curve.SpecificationCurve.fit_null). Can only be shown when sorting by estimate.

`alpha: float = ``0.05`  
Significance level used when colouring by significance.

`ci_levels: Optional[Sequence[float]] = None`  
Which of the stored confidence levels to draw. By default, all of them.

`sort_by: Union[str, Sequence[str]] = ``"estimate"`  
Order of the specifications: "estimate", another column of the results, or a list of columns. Ties are ordered by estimate.

`color_by: str = ``"significance"`  
"significance" colours estimates by sign and significance. A dimension ("y", "x", "fe", "iv", "vcov", "model", "weights" or "subset") colours by the value of that dimension and adds a legend.

`panels: Optional[Sequence[str]] = None`  
Which dimensions to show under the curve, in order. By default, every dimension that varies.

`labels: Optional[Mapping[str, str]] = None`  
Replacement display text for panel titles (keyed by dimension, e.g. `{"fe": "Fixed effects"}`) and for values (keyed by the value, e.g. `{"firm + year": "Firm and year"}`).

`highlight: Union[Selector, Sequence[Selector], None] = None`  
A preferred specification to mark and annotate, given as a `spec_id`, a dict of column values (e.g. `{"y": "log(wage)", "controls": ["age"], "fe": None}`) or a query string. It must match exactly one specification. Pass a list to mark several.

`show_median: bool = ``True`  
Draw a line at the median estimate.

`show_nobs: bool = ``False`  
Add a panel with the number of observations used by each specification.

`colors: ColorSpec = None`  
Colours to use. When colouring by significance, a dict with any of the keys "positive", "negative", "insignificant". When colouring by a dimension, a list of colours or a dict keyed by the dimension's values.

`figsize: Optional[tuple[float, float]] = None`  
Figure size in inches. By default, chosen from the number of panels.

`fig: Union[Figure, SubFigure, None] = None`  
An existing figure or subfigure to draw into. By default a new figure is created.

`style: bool = ``True`  
Use this package's fonts and sizes for the figure.


## Returns


`tuple[Union[Figure, SubFigure], dict[str, Axes]]`  
tuple\[Union\[Figure, SubFigure\], dict\[str, Axes\]\]: The figure and its axes keyed by panel name: "curve", then the dimensions shown, then "nobs" if requested.


## Raises


`ValueError`  
If no specification was estimated successfully, or an argument does not fit the results.
