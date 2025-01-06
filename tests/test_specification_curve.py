#!/usr/bin/env python
"""Tests for `specification_curve` package."""

import os
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import specification_curve as specy
import statsmodels.api as sm
from pytest import raises
from scipy.stats import norm
from specification_curve import _parse_formula
from typeguard import typeguard_ignore


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_000_basic_plot(mock_show) -> None:
    df = specy.load_example_data2()
    y_endog = "y"
    x_exog = "x_1"
    controls = ["x_2", "x_3", "x_4"]
    sc = specy.SpecificationCurve(df, y_endog, x_exog, controls)
    sc.fit()
    sc.plot()
    mock_show.assert_called_once()


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_001_fe_grp(mock_show) -> None:
    """Test expand multiple FE groups. Docs feature 1."""
    df = specy.load_example_data1()
    y_endog = "y1"
    x_exog = "x1"
    controls = ["c1", "c2", "group1", "group2"]
    sc = specy.SpecificationCurve(
        df, y_endog, x_exog, controls, cat_expand=["group1", "group2"]
    )
    sc.fit()
    sc.plot()
    mock_show.assert_called_once()


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_002_docs_feat_two(mock_show) -> None:
    """Test docs feature 2."""
    df = specy.load_example_data1()
    y_endog = "y1"
    x_exog = "x1"
    controls = ["c1", "c2", "group1", "group2"]
    sc = specy.SpecificationCurve(
        df, y_endog, x_exog, controls, exclu_grps=[["c1", "c2"]]
    )
    sc.fit()
    sc.plot()
    mock_show.assert_called_once()


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_004_docs_feat_three(mock_show) -> None:
    """Test docs feature 3: multiple dependent or independent variables"""
    df = specy.load_example_data1()
    x_exog = ["x1", "x2"]
    y_endog = "y1"
    controls = ["c1", "c2", "group1", "group2"]
    sc = specy.SpecificationCurve(df, y_endog, x_exog, controls)
    sc.fit()
    sc.plot()
    mock_show.assert_called_once()


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_005_save_fig(mock_show) -> None:
    """Test save fig.."""
    df = specy.load_example_data1()
    x_exog = ["x1", "x2"]
    y_endog = "y1"
    controls = ["c1", "c2", "group1", "group2"]
    sc = specy.SpecificationCurve(df, y_endog, x_exog, controls, cat_expand=["group1"])
    sc.fit()
    sc.plot(save_path="test_fig.pdf")
    os.remove("test_fig.pdf")
    mock_show.assert_called_once()


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_006_logit_estimator(mock_show) -> None:
    """Test running with different statsmodels estimators -
    here logitistic
    """
    n_samples = 1000
    x_2 = np.random.randint(2, size=n_samples)
    x_1 = np.random.random(size=n_samples)
    x_3 = np.random.randint(3, size=n_samples)
    x_4 = np.random.random(size=n_samples)
    x_5 = x_1 + 0.05 * np.random.randn(n_samples)
    x_beta = -1 + 3.5 * x_1 + 0.2 * x_2 + 0.3 * x_3
    prob = 1 / (1 + np.exp(-x_beta))
    y = np.random.binomial(n=1, p=prob, size=n_samples)
    y2 = np.random.binomial(n=1, p=prob * 0.98, size=n_samples)
    df = pd.DataFrame(
        [x_1, x_2, x_3, x_4, x_5, y, y2],
        ["x_1", "x_2", "x_3", "x_4", "x_5", "y", "y2"],
    ).T
    y_endog = ["y", "y2"]
    x_exog = ["x_1", "x_5"]
    controls = ["x_3", "x_2", "x_4"]
    sc = specy.SpecificationCurve(df, y_endog, x_exog, controls, cat_expand="x_3")
    sc.fit(estimator=sm.Logit)
    sc.plot()
    mock_show.assert_called_once()


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_007_probit_estimator(mock_show) -> None:
    """
    Running with different statsmodels estimators - here probit
    """
    n_samples = 1000
    x_2 = np.random.randint(2, size=n_samples)
    x_1 = np.random.random(size=n_samples)
    x_3 = np.random.randint(3, size=n_samples)
    x_4 = np.random.random(size=n_samples)
    x_5 = x_1 + 0.05 * np.random.randn(n_samples)
    x_beta = -1 + 3.5 * x_1 + 0.2 * x_2 + 0.3 * x_3
    prob = norm.cdf(x_beta)
    y = np.random.binomial(n=1, p=prob, size=n_samples)
    y2 = np.random.binomial(n=1, p=prob * 0.98, size=n_samples)
    df = pd.DataFrame(
        [x_1, x_2, x_3, x_4, x_5, y, y2],
        ["x_1", "x_2", "x_3", "x_4", "x_5", "y", "y2"],
    ).T
    y_endog = ["y", "y2"]
    x_exog = ["x_1", "x_5"]
    controls = ["x_3", "x_2", "x_4"]
    sc = specy.SpecificationCurve(df, y_endog, x_exog, controls, cat_expand="x_3")
    sc.fit(estimator=sm.Probit)
    sc.plot()
    mock_show.assert_called_once()


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_008_large_no_specifications(mock_show) -> None:
    """
    Test a very large set of specifications
    """
    n_samples = 400
    # Number of dimensions of continuous
    # random variables
    n_dim = 8
    c_rnd_vars = np.random.random(size=(n_dim, n_samples))
    c_rnd_vars_names = [f"c_{i}" for i in range(np.shape(c_rnd_vars)[0])]
    y_1 = 0.3 * c_rnd_vars[0, :] + 0.5 * c_rnd_vars[1, :]
    y_2 = y_1 + 0.05 * np.random.randn(n_samples)
    df = pd.DataFrame([y_1, y_2], ["y1", "y2"]).T
    for i, col_name in enumerate(c_rnd_vars_names):
        df[col_name] = c_rnd_vars[i, :]
    controls = c_rnd_vars_names[1:]
    sc = specy.SpecificationCurve(df, ["y1", "y2"], c_rnd_vars_names[0], controls)
    sc.fit()
    sc.plot()
    mock_show.assert_called_once()


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_009_always_include(mock_show) -> None:
    """Test of always include."""
    df = specy.load_example_data1()
    x_exog = "x1"
    y_endog = "y1"
    controls = ["c2", "group1", "group2"]
    sc = specy.SpecificationCurve(df, y_endog, x_exog, controls, always_include="c1")
    sc.fit()
    sc.plot()
    mock_show.assert_called_once()


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_010_preferred_specification(mock_show) -> None:
    """Test of labelling preferred specification."""
    df = specy.load_example_data1()
    x_exog = "x1"
    y_endog = "y1"
    controls = ["c2", "group1", "group2"]
    sc = specy.SpecificationCurve(df, y_endog, x_exog, controls)
    sc.fit()
    sc.plot(preferred_spec=["group1", "x1", "y1"])
    mock_show.assert_called_once()


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_011_formula_parsing(mock_show) -> None:
    formula = "y | y1 ~ x | x1 + c + c2 | c3"
    result = _parse_formula(formula)
    # Print results
    for category, variables in result.items():
        print(f"{category}: {variables}")

    assert result == {
        "x_exog": ["x", "x1"],
        "y_endog": ["y", "y1"],
        "always_include": ["c"],
        "controls": ["c2", "c3"],
    }, "Parser output doesn't match expected result"


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_012_formula_run(mock_show) -> None:
    df = specy.load_example_data1()
    sc = specy.SpecificationCurve(df=df, formula="y1 ~ x1 | x2 + group1 + c1 | c2")
    sc.fit()
    sc.plot()
    mock_show.assert_called_once()


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_013_formula_run_endog(mock_show) -> None:
    df = specy.load_example_data1()
    sc = specy.SpecificationCurve(df=df, formula="y1 | y2 ~ x1 | x2 + group1 | c1 | c2")
    sc.fit()
    sc.plot()
    mock_show.assert_called_once()


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_013_without_pretty(mock_show) -> None:
    df = specy.load_example_data1()
    sc = specy.SpecificationCurve(df=df, formula="y1 | y2 ~ x1 | x2 + group1 | c1 | c2")
    sc.fit()
    sc.plot(pretty_plots=False)
    mock_show.assert_called_once()


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_014_cannot_have_formula_and_lists(mock_show) -> None:
    with raises(ValueError) as exc_info:
        df = specy.load_example_data1()
        sc = specy.SpecificationCurve(  # noqa: F841
            df=df,
            formula="y1 | y2 ~ x1 | x2 + group1 | c1 | c2",
            y_endog=["y1", "y2"],
            x_exog=["x1"],
        )
    assert exc_info.type is ValueError


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_015_must_have_formula_xor_lists(mock_show) -> None:
    with raises(ValueError) as exc_info:
        df = specy.load_example_data1()
        sc = specy.SpecificationCurve(df=df)  # noqa: F841
    assert exc_info.type is ValueError


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_016_null_under_bootstraps(mock_show) -> None:
    n_samples = 200
    x_1 = np.random.random(size=n_samples)
    x_2 = np.random.randint(3, size=n_samples)
    x_3 = np.random.random(size=n_samples)
    x_4 = x_1 + 0.05 * np.random.randn(n_samples)
    y = 3.5 * x_1 + 0.2 * x_2 + 0.3 * x_3 + 5
    df = pd.DataFrame(
        [x_1, x_2, x_3, x_4, y],
        ["x_1", "x_2", "x_3", "x_4", "y"],
    ).T
    y_endog = ["y"]
    x_exog = ["x_1", "x_4"]
    controls = ["x_2", "x_3"]
    sco = specy.SpecificationCurve(df, y_endog, x_exog, controls)
    sco.fit()
    sco.fit_null(n_boot=4)
    sco.plot(show_null_curve=True, **{"n_boot": 5})  # type: ignore
    sco.plot(show_null_curve=True)


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_017_null_under_bootstraps_auto_run(mock_show) -> None:
    df = specy.load_example_data3()
    sco = specy.SpecificationCurve(df, formula="y1 | y2 ~ x1 + c1 | c2 | c3")
    sco.fit()
    sco.plot(show_null_curve=True)
    print(sco)


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_018_repr(mock_show) -> None:
    df = specy.load_example_data3()
    sco = specy.SpecificationCurve(df, formula="y1 | y2 ~ x1 + c1 | c2 | c3")
    sco.fit()
    sco.fit_null(n_boot=3)
    print(sco)


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_018a_repr_no_fit(mock_show) -> None:
    df = specy.load_example_data3()
    sco = specy.SpecificationCurve(df, formula="y1 | y2 ~ x1 + c1 | c2 | c3")
    print(sco)


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_018b_repr_with_lists(mock_show) -> None:
    df = specy.load_example_data3()
    sco = specy.SpecificationCurve(
        df,
        y_endog=["y1", "y2"],
        x_exog="x1",
        controls=["c1", "c2", "c3"],
        exclu_grps=[["c1", "c2"], ["c3", "c2"]],
    )
    sco.fit()
    print(sco)


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_019_more_exclude(mock_show) -> None:
    df = specy.load_example_data3()
    sco = specy.SpecificationCurve(
        df,
        y_endog=["y1", "y2"],
        x_exog="x1",
        controls=["c1", "c2", "c3"],
        exclu_grps=[["c1", "c2"], ["c3", "c2"]],
    )
    sco.fit()
    sco.fit_null(n_boot=3)
    sco.plot(show_null_curve=True)
    mock_show.assert_called_once()


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_020_check_version(mock_show) -> None:
    print(specy.__version__)


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_021_exclude_and_subset(mock_show) -> None:
    df = specy.load_example_data3()
    sco = specy.SpecificationCurve(
        df,
        y_endog=["y1", "y2"],
        x_exog="x1",
        controls=["ccat", "c2", "c3"],
        exclu_grps=[["c3", "c2"]],
        cat_expand=["ccat"],
    )
    sco.fit()
    sco.fit_null(n_boot=3)
    sco.plot(show_null_curve=True)
    mock_show.assert_called_once()


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_022_formula_no_controls(mock_show) -> None:
    df = specy.load_example_data3()
    sco = specy.SpecificationCurve(
        df,
        formula="y1 | y2 ~ x1",
    )
    sco.fit()
    sco.plot()
    mock_show.assert_called_once()


@typeguard_ignore
def test_023_fit_null_without_fit() -> None:
    # Setup: Create a simple DataFrame for testing
    df = pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5],
            "x": [2, 4, 6, 8, 10],
            "control1": [1, 1, 0, 0, 1],
            "control2": [0, 1, 1, 0, 0],
        }
    )

    # Initialize SpecificationCurve
    sco = specy.SpecificationCurve(
        df=df, y_endog="y", x_exog="x", controls=["control1", "control2"]
    )

    # Test that ValueError is raised when calling fit_null() before fit()
    with raises(ValueError):
        sco.fit_null(n_boot=5, f_sample=0.1)


@typeguard_ignore
def test_024_always_include_and_cat_expand() -> None:
    # Setup: Create a simple DataFrame for testing
    df = pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5],
            "x": [2, 4, 6, 8, 10],
            "control1": [1, 1, 0, 0, 1],
            "control2": [0, 1, 1, 0, 0],
        }
    )
    # Test that ValueError is raised
    with raises(ValueError):
        sco = specy.SpecificationCurve(
            df=df,
            y_endog="y",
            x_exog="x",
            always_include=["control1"],
            cat_expand=["control1"],
            controls=["control1", "control2"],
        )
        sco.fit()


@typeguard_ignore
@patch("matplotlib.pyplot.show")
def test_025_fig_axes_further_adjustment(mock_show) -> None:
    df = specy.load_example_data3()
    sco = specy.SpecificationCurve(
        df,
        y_endog=["y1", "y2"],
        x_exog="x1",
        controls=["ccat", "c2", "c3"],
        exclu_grps=[["c3", "c2"]],
        cat_expand=["ccat"],
    )
    sco.fit()
    fig, axes = sco.plot(return_fig=True)
    fig.suptitle("Spec curve")
    plt.show()
    mock_show.assert_called_once()
