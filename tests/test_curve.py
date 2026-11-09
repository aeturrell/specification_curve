"""Tests for the SpecificationCurve class, inference, plotting and data."""

import numpy as np
import pandas as pd
import pyfixest as pf
import pytest
import specification_curve as sc


@pytest.fixture()
def curve(panel) -> sc.SpecificationCurve:
    """An unfitted curve with a few dimensions."""
    return sc.SpecificationCurve(
        panel,
        y=["log(sales)", "log(sales + 1)"],
        x="treated",
        controls=["size", "age"],
        fe=[None, "firm"],
        vcov=["hetero", {"CRV1": "firm"}],
    )


def test_grid_is_available_before_fit(curve) -> None:
    assert curve.n_specs == len(curve.grid) == 2 * 4 * 2 * 2
    assert "estimate" not in curve.grid.columns
    with pytest.raises(ValueError, match="Run .fit"):
        curve.results
    with pytest.raises(ValueError, match="Run .fit"):
        curve.plot()
    with pytest.raises(ValueError, match="Run .fit"):
        curve.fit_null(n_boot=2)
    with pytest.raises(ValueError, match="Run .fit_null"):
        curve.null_curves
    with pytest.raises(ValueError, match="Run .fit_null"):
        curve.null_summary
    with pytest.raises(ValueError, match="Run .fit_null"):
        curve.null_draws


def test_constructor_errors_surface_before_estimation(panel) -> None:
    with pytest.raises(ValueError, match="not both"):
        sc.SpecificationCurve(panel, "sales ~ treated", y="sales", x="treated")


def test_fit_returns_self_and_is_repeatable(curve) -> None:
    assert curve.fit() is curve
    first = curve.results
    second = curve.fit().results
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == curve.n_specs
    assert first["error"].isna().all()
    first["estimate"] = 0.0
    assert (curve.results["estimate"] != 0).all()


def test_fit_options(curve, panel) -> None:
    sampled = curve.fit(max_specs=5, seed=0).results
    assert len(sampled) == 5
    assert list(sampled["spec_id"]) == list(
        curve.fit(max_specs=5, seed=0).results["spec_id"]
    )
    assert curve.models == {}
    curve.fit(keep_models=True, ci_levels=(0.9,), ssc=pf.ssc(k_adj=False))
    assert len(curve.models) == curve.n_specs
    assert "ci_low_90" in curve.results.columns
    reference = pf.feols(
        "log(sales) ~ treated",
        panel.dropna(subset=["age"]),
        vcov="hetero",
        ssc=pf.ssc(k_adj=False),
    )
    np.testing.assert_allclose(curve.results["se"].iloc[0], reference.se()["treated"])
    with pytest.raises(TypeError, match="not_an_argument"):
        curve.fit(not_an_argument=1)
    with pytest.raises(TypeError, match="vcov"):
        curve.fit(vcov="hetero")


def test_pass_through_arguments_reach_only_estimators_that_accept_them(panel) -> None:
    sco = sc.SpecificationCurve(
        panel, y="log(sales)", x="treated", model=["feols", "quantreg"]
    )
    results = sco.fit(fixef_rm="none").results
    assert results["error"].isna().all()


def test_repr(curve, panel) -> None:
    text = repr(curve)
    assert "Specifications: 32" in text
    assert "y: log(sales), log(sales + 1)" in text
    assert "controls: age, size" in text
    assert "fe: firm" in text
    assert "Fitted" not in text
    curve.fit()
    assert "Fitted: 32 of 32 succeeded" in repr(curve)
    assert "median" in repr(curve)
    curve.fit_null(n_boot=2, seed=0)
    assert "Under the null (bootstrap, 2 resamples)" in repr(curve)
    assert "Stouffer Z" in repr(curve)


def test_repr_with_all_failures(panel) -> None:
    sco = sc.SpecificationCurve(
        panel, y="log(sales)", x="treated", always=["nofunc(size)"]
    )
    assert "always: nofunc(size)" in repr(sco)
    assert "Fitted: 0 of 1 succeeded" in repr(sco.fit())
    with pytest.raises(ValueError, match="Nothing to plot"):
        sco.plot()
    with pytest.raises(ValueError, match="No specification was estimated"):
        sco.fit_null(n_boot=2)


@pytest.mark.parametrize(
    "loader, shape",
    [
        (sc.load_example_data1, (500, 8)),
        (sc.load_example_data2, (500, 5)),
        (sc.load_example_data3, (20000, 7)),
        (sc.load_example_panel, (2400, 14)),
    ],
)
def test_example_data(loader, shape) -> None:
    df = loader()
    assert df.shape == shape
    pd.testing.assert_frame_equal(df, loader())


def test_example_data_runs(panel) -> None:
    df = sc.load_example_data1()
    sco = sc.SpecificationCurve(
        df,
        y=["y1", "y2"],
        x=["x1", "x2"],
        controls=["c1", "c2"],
        fe=[None, "group1", "group1 + group2"],
    )
    assert sco.fit().results["error"].isna().all()


def test_panel_recovers_true_effect() -> None:
    df = sc.load_example_panel()
    results = sc.SpecificationCurve(
        df, y="log(sales)", x="treated", always=["size", "age", "leverage"],
        fe=[None, "firm + year"], iv=[None, "treated ~ eligible"],
    ).fit().results  # fmt: skip
    naive = results[results["fe"].isna() & results["iv"].isna()].iloc[0]
    within = results[results["fe"].notna() & results["iv"].isna()].iloc[0]
    assert naive["ci_low_95"] > 0.3
    assert within["ci_low_95"] < 0.3 < within["ci_high_95"]
    for _, row in results[results["iv"].notna()].iterrows():
        assert row["ci_low_95"] < 0.3 < row["ci_high_95"]
