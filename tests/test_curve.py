"""Tests for the SpecificationCurve class, inference, plotting and data."""

import matplotlib as mpl
import numpy as np
import pandas as pd
import pyfixest as pf
import pytest
import specification_curve as sc
from matplotlib.figure import Figure


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
    assert "Under the null" in repr(curve)
    failing = sc.SpecificationCurve(
        panel, y="log(sales)", x="treated", controls=["nofunc(size)"]
    )
    failing.fit(max_specs=1, seed=1)
    if failing.results["error"].notna().all():
        assert "Fitted: 0 of 1 succeeded" in repr(failing)


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


def test_fit_null(panel) -> None:
    sco = sc.SpecificationCurve(
        panel,
        y=["log(sales)", "log(sales + 1)"],
        x="treated",
        controls=["size"],
        fe=[None, "firm"],
        vcov=[None, {"CRV1": "firm"}],
    ).fit()
    assert sco.fit_null(n_boot=6, seed=1) is sco
    curves, summary = sco.null_curves, sco.null_summary
    assert curves.shape == (sco.n_specs, 3)
    assert list(curves.columns) == [0.025, 0.5, 0.975]
    assert (curves[0.025] <= curves[0.5]).all() and (curves[0.5] <= curves[0.975]).all()
    assert curves[0.5].is_monotonic_increasing
    assert list(summary.columns) == ["observed", "p-value"]
    assert summary.loc["median estimate", "observed"] == pytest.approx(
        sco.results["estimate"].median()
    )
    assert summary["p-value"].between(1 / 7, 1).all()
    # the effect is strong, so the observed median is beyond every resample
    assert summary.loc["median estimate", "p-value"] == pytest.approx(1 / 7)
    again = sco.fit_null(n_boot=6, seed=1).null_curves
    pd.testing.assert_frame_equal(curves, again)
    # refitting discards inference that belonged to the previous fit
    sco.fit()
    with pytest.raises(ValueError, match="Run .fit_null"):
        sco.null_summary


def test_null_outcome_has_no_effect_left(panel) -> None:
    from specification_curve.inference import _null_grid

    sco = sc.SpecificationCurve(
        panel, y="log(sales)", x="treated", controls=["size"], fe=[None, "firm"]
    ).fit()
    grid, augmented = _null_grid(sco.results, panel)
    assert grid["y"].nunique() == 4
    for formula in grid["formula"]:
        refit = pf.feols(formula, augmented)
        assert abs(refit.coef()["treated"]) < 1e-10


def test_fit_null_needs_feols(panel) -> None:
    sco = sc.SpecificationCurve(panel, y="patents", x="treated", model="fepois").fit()
    with pytest.raises(ValueError, match="feols"):
        sco.fit_null(n_boot=2)


def test_plot_structure(curve) -> None:
    curve.fit()
    before = dict(mpl.rcParams)
    fig, axes = curve.plot()
    assert dict(mpl.rcParams) == before
    assert isinstance(fig, Figure)
    # top panel plus y, controls, fixed effects and standard errors
    assert len(axes) == 5
    assert [ax.get_ylabel() for ax in axes] == [
        "Coefficient", "y", "controls", "fixed effects", "std. errors",
    ]  # fmt: skip
    points = axes[0].collections[-1].get_offsets()
    assert len(points) == curve.n_specs
    assert np.all(np.diff(points[:, 1]) >= 0)
    assert [t.get_text() for t in axes[2].get_yticklabels()] == ["age", "size"]


def test_plot_colours_follow_sign_and_significance(panel) -> None:
    sco = sc.SpecificationCurve(
        panel, y="log(sales)", x=["treated", "eligible"], fe="firm"
    ).fit()
    results = sco.results.sort_values("estimate")
    fig, axes = sc.plot_curve(sco.results, alpha=0.01, style=False)
    colours = axes[0].collections[-1].get_facecolors()
    for (_, row), colour in zip(results.iterrows(), colours):
        expected = "black"
        if row["pvalue"] < 0.01:
            expected = "blue" if row["estimate"] > 0 else "red"
        assert np.allclose(colour[:3], mpl.colors.to_rgb(expected))
    flipped = sco.results.assign(estimate=-sco.results["estimate"])
    _, flipped_axes = sc.plot_curve(flipped)
    reds = [
        np.allclose(c[:3], (1, 0, 0))
        for c in flipped_axes[0].collections[-1].get_facecolors()
    ]
    assert any(reds)


def test_plot_null_curve_save_and_failures(curve, panel, tmp_path) -> None:
    curve.fit(ci_levels=(0.9, 0.95)).fit_null(n_boot=3, seed=0)
    target = tmp_path / "curve.png"
    fig, axes = curve.plot(target, show_null_curve=True)
    assert target.stat().st_size > 0
    assert any("null" in t.get_text() for t in axes[0].texts)
    results = curve.results
    results.loc[0, "error"] = "boom"
    fig, axes = sc.plot_curve(results)
    assert "1 specification(s) failed" in axes[0].get_title(loc="right")
    assert len(axes[0].collections[-1].get_offsets()) == curve.n_specs - 1


def test_plot_all_dimensions_and_dense_style(panel) -> None:
    sco = sc.SpecificationCurve(
        panel,
        y="patents",
        x=["treated", "eligible"],
        controls=["size", "age", "leverage", "log(sales)"],
        model=["feols", "fepois", "feglm", "quantreg"],
        family="poisson",
        quantile=[0.75],
        weights=[None, "weight"],
        iv=[None, "treated ~ eligible"],
        subsets={"early": "year < 2013", "late": "year >= 2013", "all": None},
        skip_invalid=True,
    )
    fig, axes = sco.fit().plot()
    labels = [ax.get_ylabel() for ax in axes[1:]]
    # the default standard errors differ by estimator, so they get a panel too
    assert labels == [
        "x", "controls", "instruments", "std. errors", "model", "weights", "sample",
    ]  # fmt: skip
    assert sco.n_specs > 160
    model_ticks = [t.get_text() for t in axes[5].get_yticklabels()]
    assert set(model_ticks) == {
        "feols",
        "fepois",
        "feglm (poisson)",
        "quantreg (q = 0.75)",
    }


def test_single_specification_plot(panel) -> None:
    fig, axes = sc.SpecificationCurve(panel, y="log(sales)", x="treated").fit().plot()
    assert len(axes) == 1


@pytest.mark.parametrize(
    "loader, shape",
    [
        (sc.load_example_data1, (500, 8)),
        (sc.load_example_data2, (500, 5)),
        (sc.load_example_data3, (20000, 7)),
        (sc.load_example_panel, (2400, 13)),
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
