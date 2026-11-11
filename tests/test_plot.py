"""Tests for drawing specification curves."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import specification_curve as sc


@pytest.fixture(scope="module")
def fitted(panel) -> sc.SpecificationCurve:
    """A fitted curve with several dimensions and two confidence levels."""
    return sc.SpecificationCurve(
        panel,
        y=["log(sales)", "log(sales + 1)"],
        x="treated",
        controls=["size", "age"],
        fe=[None, "firm"],
        vcov=["hetero", {"CRV1": "firm"}],
    ).fit(ci_levels=(0.9, 0.95))


def _artists(axes, gid: str) -> list:
    """Collections in the top panel with a given identifier."""
    return [c for c in axes["curve"].collections if c.get_gid() == gid]


def _points(axes) -> np.ndarray:
    """Positions of the estimate markers in the top panel."""
    return np.asarray(_artists(axes, "estimate")[0].get_offsets())


def _colours(axes) -> np.ndarray:
    """Colours of the estimate markers in the top panel."""
    return np.asarray(_artists(axes, "estimate")[0].get_facecolors())[:, :3]


def test_structure_and_global_state(fitted) -> None:
    before = dict(mpl.rcParams)
    other = plt.figure()
    fig, axes = fitted.plot()
    assert dict(mpl.rcParams) == before
    assert plt.fignum_exists(other.number)
    assert isinstance(fig, Figure)
    assert list(axes) == ["curve", "y", "controls", "fe", "vcov"]
    assert [ax.get_ylabel() for ax in axes.values()] == [
        "Coefficient", "y", "controls", "fixed effects", "std. errors",
    ]  # fmt: skip
    points = _points(axes)
    assert len(points) == fitted.n_specs
    assert np.all(np.diff(points[:, 1]) >= 0)
    assert [t.get_text() for t in axes["controls"].get_yticklabels()] == ["age", "size"]
    assert any(t.get_text() == "Median" for t in axes["curve"].texts)
    # one layer of intervals per stored confidence level
    assert len(_artists(axes, "interval")) == 2


def test_style_can_be_turned_off(fitted) -> None:
    styled, _ = sc.plot_curve(fitted.results)
    plain, axes = sc.plot_curve(fitted.results, style=False, show_median=False)
    label = axes["curve"].yaxis.get_label()
    assert label.get_fontfamily() == mpl.rcParams["font.family"]
    assert not axes["curve"].texts


def test_colours_follow_sign_and_significance(panel) -> None:
    sco = sc.SpecificationCurve(
        panel, y="log(sales)", x=["treated", "eligible"], fe="firm"
    ).fit()
    results = sco.results.sort_values("estimate")
    _, axes = sc.plot_curve(sco.results, alpha=0.01)
    for (_, row), colour in zip(results.iterrows(), _colours(axes)):
        expected = "black"
        if row["pvalue"] < 0.01:
            expected = "blue" if row["estimate"] > 0 else "red"
        assert np.allclose(colour, mpl.colors.to_rgb(expected))
    flipped = sco.results.assign(estimate=-sco.results["estimate"])
    _, flipped_axes = sc.plot_curve(flipped, colors={"negative": "orange"})
    assert any(
        np.allclose(c, mpl.colors.to_rgb("orange")) for c in _colours(flipped_axes)
    )
    with pytest.raises(ValueError, match="colors must be a dict"):
        sc.plot_curve(sco.results, colors=["red"])
    with pytest.raises(ValueError, match="colors must be a dict"):
        sc.plot_curve(sco.results, colors={"up": "red"})


def test_colour_by_dimension(fitted) -> None:
    _, axes = fitted.plot(color_by="fe")
    legend = axes["curve"].get_legend()
    assert legend.get_title().get_text() == "fixed effects"
    assert {t.get_text() for t in legend.get_texts()} == {"none", "firm"}
    assert len(np.unique(_colours(axes), axis=0)) == 2
    _, axes = fitted.plot(color_by="fe", colors={"none": "green", "firm": "purple"})
    ordered = fitted.results.sort_values("estimate")
    for fe, colour in zip(ordered["fe"], _colours(axes)):
        assert np.allclose(
            colour, mpl.colors.to_rgb("green" if fe is None else "purple")
        )
    _, axes = fitted.plot(
        color_by="model", colors=["teal"], labels={"model": "Estimator"}
    )
    assert axes["curve"].get_legend().get_title().get_text() == "Estimator"
    assert np.allclose(_colours(axes), mpl.colors.to_rgb("teal"))
    with pytest.raises(ValueError, match="no entry for"):
        fitted.plot(color_by="fe", colors={"firm": "purple"})
    with pytest.raises(ValueError, match="has 1 entries"):
        fitted.plot(color_by="fe", colors=["purple"])
    with pytest.raises(ValueError, match="color_by must be"):
        fitted.plot(color_by="controls")


def test_sort_orders(fitted) -> None:
    results = fitted.results
    _, axes = fitted.plot(sort_by="se")
    np.testing.assert_allclose(
        _points(axes)[:, 1], results.sort_values(["se", "estimate"])["estimate"]
    )
    _, axes = fitted.plot(sort_by=["fe", "y"])
    estimates = _points(axes)[:, 1]
    # grouped by fixed effects, then outcome, then sorted by estimate within groups
    keyed = results.assign(fe=results["fe"].map(str)).sort_values(
        ["fe", "y", "estimate"]
    )
    np.testing.assert_allclose(estimates, keyed["estimate"])
    with pytest.raises(ValueError, match="Cannot sort by"):
        fitted.plot(sort_by="nope")


def test_panels_and_labels(fitted) -> None:
    labels = {
        "fe": "Fixed effects",
        "none": "None",
        "size": "Firm size",
        "model": "Estimator",
    }
    _, axes = fitted.plot(panels=["fe", "controls", "model"], labels=labels)
    assert list(axes) == ["curve", "fe", "controls", "model"]
    assert axes["fe"].get_ylabel() == "Fixed effects"
    assert {t.get_text() for t in axes["fe"].get_yticklabels()} == {"None", "firm"}
    assert [t.get_text() for t in axes["controls"].get_yticklabels()] == [
        "age",
        "Firm size",
    ]
    # a panel that was asked for is shown even if it has a single value
    assert [t.get_text() for t in axes["model"].get_yticklabels()] == ["feols"]
    _, axes = fitted.plot(panels=[])
    assert list(axes) == ["curve"]
    with pytest.raises(ValueError, match="Unknown panel"):
        fitted.plot(panels=["nope"])


def test_ci_levels(fitted) -> None:
    _, axes = fitted.plot(ci_levels=[0.9])
    assert len(_artists(axes, "interval")) == 1
    segments = _artists(axes, "interval")[0].get_segments()
    ordered = fitted.results.sort_values("estimate")
    # specifications with equal estimates can be in either order
    np.testing.assert_allclose(
        sorted(s[0][1] for s in segments), sorted(ordered["ci_low_90"])
    )
    np.testing.assert_allclose(
        sorted(s[1][1] for s in segments), sorted(ordered["ci_high_90"])
    )
    _, axes = fitted.plot(ci_levels=[])
    assert not _artists(axes, "interval")
    with pytest.raises(ValueError, match="not in the results"):
        fitted.plot(ci_levels=[0.99])


@pytest.mark.parametrize(
    "selector",
    [
        3,
        {"y": "log(sales)", "controls": ["size"], "fe": None, "vcov": "CRV1: firm"},
        {"y": "log(sales)", "controls": "size", "fe": None, "vcov": "CRV1: firm"},
        "spec_id == 3",
    ],
)
def test_highlight_selectors_pick_the_same_specification(fitted, selector) -> None:
    results = fitted.results
    assert results.loc[results["spec_id"] == 3, "controls"].iloc[0] == ("size",)
    _, axes = fitted.plot(highlight=selector)
    target = results.loc[results["spec_id"] == 3, "estimate"].iloc[0]
    marker = np.asarray(_artists(axes, "highlight")[0].get_offsets())
    assert marker.shape == (1, 2)
    assert marker[0, 1] == pytest.approx(target)
    texts = [t.get_text() for t in axes["curve"].texts]
    assert f"Preferred specification: {target:+.3g}" in texts


def test_highlight_several_and_errors(fitted) -> None:
    _, axes = fitted.plot(highlight=[0, 31])
    texts = [t.get_text() for t in axes["curve"].texts]
    estimates = fitted.results.set_index("spec_id")["estimate"]
    assert f"{estimates[0]:+.3g}" in texts and f"{estimates[31]:+.3g}" in texts
    assert not any("Preferred" in text for text in texts)
    with pytest.raises(ValueError, match="matches 16"):
        fitted.plot(highlight={"fe": "firm"})
    with pytest.raises(ValueError, match="matches 0"):
        fitted.plot(highlight=999)
    with pytest.raises(ValueError, match="not in the results"):
        fitted.plot(highlight={"nope": 1})
    with pytest.raises(ValueError, match="query .* failed"):
        fitted.plot(highlight="nope > 1")


def test_null_curve_nobs_save_and_failures(fitted, panel, tmp_path) -> None:
    sco = sc.SpecificationCurve(
        panel,
        y="log(sales)",
        x="treated",
        controls=["age", "leverage"],
        common_sample=False,
    ).fit()
    sco.fit_null(n_boot=3, seed=0)
    target = tmp_path / "curve.png"
    _, axes = sco.plot(target, show_null_curve=True, show_nobs=True)
    assert target.stat().st_size > 0
    assert list(axes) == ["curve", "controls", "nobs"]
    assert any("null" in t.get_text() for t in axes["curve"].texts)
    heights = [bar.get_height() for bar in axes["nobs"].patches]
    assert heights == list(sco.results.sort_values("estimate")["nobs"])
    with pytest.raises(ValueError, match="sort_by='estimate'"):
        sco.plot(show_null_curve=True, sort_by="se")
    results = sco.results
    results.loc[0, "error"] = "boom"
    _, axes = sc.plot_curve(results)
    assert "1 specification(s) failed" in axes["curve"].get_title(loc="right")
    assert len(_points(axes)) == sco.n_specs - 1
    with pytest.raises(ValueError, match="Nothing to plot"):
        sc.plot_curve(results.assign(error="boom"))


def test_drawing_into_an_existing_figure(fitted, tmp_path) -> None:
    outer = plt.figure(figsize=(14, 6))
    left, right = outer.subfigures(1, 2)
    fig, axes = fitted.plot(tmp_path / "both.png", fig=left, panels=["fe"])
    assert fig is left
    assert all(ax.get_figure() is left for ax in axes.values())
    sc.plot_curve(fitted.results, fig=right, panels=["y"])
    assert len(outer.subfigs) == 2
    assert (tmp_path / "both.png").stat().st_size > 0
    with pytest.raises(ValueError, match="figsize or fig"):
        fitted.plot(fig=plt.figure(), figsize=(4, 4))
    sized, _ = fitted.plot(figsize=(5, 4))
    assert tuple(sized.get_size_inches()) == (5, 4)


def test_results_from_elsewhere_can_be_plotted(fitted, tmp_path) -> None:
    path = tmp_path / "results.pkl"
    fitted.results.to_pickle(path)
    combined = pd.concat([pd.read_pickle(path), fitted.results.assign(subset="again")])
    _, axes = sc.plot_curve(combined.reset_index(drop=True))
    assert "subset" in axes
    assert len(_points(axes)) == 2 * fitted.n_specs


def test_all_dimensions_and_dense_style(panel) -> None:
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
    _, axes = sco.fit().plot()
    # the default standard errors differ by estimator, so they get a panel too
    assert list(axes) == [
        "curve",
        "x",
        "controls",
        "iv",
        "vcov",
        "model",
        "weights",
        "subset",
    ]
    assert sco.n_specs > 160
    ticks = {t.get_text() for t in axes["model"].get_yticklabels()}
    assert ticks == {"feols", "fepois", "feglm (poisson)", "quantreg (q = 0.75)"}
    # with many specifications, only the choices that are made are drawn
    assert len(axes["x"].collections) == 1


def test_single_specification(panel) -> None:
    _, axes = sc.SpecificationCurve(panel, y="log(sales)", x="treated").fit().plot()
    assert list(axes) == ["curve"]


def test_returns_tuple_like_curve_plot(fitted) -> None:
    result = fitted.plot()
    assert isinstance(result, sc.CurvePlot)
    assert isinstance(result, tuple)
    fig, axes = result
    assert fig is result.fig and axes is result.axes
    # shows nothing as text in Jupyter; the backend shows the figure itself
    assert result._ipython_display_() is None
