"""Tests for inference under the null."""

import numpy as np
import pandas as pd
import pyfixest as pf
import pytest
import specification_curve as sc
from scipy.stats import norm
from specification_curve.inference import (
    _null_grid,
    _units,
    bootstrap_rows,
    curve_statistics,
    permute_rows,
    resampling_pvalues,
)


@pytest.fixture()
def fitted(panel) -> sc.SpecificationCurve:
    """A small fitted linear curve."""
    return sc.SpecificationCurve(
        panel,
        y=["log(sales)", "log(sales + 1)"],
        x="treated",
        controls=["size"],
        fe=[None, "firm"],
        vcov=[None, {"CRV1": "firm"}],
    ).fit()


def test_statistics_by_hand() -> None:
    estimates = np.array([0.5, 1.0, -0.2, 2.0])
    pvalues = np.array([0.01, 0.20, 0.03, 0.001])
    z = np.sign(estimates) * norm.isf(pvalues / 2)
    two_sided = curve_statistics(estimates, pvalues)
    np.testing.assert_allclose(two_sided, [0.75, 0.5, z.sum() / 2])
    assert curve_statistics(estimates, pvalues, alternative="greater")[1] == 0.5
    assert curve_statistics(estimates, pvalues, alternative="less")[1] == 0.25
    assert curve_statistics(estimates, pvalues, alpha=0.005)[1] == 0.25
    # the dominant sign is whichever has more significant estimates
    assert curve_statistics(-estimates, pvalues)[1] == 0.5


def test_statistics_ignore_missing_and_work_by_row() -> None:
    estimates = np.array([[0.5, np.nan, 1.5], [np.nan, np.nan, np.nan]])
    pvalues = np.array([[0.01, np.nan, 0.5], [np.nan, np.nan, np.nan]])
    stats = curve_statistics(estimates, pvalues)
    assert stats.shape == (2, 3)
    np.testing.assert_allclose(stats[0, :2], [1.0, 0.5])
    assert np.isnan(stats[1, 0]) and stats[1, 1] == 0 and stats[1, 2] == 0
    # a p-value of zero gives a large but finite z
    assert np.isfinite(curve_statistics(np.array([1.0]), np.array([0.0]))[2])


def test_resampling_pvalues() -> None:
    null = np.array([[-3.0, 0.1, -1.0], [1.0, 0.2, 0.5], [2.0, 0.6, 3.0]])
    observed = np.array([2.0, 0.5, 2.0])
    np.testing.assert_allclose(
        resampling_pvalues(observed, null), [3 / 4, 2 / 4, 2 / 4]
    )
    np.testing.assert_allclose(
        resampling_pvalues(observed, null, "greater"), [2 / 4, 2 / 4, 2 / 4]
    )
    # for "less", small medians and Z are extreme, but a large share still is
    np.testing.assert_allclose(
        resampling_pvalues(observed, null, "less"), [1, 2 / 4, 3 / 4]
    )


def test_bootstrap(fitted) -> None:
    assert fitted.fit_null(n_boot=8, seed=1) is fitted
    curves, summary = fitted.null_curves, fitted.null_summary
    assert curves.shape == (fitted.n_specs, 3)
    assert list(curves.columns) == [0.025, 0.5, 0.975]
    assert (curves[0.025] <= curves[0.5]).all() and (curves[0.5] <= curves[0.975]).all()
    assert curves[0.5].is_monotonic_increasing
    assert list(summary.index) == ["median", "share significant", "Stouffer Z"]
    assert list(summary.columns) == ["observed", "p-value"]
    results = fitted.results
    assert summary.loc["median", "observed"] == pytest.approx(
        results["estimate"].median()
    )
    assert summary.loc["share significant", "observed"] == pytest.approx(
        (results["pvalue"] < 0.05).mean()
    )
    # the effect is strong, so the observed curve is beyond every resample
    np.testing.assert_allclose(summary["p-value"], 1 / 9)
    with pytest.raises(ValueError, match="keep_draws"):
        fitted.null_draws


def test_bootstrap_is_reproducible_for_any_number_of_jobs(fitted) -> None:
    first = fitted.fit_null(n_boot=4, seed=7, keep_draws=True).null_draws
    assert first.shape == (4, fitted.n_specs)
    assert list(first.columns) == list(fitted.results["spec_id"])
    again = fitted.fit_null(n_boot=4, seed=7, keep_draws=True).null_draws
    parallel = fitted.fit_null(n_boot=4, seed=7, keep_draws=True, n_jobs=2).null_draws
    other = fitted.fit_null(n_boot=4, seed=8, keep_draws=True).null_draws
    pd.testing.assert_frame_equal(first, again)
    pd.testing.assert_frame_equal(first, parallel, check_exact=False, rtol=1e-9)
    assert not np.allclose(first, other)
    # specifications that differ only in their standard errors share estimates
    np.testing.assert_allclose(first[0], first[1])


def test_refitting_discards_inference(fitted) -> None:
    fitted.fit_null(n_boot=2, seed=0)
    fitted.fit()
    with pytest.raises(ValueError, match="Run .fit_null"):
        fitted.null_summary


def test_null_outcome_has_no_effect_left(panel) -> None:
    sco = sc.SpecificationCurve(
        panel,
        y="log(sales)",
        x=["treated", "log(size + 10)"],
        controls=["age"],
        fe=[None, "firm"],
        iv=[None, "treated ~ eligible"],
        skip_invalid=True,
    ).fit()
    ok = sco.results[sco.results["error"].isna()]
    grid, augmented = _null_grid(ok, panel)
    assert grid["y"].nunique() == len(ok)
    sample = augmented.dropna(subset=["age"])
    for row in grid.to_dict("records"):
        refit = pf.feols(row["formula"], sample)
        table = refit.tidy()
        focal = [
            name
            for name in table.index
            if name.replace(" ", "") == row["x"].replace(" ", "")
        ]
        assert abs(table.loc[focal[0], "Estimate"]) < 1e-8


def test_units_and_bootstrap_rows() -> None:
    data = pd.DataFrame(
        {
            "cluster": ["a", "a", "b", "b", "b", "c", "d", "d"],
            "stratum": [0, 0, 0, 0, 0, 1, 1, 1],
        }
    )
    rng = np.random.default_rng(0)
    rows = _units(data, None, None)
    assert len(rows) == 1 and len(rows[0]) == 8
    assert len(bootstrap_rows(rows, rng)) == 8
    clustered = _units(data, "cluster", None)
    assert [list(u) for u in clustered[0]] == [[0, 1], [2, 3, 4], [5], [6, 7]]
    for _ in range(20):
        drawn = bootstrap_rows(clustered, rng)
        counts = pd.Series(drawn).value_counts()
        # whole clusters are drawn: rows of a cluster appear equally often
        for unit in clustered[0]:
            assert counts.reindex(unit, fill_value=0).nunique() == 1
    stratified = _units(data, None, "stratum")
    assert [len(s) for s in stratified] == [5, 3]
    for _ in range(20):
        drawn = bootstrap_rows(stratified, rng)
        assert (data["stratum"].to_numpy()[drawn] == data["stratum"].to_numpy()).all()
    both = _units(data, "cluster", "stratum")
    assert [[list(u) for u in s] for s in both] == [[[0, 1], [2, 3, 4]], [[5], [6, 7]]]


def test_permute_rows() -> None:
    data = pd.DataFrame(
        {
            "cluster": ["a", "a", "b", "b", "b", "c", "d", "d"],
            "stratum": [0, 0, 0, 0, 0, 1, 1, 1],
            "x": [1, 1, 2, 2, 2, 3, 4, 4],
        }
    )
    rng = np.random.default_rng(0)
    plain = permute_rows(_units(data, None, None), 8, rng)
    assert sorted(plain) == list(range(8))
    seen = set()
    for _ in range(20):
        source = permute_rows(_units(data, "cluster", "stratum"), 8, rng)
        permuted = data["x"].to_numpy()[source]
        # clusters keep a single value, taken from a cluster in the same stratum
        assert pd.Series(permuted).groupby(data["cluster"]).nunique().eq(1).all()
        assert set(permuted[:5]) == {1, 2} and set(permuted[5:]) == {3, 4}
        seen.add(tuple(permuted))
    assert len(seen) > 1


def test_permutation_with_any_estimator(panel) -> None:
    sco = sc.SpecificationCurve(
        panel, y="patents", x="eligible", controls=["size"], model=["feols", "fepois"]
    ).fit()
    with pytest.raises(ValueError, match='method="permutation"'):
        sco.fit_null(n_boot=2)
    with pytest.raises(ValueError, match="needs every specification"):
        sco.fit_null(n_boot=2, method="bootstrap")
    sco.fit_null(n_boot=9, method="permutation", seed=3, keep_draws=True, strata="year")
    assert "permutation, 9 resamples" in repr(sco)
    # eligible is randomly assigned and has no effect on patents
    assert (sco.null_summary["p-value"] > 0.1).all()
    assert sco.null_draws.notna().all().all()


def test_permutation_detects_a_real_effect(panel) -> None:
    sco = sc.SpecificationCurve(
        panel, y="log(sales)", x="treated", controls=["size"], fe=[None, "year"]
    ).fit()
    sco.fit_null(n_boot=9, method="permutation", seed=3, alternative="greater")
    np.testing.assert_allclose(sco.null_summary["p-value"], 1 / 10)
    sco.fit_null(n_boot=9, method="permutation", seed=3, alternative="less")
    assert (sco.null_summary.loc[["median", "Stouffer Z"], "p-value"] == 1).all()


def test_permutation_by_cluster(panel) -> None:
    firm_level = panel.assign(
        assigned=panel.groupby("firm")["eligible"].transform("first")
    )
    sco = sc.SpecificationCurve(firm_level, y="log(sales)", x="assigned").fit()
    sco.fit_null(n_boot=4, method="permutation", cluster="firm", seed=0)
    assert len(sco.null_curves) == 1
    varying = sc.SpecificationCurve(panel, y="log(sales)", x="treated").fit()
    with pytest.raises(ValueError, match="constant within"):
        varying.fit_null(n_boot=2, method="permutation", cluster="firm")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(method="jackknife"), "method must be"),
        (dict(alternative="bigger"), "alternative must be"),
        (dict(n_boot=0), "at least 1"),
        (dict(cluster="nope"), "not a column"),
        (dict(strata="age"), "missing values"),
        (dict(cluster="year", strata="firm"), "within one stratum"),
    ],
)
def test_argument_errors(fitted, kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fitted.fit_null(**{"n_boot": 2, **kwargs})


def test_cluster_and_strata_bootstrap_runs(fitted) -> None:
    fitted.fit_null(n_boot=3, cluster="firm", strata="industry", seed=0, progress=True)
    assert fitted.null_summary["p-value"].between(0.25, 1).all()


def test_resamples_with_too_many_failures_are_left_out(panel, monkeypatch) -> None:
    import specification_curve.inference as inference

    sco = sc.SpecificationCurve(
        panel, y="log(sales)", x="treated", controls=["size"]
    ).fit()
    original = inference.run_grid
    calls = {"n": 0}

    def failing(grid, data, **kwargs):
        results, models = original(grid, data, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            results[["estimate", "pvalue"]] = np.nan
        return results, models

    monkeypatch.setattr(inference, "run_grid", failing)
    with pytest.warns(UserWarning, match="1 of 3 resamples were left out"):
        sco.fit_null(n_boot=3, seed=0)
    assert "2 resamples" in repr(sco)
    calls["n"] = -10
    monkeypatch.setattr(
        inference,
        "run_grid",
        lambda g, d, **k: (original(g, d, **k)[0].assign(estimate=np.nan), {}),
    )
    with pytest.raises(ValueError, match="No resample could be used"):
        sco.fit_null(n_boot=2, seed=0)


def test_no_successful_specification(panel) -> None:
    sco = sc.SpecificationCurve(
        panel, y="log(sales)", x="treated", always=["nofunc(size)"]
    )
    with pytest.raises(ValueError, match="No specification was estimated"):
        sco.fit().fit_null(n_boot=2)


@pytest.mark.slow
def test_pvalues_are_calibrated_when_there_is_no_effect() -> None:
    rng = np.random.default_rng(0)
    pvalues = {"bootstrap": [], "permutation": []}
    for _ in range(20):
        n = 150
        data = pd.DataFrame({"x": rng.normal(size=n), "a": rng.normal(size=n)})
        data["b"] = 0.5 * data["a"] + rng.normal(size=n)
        data["y"] = data["a"] + 0.5 * data["b"] + rng.normal(size=n)
        sco = sc.SpecificationCurve(data, y="y", x="x", controls=["a", "b"]).fit()
        for method in pvalues:
            seed = int(rng.integers(1_000_000))
            sco.fit_null(n_boot=19, method=method, seed=seed, progress=False)
            pvalues[method].append(sco.null_summary["p-value"].to_numpy())
    for method, values in pvalues.items():
        table = np.array(values)
        # the median and Stouffer's Z have roughly uniform p-values, averaging 0.5
        assert (np.abs(table[:, [0, 2]].mean(axis=0) - 0.5) < 0.15).all(), method
        # the share of significant estimates is usually zero in both the data and
        # the resamples, which gives a p-value of one, so it is conservative
        assert table[:, 1].mean() > 0.5, method
        # none of the three rejects much more often than its nominal rate
        assert ((table <= 0.1).mean(axis=0) < 0.3).all(), method
