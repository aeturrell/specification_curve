"""Tests for the estimation engine."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pyfixest as pf
import pytest

import specification_curve as sc
from specification_curve import _pyfixest as backend
from specification_curve.engine import result_columns, run_grid
from specification_curve.grid import build_grid, row_variables

NUMERIC = ["estimate", "se", "statistic", "pvalue", "ci_low_95", "ci_high_95"]


def _run(panel: pd.DataFrame, grid_kwargs: dict, **run_kwargs: Any) -> pd.DataFrame:
    """Builds a grid and runs it."""
    grid = build_grid(panel, **grid_kwargs)
    results, _ = run_grid(
        grid.frame,
        panel,
        vcov_specs=grid.vcov_specs,
        subsets=grid.subsets,
        **run_kwargs,
    )
    return results


def _direct(
    panel: pd.DataFrame, grid_kwargs: dict, common_sample: bool = True
) -> pd.DataFrame:
    """Fits every row of a grid with its own plain pyfixest call."""
    grid = build_grid(panel, **grid_kwargs)
    columns = list(panel.columns)
    estimators = {
        "feols": pf.feols,
        "fepois": pf.fepois,
        "feglm": pf.feglm,
        "quantreg": pf.quantreg,
    }
    out = []
    for subset, rows in grid.frame.groupby("subset", sort=False):
        query = grid.subsets[subset]
        sample = panel if query is None else panel.query(query)
        if common_sample:
            used: set[str] = set()
            for row in rows.to_dict("records"):
                used.update(row_variables(row, columns, grid.vcov_specs[row["vcov"]]))
            sample = sample.dropna(subset=sorted(used))
        for row in rows.to_dict("records"):
            kwargs: dict[str, Any] = {}
            spec = grid.vcov_specs[row["vcov"]]
            if spec is not None:
                kwargs["vcov"] = spec
            if row["weights"] is not None:
                kwargs["weights"] = row["weights"]
            if row["model"] == "feglm":
                kwargs["family"] = row["family"]
            if row["model"] == "quantreg":
                kwargs["quantile"] = row["quantile"]
            fit = estimators[row["model"]](row["formula"], sample, **kwargs)
            table = fit.tidy().loc[row["x"]]
            out.append(
                {
                    "spec_id": row["spec_id"],
                    "estimate": table["Estimate"],
                    "se": table["Std. Error"],
                    "statistic": table.iloc[2],
                    "pvalue": table.iloc[3],
                    "ci_low_95": table["2.5%"],
                    "ci_high_95": table["97.5%"],
                    "nobs": fit._N,
                }
            )
    return pd.DataFrame(out).sort_values("spec_id", ignore_index=True)


def _assert_same(results: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Checks that results agree with reference numbers."""
    assert results["error"].isna().all(), results["error"].dropna().tolist()
    assert list(results["spec_id"]) == list(expected["spec_id"])
    np.testing.assert_allclose(
        results[NUMERIC].to_numpy(dtype=float),
        expected[NUMERIC].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-8,
    )
    assert list(results["nobs"]) == list(expected["nobs"])


CASES: dict[str, dict] = {
    "feols": dict(
        y=["log(sales)", "sales"],
        x="treated",
        controls=["size", "age", "leverage"],
        always=["i(industry)"],
        exclude=[["age", "leverage"]],
        fe=[None, "firm", "firm + year"],
        vcov=[None, "hetero", {"CRV1": "firm"}, {"CRV1": "firm + year"}],
    ),
    "feols_weights_subsets": dict(
        y="log(sales)",
        x=["treated", "eligible"],
        controls=["size", "log(age)"],
        fe=[None, "firm"],
        weights=[None, "weight"],
        subsets={"all": None, "late": "year >= 2013"},
        vcov="HC1",
    ),
    "feols_iv": dict(
        y="log(sales)",
        x="treated",
        controls=["size", "age"],
        fe=[None, "year"],
        iv=[None, "treated ~ eligible"],
        vcov=["hetero", {"CRV1": "firm"}],
    ),
    "fepois": dict(
        y="patents",
        x="treated",
        controls=["size", "leverage"],
        fe=[None, "year"],
        model="fepois",
        vcov=[None, "hetero", {"CRV1": "firm"}],
    ),
    "feglm": dict(
        y="exporter",
        x="treated",
        controls=["size", "leverage"],
        fe=[None, "year"],
        model="feglm",
        family=["logit", "probit", "gaussian"],
        vcov=["iid", "hetero"],
    ),
    "quantreg": dict(
        y="log(sales)",
        x="treated",
        controls=["size", "age"],
        model="quantreg",
        quantile=[0.25, 0.5],
        vcov=[None, "iid", {"CRV1": "firm"}],
    ),
    "mixed_models": dict(
        y=["patents", "exporter"],
        x="treated",
        controls=["size"],
        model=["feols", "fepois", "feglm"],
        family="gaussian",
        vcov="hetero",
    ),
    "formula": dict(
        formula="log(sales) + sales ~ treated + size + comb0(age, leverage) | sw0(firm, firm + year)",
        x="treated",
        vcov=["hetero", {"CRV1": "firm"}],
    ),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("batch", [True, False])
def test_results_match_direct_pyfixest_calls(panel, case: str, batch: bool) -> None:
    results = _run(panel, CASES[case], batch=batch)
    _assert_same(results, _direct(panel, CASES[case]))


def test_without_common_sample_each_model_uses_its_own_rows(panel) -> None:
    kwargs = dict(
        y="log(sales)",
        x="treated",
        controls=["age", "leverage"],
        vcov=[None, {"CRV1": "firm"}],
    )
    results = _run(panel, kwargs, common_sample=False)
    _assert_same(results, _direct(panel, kwargs, common_sample=False))
    assert results["nobs"].nunique() > 1
    common = _run(panel, kwargs)
    assert common["nobs"].nunique() == 1
    assert common["nobs"].iloc[0] == len(panel.dropna(subset=["age", "leverage"]))


@pytest.mark.parametrize("common_sample", [True, False])
def test_missing_cluster_values_are_dropped(panel, common_sample: bool) -> None:
    df = panel.copy()
    df["state"] = df["state"].astype(float)
    df.loc[df.index[:25], "state"] = np.nan
    kwargs = dict(
        y="log(sales)",
        x="treated",
        controls=["size"],
        vcov=["hetero", {"CRV1": "state"}],
    )
    results = _run(df, kwargs, common_sample=common_sample)
    assert results["error"].isna().all()
    clustered = results[results["vcov"] == "CRV1: state"]
    robust = results[results["vcov"] == "hetero"]
    assert (clustered["nobs"] == len(df) - 25).all()
    assert (robust["nobs"] == (len(df) - 25 if common_sample else len(df))).all()
    reference = pf.feols(
        "log(sales) ~ treated + size",
        df.dropna(subset=["state"]),
        vcov={"CRV1": "state"},
    )
    np.testing.assert_allclose(clustered["se"].iloc[1], reference.se()["treated"])


def test_vcov_variants_share_one_fit(panel, monkeypatch) -> None:
    calls = []
    original = backend.estimate

    def counting(formula: str, data: pd.DataFrame, **kwargs: Any) -> list[Any]:
        calls.append(formula)
        return original(formula, data, **kwargs)

    monkeypatch.setattr(backend, "estimate", counting)
    kwargs = dict(
        y="log(sales)",
        x="treated",
        controls=["size", "age"],
        vcov=[None, "hetero", {"CRV1": "firm"}],
    )
    results = _run(panel, kwargs, batch=False)
    assert len(results) == 12
    assert len(calls) == 4
    calls.clear()
    _run(panel, kwargs, batch=True)
    assert len(calls) == 1
    assert "sw0(" in calls[0]


def test_default_vcov_is_labelled_from_the_model(panel) -> None:
    results = _run(
        panel, dict(y="log(sales)", x="treated", model=["feols", "quantreg"])
    )
    assert dict(zip(results["model"], results["vcov"])) == {
        "feols": "iid",
        "quantreg": "nid",
    }


def test_batches_are_split_and_sampled_grids_work(panel, monkeypatch) -> None:
    monkeypatch.setattr("specification_curve.engine.BATCH_SIZE", 3)
    kwargs = dict(
        y=["log(sales)", "sales"], x="treated", controls=["size", "age", "leverage"]
    )
    grid = build_grid(panel, **kwargs)
    expected = _direct(panel, kwargs)
    full, _ = run_grid(
        grid.frame, panel, vcov_specs=grid.vcov_specs, subsets=grid.subsets
    )
    _assert_same(full, expected)
    sampled = grid.sample(7, seed=3)
    part, _ = run_grid(
        sampled.frame, panel, vcov_specs=grid.vcov_specs, subsets=grid.subsets
    )
    # The complete-case sample depends on which specifications are in the grid, so
    # compare with the unbatched engine on the same sampled grid.
    single, _ = run_grid(
        sampled.frame,
        panel,
        vcov_specs=grid.vcov_specs,
        subsets=grid.subsets,
        batch=False,
    )
    assert list(part["spec_id"]) == list(sampled.frame["spec_id"])
    _assert_same(part, single)


def test_batched_call_that_fails_falls_back_to_single_fits(panel, monkeypatch) -> None:
    original = backend.estimate

    def flaky(formula: str, data: pd.DataFrame, **kwargs: Any) -> list[Any]:
        if "sw" in formula:
            raise RuntimeError("no batching today")
        return original(formula, data, **kwargs)

    monkeypatch.setattr(backend, "estimate", flaky)
    kwargs = dict(y="log(sales)", x="treated", controls=["size", "age"])
    _assert_same(_run(panel, kwargs), _direct(panel, kwargs))


def test_batched_call_with_unexpected_models_falls_back(panel, monkeypatch) -> None:
    original = backend.estimate

    def short(formula: str, data: pd.DataFrame, **kwargs: Any) -> list[Any]:
        models = original(formula, data, **kwargs)
        return models[:-1] if len(models) > 1 else models

    monkeypatch.setattr(backend, "estimate", short)
    kwargs = dict(y="log(sales)", x="treated", controls=["size", "age"])
    _assert_same(_run(panel, kwargs), _direct(panel, kwargs))


def test_errors_are_recorded_or_raised(panel) -> None:
    # industry is constant within firm, so firm fixed effects absorb it
    kwargs = dict(y="log(sales)", x="industry", controls=["size"], fe=[None, "firm"])
    results = _run(panel, kwargs)
    failed = results[results["error"].notna()]
    assert list(failed["fe"]) == ["firm", "firm"]
    assert failed["error"].str.contains("no coefficient named 'industry'").any()
    assert failed["estimate"].isna().all() and failed["nobs"].isna().all()
    assert results["error"].isna().sum() == 2
    with pytest.raises(RuntimeError, match="Specification 2"):
        _run(panel, kwargs, on_error="raise")
    with pytest.raises(ValueError, match="on_error"):
        _run(panel, kwargs, on_error="ignore")


def test_estimation_errors_are_recorded_or_raised(panel) -> None:
    kwargs = dict(y="log(sales)", x="treated", controls=["nofunc(size)"])
    results = _run(panel, kwargs)
    assert results["error"].isna().tolist() == [True, False]
    with pytest.raises(RuntimeError, match="Specification 1"):
        _run(panel, kwargs, on_error="raise")


def test_unsupported_vcov_fails_only_its_own_rows(panel) -> None:
    kwargs = dict(
        y="exporter",
        x="treated",
        model="feglm",
        family="logit",
        vcov=["hetero", {"CRV3": "firm"}],
    )
    results = _run(panel, kwargs)
    assert results["error"].isna().tolist() == [True, False]
    assert results["vcov"].tolist() == ["hetero", "CRV3: firm"]
    with pytest.raises(RuntimeError, match="vcov=CRV3: firm"):
        _run(panel, kwargs, on_error="raise")


def test_empty_sample_is_an_error_row(panel) -> None:
    df = panel.assign(empty=np.nan)
    results = _run(df, dict(y="log(sales)", x="treated", controls=["empty"]))
    assert results["error"].str.contains("no rows").all()


def test_parallel_matches_serial(panel) -> None:
    serial = _run(panel, CASES["feols_weights_subsets"])
    for batch in (True, False):
        parallel = _run(panel, CASES["feols_weights_subsets"], n_jobs=2, batch=batch)
        pd.testing.assert_frame_equal(serial, parallel, check_exact=False, rtol=1e-9)


def test_data_is_not_modified(panel) -> None:
    before = panel.copy()
    _run(panel, CASES["formula"])
    pd.testing.assert_frame_equal(panel, before)


def test_ci_levels_and_result_columns(panel) -> None:
    results = _run(panel, dict(y="log(sales)", x="treated"), ci_levels=(0.9, 0.995))
    assert result_columns((0.9, 0.995)) == [
        "estimate", "se", "statistic", "pvalue",
        "ci_low_90", "ci_high_90", "ci_low_99.5", "ci_high_99.5", "nobs", "error",
    ]  # fmt: skip
    assert list(results.columns[-10:]) == result_columns((0.9, 0.995))
    row = results.iloc[0]
    assert row["ci_low_99.5"] < row["ci_low_90"] < row["estimate"] < row["ci_high_90"]


def test_keep_models_and_pass_through_arguments(panel) -> None:
    grid = build_grid(
        panel, y="log(sales)", x="treated", controls=["size"], vcov=["iid", "hetero"]
    )
    results, models = run_grid(
        grid.frame,
        panel,
        vcov_specs=grid.vcov_specs,
        subsets=grid.subsets,
        keep_models=True,
        fit_kwargs={"drop_intercept": True},
    )
    assert sorted(models) == [0, 1, 2, 3]
    assert models[0] is models[1] and models[0] is not models[2]
    assert "Intercept" not in models[0].coef().index
    reference = pf.feols("log(sales) ~ treated + size", panel, drop_intercept=True)
    np.testing.assert_allclose(results["estimate"].iloc[2], reference.coef()["treated"])


def test_progress_bar_can_be_forced(panel, capsys) -> None:
    _run(panel, dict(y="log(sales)", x="treated"), progress=True)
    assert "Fitting specifications" in capsys.readouterr().err


def test_transformed_focal_regressor_is_matched_by_name(panel) -> None:
    results = _run(panel, dict(y="log(sales)", x="log( size+10 )", controls=["age"]))
    assert results["error"].isna().all()
    reference = pf.feols("log(sales) ~ log(size + 10)", panel.dropna(subset=["age"]))
    np.testing.assert_allclose(results["estimate"].iloc[0], reference.coef().iloc[1])


def test_package_exports() -> None:
    assert set(sc.__all__) <= set(dir(sc))
    assert isinstance(sc.__version__, str)


def test_estimation_warnings_are_summarised_once(panel) -> None:
    # industry is constant within firm, so pyfixest warns that it drops it
    kwargs = dict(
        y="log(sales)", x="treated", controls=["size", "i(industry)"], fe="firm"
    )
    with pytest.warns(UserWarning, match=r"2 warning\(s\) were raised") as caught:
        results = _run(panel, kwargs)
    assert len(caught) == 1
    assert "multicollinearity" in str(caught[0].message)
    assert results["error"].isna().all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _run(panel, kwargs, warn=False)
