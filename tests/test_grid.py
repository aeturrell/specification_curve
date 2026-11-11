"""Tests for building the specification grid."""

import warnings

import pandas as pd
import pytest

import specification_curve as sc
from specification_curve.grid import (
    GRID_COLUMNS,
    build_grid,
    cluster_variables,
    row_variables,
    variables_in,
)


def test_columns_and_ids(panel) -> None:
    grid = build_grid(panel, y="sales", x="treated", controls=["size", "age"])
    assert list(grid.frame.columns) == GRID_COLUMNS
    assert list(grid.frame["spec_id"]) == [0, 1, 2, 3]
    assert list(grid.frame["controls"]) == [(), ("size",), ("age",), ("size", "age")]
    assert list(grid.frame["formula"]) == [
        "sales ~ treated",
        "sales ~ treated + size",
        "sales ~ treated + age",
        "sales ~ treated + size + age",
    ]
    assert grid.frame["fe"].tolist() == [None] * 4
    assert grid.vcov_specs == {"default": None}
    assert grid.subsets == {"all": None}


def test_dimension_counts_multiply(panel) -> None:
    grid = build_grid(
        panel,
        y=["sales", "log(sales)"],
        x=["treated", "eligible"],
        controls=["size", "age", "leverage"],
        always="i(industry)",
        fe=[None, "firm", "firm + year"],
        vcov=[None, "hetero", {"CRV1": "firm"}],
        weights=[None, "weight"],
        subsets={"all": None, "late": "year >= 2013"},
    )
    assert len(grid.frame) == 2 * 2 * 8 * 3 * 3 * 2 * 2
    assert set(grid.frame["always"]) == {("i(industry)",)}
    assert set(grid.vcov_specs) == {"default", "hetero", "CRV1: firm"}


def test_exclude_and_size_limits(panel) -> None:
    kwargs = dict(y="sales", x="treated", controls=["size", "age", "leverage"])
    excluded = build_grid(panel, exclude=[["size", "age"]], **kwargs).frame
    assert len(excluded) == 6
    assert not any({"size", "age"} <= set(c) for c in excluded["controls"])
    limited = build_grid(panel, min_controls=1, max_controls=2, **kwargs).frame
    assert sorted({len(c) for c in limited["controls"]}) == [1, 2]
    assert len(limited) == 6


def test_exclude_matches_whole_terms_only(panel) -> None:
    df = panel.assign(size2=panel["size"] ** 2)
    grid = build_grid(
        df,
        y="sales",
        x="treated",
        controls=["size", "size2", "age"],
        exclude=[["size", "age"]],
    )
    assert ("size2", "age") in set(grid.frame["controls"])


def test_overlapping_terms_are_removed(panel) -> None:
    grid = build_grid(
        panel,
        y="sales",
        x=["treated", "size"],
        controls=["size", "age"],
        always=["treated"],
    ).frame
    for row in grid.itertuples():
        assert row.x not in row.controls
        assert row.x not in row.always
    assert len(grid) == 4 + 2
    assert not grid.duplicated(["y", "x", "controls", "always"]).any()


def test_formula_and_keywords_agree(panel) -> None:
    from_keywords = build_grid(
        panel,
        y=["sales", "log(sales)"],
        x="treated",
        controls=["age", "leverage"],
        always=["size"],
        fe=[None, "firm", "firm + year"],
    ).frame
    from_formula = build_grid(
        panel,
        "sales + log(sales) ~ treated + size + comb0(age, leverage) | sw0(firm, firm + year)",
        x="treated",
    ).frame
    key = ["y", "x", "controls", "always", "fe"]
    assert len(from_formula) == len(from_keywords) == 24
    assert set(map(tuple, from_formula[key].values)) == set(
        map(tuple, from_keywords[key].values)
    )


def test_formula_respects_exclude_and_size_limits(panel) -> None:
    formula = "sales ~ treated + comb0(size, age, leverage)"
    limited = build_grid(
        panel, formula, x="treated", min_controls=1, max_controls=2
    ).frame
    assert sorted({len(c) for c in limited["controls"]}) == [1, 2]
    excluded = build_grid(panel, formula, x="treated", exclude=[["size", "age"]]).frame
    assert len(excluded) == 6


def test_formula_with_alternative_focal_regressors(panel) -> None:
    grid = build_grid(
        panel, "sales ~ sw(treated, eligible) + size", x=["treated", "eligible"]
    )
    assert list(grid.frame["x"]) == ["treated", "eligible"]
    assert set(grid.frame["always"]) == {("size",)}


def test_formula_with_instruments(panel) -> None:
    grid = build_grid(
        panel, "sales ~ sw0(size) | firm | treated ~ eligible", x="treated"
    )
    assert set(grid.frame["iv"]) == {"treated ~ eligible"}
    assert list(grid.frame["formula"]) == [
        "sales ~ 1 | firm | treated ~ eligible",
        "sales ~ size | firm | treated ~ eligible",
    ]


def test_model_variants(panel) -> None:
    grid = build_grid(
        panel,
        y="patents",
        x="treated",
        model=["feols", "fepois", "feglm", "quantreg"],
        family=["poisson", "gaussian"],
        quantile=[0.25, 0.5],
    ).frame
    got = set(zip(grid["model"], grid["family"], grid["quantile"]))
    assert got == {
        ("feols", None, None),
        ("fepois", None, None),
        ("feglm", "poisson", None),
        ("feglm", "gaussian", None),
        ("quantreg", None, 0.25),
        ("quantreg", None, 0.5),
    }
    default_q = build_grid(panel, y="sales", x="treated", model="quantreg").frame
    assert default_q["quantile"].tolist() == [0.5]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(y="sales"), "x .* is required"),
        (dict(x="treated"), "either a formula or y"),
        (dict(formula="sales ~ treated", y="sales", x="treated"), "not both"),
        (dict(formula="sales ~ size", x="treated"), "exactly one focal"),
        (
            dict(formula="sales ~ treated + size", x=["treated", "size"]),
            "exactly one focal",
        ),
        (dict(formula="sales ~ treated - 1", x="treated"), "drop_intercept"),
        (dict(y="sales", x="treated", model="ols"), "Unknown model"),
        (dict(y="sales", x="treated", model="feglm"), "needs a family"),
        (dict(y="sales", x="treated", family="logit"), "only used by"),
        (dict(y="sales", x="treated", quantile=0.5), "only used by"),
        (
            dict(y="sales", x="treated", controls=["nope"]),
            "does not refer to any column",
        ),
        (dict(y="sales", x="treated", fe="firm + nope"), "not in the data"),
        (dict(y="sales", x="treated", weights="nope"), "not in the data"),
        (dict(y="sales", x="treated", vcov={"CRV1": "nope"}), "not in the data"),
        (
            dict(y="sales", x="treated", iv="treated ~ nope"),
            "does not refer to any column",
        ),
        (
            dict(y="sales", x="treated", subsets={"none": "year > 3000"}),
            "selects no rows",
        ),
        (dict(y="sales", x="treated", subsets={"bad": "nope > 1"}), "failed"),
        (dict(y="sales", x="treated", subsets={}), "at least one"),
        (
            dict(y="sales", x="treated", controls=["size"], min_controls=2),
            "No specifications",
        ),
        (dict(y="sales", x="treated", model="quantreg", fe="firm"), "fixed effects"),
        (dict(y="sales", x="treated", model="quantreg", weights="weight"), "weights"),
        (
            dict(y="sales", x="treated", model="quantreg", iv="treated ~ eligible"),
            "instrumental",
        ),
        (
            dict(y="patents", x="treated", model="fepois", iv="treated ~ eligible"),
            "instrumental",
        ),
    ],
)
def test_validation_errors(panel, kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_grid(panel, **kwargs)


def test_skip_invalid(panel) -> None:
    kwargs = dict(
        y="sales", x="treated", model=["feols", "quantreg"], fe=[None, "firm"]
    )
    with pytest.raises(ValueError, match="skip_invalid"):
        build_grid(panel, **kwargs)
    with pytest.warns(UserWarning, match="Left out 1"):
        grid = build_grid(panel, skip_invalid=True, **kwargs)
    assert grid.n_skipped == 1
    assert len(grid.frame) == 3
    with pytest.raises(ValueError, match="No valid specifications"):
        build_grid(
            panel,
            y="sales",
            x="treated",
            model="quantreg",
            fe="firm",
            skip_invalid=True,
        )


def test_large_grid_warns(panel, monkeypatch) -> None:
    monkeypatch.setattr("specification_curve.grid.LARGE_GRID", 3)
    with pytest.warns(UserWarning, match="max_specs"):
        build_grid(panel, y="sales", x="treated", controls=["size", "age"])


def test_sample_is_reproducible(panel) -> None:
    grid = build_grid(
        panel, y="sales", x="treated", controls=["size", "age", "leverage"]
    )
    first = grid.sample(3, seed=1).frame
    assert len(first) == 3
    assert first["spec_id"].is_monotonic_increasing
    assert first.equals(grid.sample(3, seed=1).frame)
    assert not first.equals(grid.sample(3, seed=2).frame)
    assert grid.sample(100) is grid


def test_variables_helpers() -> None:
    columns = ["a", "b c", "firm", "year", "w"]
    assert variables_in("log(a) + `b c` + a", columns) == ["a", "b c"]
    assert variables_in("np.log(z)", columns) == []
    assert cluster_variables({"CRV1": "firm + year"}, columns) == ["firm", "year"]
    assert cluster_variables("hetero", columns) == []
    row = {"formula": "a ~ `b c` | firm", "weights": "w"}
    assert row_variables(row, columns, {"CRV1": "year"}) == [
        "a",
        "b c",
        "firm",
        "w",
        "year",
    ]


def test_subsets_by(panel) -> None:
    subsets = sc.subsets_by(panel, "industry", include_all=True)
    assert list(subsets)[0] == "all"
    assert len(subsets) == panel["industry"].nunique() + 1
    for name, query in subsets.items():
        if query is not None:
            assert (
                len(panel.query(query))
                == (panel["industry"] == int(name.split(" = ")[1])).sum()
            )
    text = sc.subsets_by(pd.DataFrame({"g": ["a", "b"]}), "g")
    assert text == {"g = a": "`g` == 'a'", "g = b": "`g` == 'b'"}


def test_no_warning_for_ordinary_grid(panel) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_grid(panel, y="sales", x="treated", controls=["size"])
