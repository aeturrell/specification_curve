"""Tests for the formula helpers and the pyfixest adapter's parsing."""

import pytest

from specification_curve import _pyfixest as backend
from specification_curve.formula import (
    expand,
    expand_comb,
    render,
    split_top_level,
)


def test_comb_expands_to_sw() -> None:
    assert expand_comb("y ~ x + comb(a, b)") == "y ~ x + sw(a, b, a + b)"
    assert expand_comb("y ~ x + comb0(a, b)") == "y ~ x + sw0(a, b, a + b)"
    assert expand_comb("y ~ x + comb0(a)") == "y ~ x + sw0(a)"
    assert expand_comb("y ~ x + a") == "y ~ x + a"


def test_comb_three_terms_orders_by_size() -> None:
    out = expand_comb("y ~ x + comb(a, b, c)")
    assert out == "y ~ x + sw(a, b, c, a + b, a + c, b + c, a + b + c)"


def test_comb_keeps_nested_parentheses_and_keywords() -> None:
    out = expand_comb("y ~ x + comb0(log(a + 1), i(g, ref=1)) | f")
    assert out == "y ~ x + sw0(log(a + 1), i(g, ref=1), log(a + 1) + i(g, ref=1)) | f"


def test_comb_several_and_in_fixed_effects() -> None:
    out = expand_comb("y ~ comb(x1, x2) + comb0(a) | comb0(f, g)")
    assert out == "y ~ sw(x1, x2, x1 + x2) + sw0(a) | sw0(f, g, f + g)"


def test_comb_ignores_names_that_only_end_in_comb() -> None:
    assert expand_comb("y ~ x + mycomb(a, b)") == "y ~ x + mycomb(a, b)"


@pytest.mark.parametrize(
    "formula, message",
    [
        ("y ~ x + comb(a, comb(b, c))", "nested"),
        ("y ~ x + comb()", "at least one"),
        ("comb(y1, y2) ~ x", "can only be used"),
        ("y ~ a | f | x ~ comb(z1, z2)", "can only be used"),
        ("y ~ x + comb(a, b", "Unbalanced"),
    ],
)
def test_comb_errors(formula: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        expand_comb(formula)


def test_split_top_level() -> None:
    assert split_top_level("a + f(b + c) + d[e + g]", "+") == [
        "a",
        "f(b + c)",
        "d[e + g]",
    ]
    assert split_top_level("a", "+") == ["a"]


def test_expand_counts_models() -> None:
    parsed = expand("y1 + y2 ~ x + comb0(a, b) | sw0(f, f + g)")
    assert len(parsed) == 2 * 4 * 3
    assert {p.y for p in parsed} == {"y1", "y2"}
    assert {p.fe for p in parsed} == {None, "f", "f + g"}
    assert {p.exog for p in parsed} == {("x",), ("x", "a"), ("x", "b"), ("x", "a", "b")}


def test_parse_single_formula_parts() -> None:
    (parsed,) = backend.parse_formula("log(y) ~ a + x1:x2 | f | x ~ z1 + z2")
    assert parsed.y == "log(y)"
    assert parsed.exog == ("a", "x1:x2")
    assert parsed.fe == "f"
    assert parsed.endog == ("x",)
    assert parsed.instruments == ("z1", "z2")
    assert parsed.intercept


def test_parse_intercept_flag() -> None:
    assert backend.parse_formula("y ~ x")[0].intercept
    assert not backend.parse_formula("y ~ x - 1")[0].intercept


def test_parse_error_is_value_error() -> None:
    with pytest.raises(ValueError, match="could not parse"):
        backend.parse_formula("y ~ ~ x")


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(y="y", x="x"), "y ~ x"),
        (dict(y="y", x="x", controls=["a"], always=["b"]), "y ~ x + b + a"),
        (dict(y="y", x="x", fe="f + g"), "y ~ x | f + g"),
        (dict(y="y", x="x", iv="x ~ z"), "y ~ 1 | x ~ z"),
        (dict(y="y", x="x", controls=["a"], fe="f", iv="x ~ z"), "y ~ a | f | x ~ z"),
        (dict(y="y", x="a", iv="x ~ z"), "y ~ a | x ~ z"),
    ],
)
def test_render(kwargs: dict, expected: str) -> None:
    assert render(**kwargs) == expected


@pytest.mark.parametrize(
    "formula",
    ["y ~ x + a + log(b) | f + g", "y ~ a | f | x ~ z", "y ~ x", "y ~ 1 | x ~ z1 + z2"],
)
def test_render_parse_round_trip(formula: str) -> None:
    (parsed,) = backend.parse_formula(formula)
    focal = parsed.endog[0] if parsed.endog else parsed.exog[0]
    iv = None
    if parsed.endog:
        iv = f"{' + '.join(parsed.endog)} ~ {' + '.join(parsed.instruments)}"
    rendered = render(
        parsed.y, focal, [t for t in parsed.exog if t != focal], fe=parsed.fe, iv=iv
    )
    assert backend.parse_formula(rendered) == [parsed]


def test_vcov_label() -> None:
    assert backend.vcov_label(None) == "default"
    assert backend.vcov_label("hetero") == "hetero"
    assert backend.vcov_label({"CRV1": "firm"}) == "CRV1: firm"


def test_evaluate_term(panel) -> None:
    import numpy as np

    values = backend.evaluate_term("log(sales)", panel)
    assert np.allclose(values, np.log(panel["sales"]))
    assert values.index.equals(panel.index)
    with pytest.raises(ValueError, match="columns"):
        backend.evaluate_term("C(industry)", panel)
    with pytest.raises(ValueError, match="could not evaluate"):
        backend.evaluate_term("not_a_column", panel)
