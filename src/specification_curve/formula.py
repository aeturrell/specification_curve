"""Formula helpers: the ``comb``/``comb0`` operators and formula rendering."""

import re
from itertools import combinations
from typing import Optional, Sequence

from ._pyfixest import ParsedFormula, normalise, parse_formula

_COMB_PATTERN = re.compile(r"(?<![A-Za-z0-9_.])(comb0|comb)\s*\(")


def _matching_paren(text: str, open_index: int) -> int:
    """Index of the parenthesis that closes the one at ``open_index``."""
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"Unbalanced parentheses in formula {text!r}")


def split_top_level(text: str, separator: str) -> list[str]:
    """Splits text on a one-character separator, ignoring any inside parentheses.

    Args:
        text (str): Text to split.
        separator (str): Single character to split on.

    Returns:
        list[str]: The pieces, with surrounding whitespace removed.
    """
    pieces: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == separator and depth == 0:
            pieces.append(text[start:index].strip())
            start = index + 1
    pieces.append(text[start:].strip())
    return pieces


def _check_comb_placement(formula: str) -> None:
    """Raises if comb()/comb0() is used in the outcome or first-stage part."""
    parts = split_top_level(formula, "|")
    outcome = split_top_level(parts[0], "~")[0]
    first_stages = [part for part in parts[1:] if "~" in part]
    for part in [outcome, *first_stages]:
        if _COMB_PATTERN.search(part):
            raise ValueError(
                "comb() and comb0() can only be used among the right-hand-side terms "
                f"or the fixed effects, not in {part!r}"
            )


def expand_comb(formula: str) -> str:
    """Rewrites ``comb()`` and ``comb0()`` into pyfixest's ``sw()`` and ``sw0()``.

    ``comb(a, b)`` stands for every non-empty subset of its arguments, and
    ``comb0(a, b)`` also includes the empty subset.

    Args:
        formula (str): A formula that may contain ``comb``/``comb0``.

    Returns:
        str: The formula with each ``comb``/``comb0`` replaced by an equivalent
            ``sw``/``sw0`` over the enumerated subsets.

    Raises:
        ValueError: If a ``comb`` is nested inside another, has no arguments, or
            appears in the outcome or first-stage part of the formula.

    Examples:
        >>> expand_comb("y ~ x + comb0(a, b)")
        'y ~ x + sw0(a, b, a + b)'
    """
    _check_comb_placement(formula)
    out = formula
    position = 0
    while True:
        match = _COMB_PATTERN.search(out, position)
        if match is None:
            return out
        open_index = match.end() - 1
        close_index = _matching_paren(out, open_index)
        inner = out[open_index + 1 : close_index]
        if _COMB_PATTERN.search(inner):
            raise ValueError("comb() and comb0() cannot be nested")
        args = [arg for arg in split_top_level(inner, ",") if arg]
        if not args:
            raise ValueError(f"{match.group(1)}() needs at least one term")
        subsets = [
            " + ".join(subset)
            for size in range(1, len(args) + 1)
            for subset in combinations(args, size)
        ]
        operator = "sw0" if match.group(1) == "comb0" else "sw"
        replacement = f"{operator}({', '.join(subsets)})"
        out = out[: match.start()] + replacement + out[close_index + 1 :]
        position = match.start() + len(replacement)


def expand(formula: str) -> list[ParsedFormula]:
    """Expands a formula into the single models it describes.

    Args:
        formula (str): A pyfixest formula that may use ``comb``, ``comb0``, ``sw``,
            ``sw0``, ``csw``, ``csw0`` and several outcomes.

    Returns:
        list[ParsedFormula]: The parts of each model.
    """
    return parse_formula(expand_comb(formula))


def iv_endogenous(iv: str) -> str:
    """Endogenous side of a first-stage string such as "x ~ z"."""
    return split_top_level(iv, "~")[0]


def render(
    y: str,
    x: str,
    controls: Sequence[str] = (),
    always: Sequence[str] = (),
    fe: Optional[str] = None,
    iv: Optional[str] = None,
) -> str:
    """Builds a single-model pyfixest formula from its parts.

    Args:
        y (str): Outcome term.
        x (str): Focal regressor.
        controls (Sequence[str]): Control terms.
        always (Sequence[str]): Terms included in every specification.
        fe (Optional[str]): Fixed effects, e.g. "firm + year".
        iv (Optional[str]): First stage, e.g. "x ~ z".

    Returns:
        str: The formula.

    Examples:
        >>> render("y", "x", ["a", "b"], fe="firm")
        'y ~ x + a + b | firm'
        >>> render("y", "x", iv="x ~ z")
        'y ~ 1 | x ~ z'
    """
    rhs = [*always, *controls]
    if iv is None or normalise(iv_endogenous(iv)) != normalise(x):
        rhs = [x, *rhs]
    formula = f"{y} ~ {' + '.join(rhs) if rhs else '1'}"
    if fe is not None:
        formula += f" | {fe}"
    if iv is not None:
        formula += f" | {iv}"
    return formula
