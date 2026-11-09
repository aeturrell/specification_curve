"""Pytest configuration file for specification_curve tests."""

import warnings

import matplotlib
import pandas as pd
import pytest

# Set matplotlib to use non-interactive backend to avoid tkinter issues on CI
# This must be done before importing pyplot
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    """A small firm-year panel."""
    # imported here so that typeguard can instrument the package first
    import specification_curve as sc

    return sc.load_example_panel(n_firms=60, n_years=6)


@pytest.fixture(autouse=True)
def _quiet_and_clean():
    """Silences estimator warnings and closes figures after each test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
    plt.close("all")
