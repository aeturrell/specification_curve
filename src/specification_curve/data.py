"""Example data sets."""

from importlib import resources

import numpy as np
import pandas as pd


def load_example_data1() -> pd.DataFrame:
    """Retrieves example data from a file included with the package.

    Returns:
        pd.DataFrame: Example data suitable for regression.
    """
    ref = resources.files("specification_curve") / "data" / "example_data.csv"
    with resources.as_file(ref) as path:
        df = pd.read_csv(path, index_col=0)
    cat_cols = ["group1", "group2"]
    for col in df.columns:
        df[col] = df[col].astype("category" if col in cat_cols else np.double)
    return df


def load_example_data2() -> pd.DataFrame:
    """Generates example data with one outcome and four regressors.

    Returns:
        pd.DataFrame: Example data suitable for regression.
    """
    prng = np.random.default_rng(78557)
    n_samples = 500
    x_1 = prng.random(size=n_samples)
    x_2 = prng.random(size=n_samples)
    x_3 = prng.random(size=n_samples)
    x_4 = prng.integers(2, size=n_samples)
    y = (
        0.8 * x_1
        + 0.1 * x_2
        + 0.5 * x_3
        + x_4 * 0.6
        + 2 * prng.standard_normal(n_samples)
    )
    return pd.DataFrame({"x_1": x_1, "x_2": x_2, "x_3": x_3, "x_4": x_4, "y": y})


def load_example_data3() -> pd.DataFrame:
    """Generates example data with two outcomes, where the true effect of x1 is 0.4.

    Returns:
        pd.DataFrame: Example data suitable for regression.
    """
    prng = np.random.default_rng(78557)
    n_samples = 20000
    c_rnd_vars = prng.random(size=(4, n_samples))
    x_bool = prng.integers(2, size=n_samples)
    y_1 = (
        0.4 * c_rnd_vars[0, :]
        - 0.2 * c_rnd_vars[1, :]
        + 0.3 * prng.standard_normal(n_samples)
        + 0.6
        + 0.2 * x_bool
    )
    # y2 is a noisier version of y1
    y_2 = y_1 - 0.5 * np.abs(prng.standard_normal(n_samples))
    return pd.DataFrame(
        {
            "y1": y_1,
            "y2": y_2,
            "x1": c_rnd_vars[0, :],
            "c1": c_rnd_vars[1, :],
            "c2": c_rnd_vars[2, :],
            "c3": c_rnd_vars[3, :],
            "ccat": x_bool,
        }
    )


def load_example_panel(n_firms: int = 300, n_years: int = 8) -> pd.DataFrame:
    """Generates a synthetic firm-year panel.

    The true effect of ``treated`` on ``log(sales)`` is 0.3. Treatment is more
    likely for firms with a high unobserved firm effect, so specifications
    without firm fixed effects overstate it. ``eligible`` is a randomly assigned
    instrument for ``treated``.

    Columns: ``firm``, ``year``, ``industry``, ``state`` (identifiers);
    ``treated`` (0/1), ``eligible`` (0/1); ``size``, ``age``, ``leverage``,
    ``rd`` (controls; ``age`` and ``leverage`` have a few missing values);
    ``sales`` (positive), ``patents`` (count), ``exporter`` (0/1) (outcomes);
    ``weight``. Note that ``age`` rises by one each year, so it cannot be
    separated from firm and year fixed effects used together.

    Args:
        n_firms (int): Number of firms.
        n_years (int): Number of years, starting in 2010.

    Returns:
        pd.DataFrame: One row per firm and year.
    """
    prng = np.random.default_rng(78557)
    firm_effect = prng.normal(size=n_firms)
    industry = prng.integers(6, size=n_firms)
    state = prng.integers(25, size=n_firms)
    start_age = prng.integers(1, 30, size=n_firms)
    weight_firm = prng.uniform(0.5, 2.0, size=n_firms)

    firm = np.repeat(np.arange(n_firms), n_years)
    year_index = np.tile(np.arange(n_years), n_firms)
    n = len(firm)
    year_effect = 0.05 * year_index
    fe = firm_effect[firm]
    eligible = prng.integers(2, size=n)
    propensity = 1 / (1 + np.exp(-(-1.0 + 1.6 * eligible + 0.8 * fe)))
    treated = (prng.random(n) < propensity).astype(int)
    size = 3 + 0.5 * fe + 0.02 * year_index + prng.normal(scale=0.5, size=n)
    age = (start_age[firm] + year_index).astype(float)
    leverage = np.clip(0.4 - 0.05 * fe + prng.normal(scale=0.15, size=n), 0, 1)
    log_sales = (
        2.0
        + 0.3 * treated
        + 0.4 * size
        + 0.01 * age
        - 0.5 * leverage
        + fe
        + year_effect
        + 0.1 * industry[firm]
        + prng.normal(scale=0.5, size=n)
    )
    patents = prng.poisson(np.exp(-0.5 + 0.2 * treated + 0.2 * size + 0.3 * fe))
    exporter = (
        prng.random(n) < 1 / (1 + np.exp(-(-1.0 + 0.5 * treated + 0.3 * size - 1.0)))
    ).astype(int)
    age[prng.random(n) < 0.01] = np.nan
    leverage[prng.random(n) < 0.02] = np.nan
    rd = np.clip(0.05 + 0.01 * fe + prng.normal(scale=0.02, size=n), 0, None)
    return pd.DataFrame(
        {
            "firm": firm,
            "year": 2010 + year_index,
            "industry": industry[firm],
            "state": state[firm],
            "treated": treated,
            "eligible": eligible,
            "size": size,
            "age": age,
            "leverage": leverage,
            "rd": rd,
            "sales": np.exp(log_sales),
            "patents": patents,
            "exporter": exporter,
            "weight": weight_firm[firm],
        }
    )
