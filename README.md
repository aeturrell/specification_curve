# Specification Curve

Specification Curve is a Python package that performs specification curve analysis; it helps with the problem of the "garden of forking paths" (many defensible choices) when doing analysis by running many regressions and summarising the effects in an easy to digest chart.

[![PyPI](https://img.shields.io/pypi/v/specification_curve.svg)](https://pypi.org/project/specification_curve/)
[![Status](https://img.shields.io/pypi/status/specification_curve.svg)](https://pypi.org/project/specification_curve/)
[![Python Version](https://img.shields.io/pypi/pyversions/specification_curve)](https://pypi.org/project/specification_curve)
[![License](https://img.shields.io/pypi/l/specification_curve)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/aeturrell/specification_curve/workflows/Tests/badge.svg)](https://github.com/aeturrell/specification_curve/actions?workflow=Tests)
[![Codecov](https://codecov.io/gh/aeturrell/specification_curve/branch/main/graph/badge.svg)](https://codecov.io/gh/aeturrell/specification_curve)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/aeturrell/438fb066e4471312667268669cef2c11/specification_curve-examples.ipynb)
[![DOI](https://zenodo.org/badge/282989537.svg)](https://zenodo.org/badge/latestdoi/282989537)
[![Downloads](https://static.pepy.tech/badge/specification-curve)](https://pepy.tech/project/Specification_curve)

![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![macOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)

[![Source](https://img.shields.io/badge/source%20code-github-lightgrey?style=for-the-badge)](https://github.com/aeturrell/specification_curve)

[Go to the full documentation for **Specification Curve**](https://aeturrell.github.io/specification_curve/).

## Quickstart

You can try out specification curve right now in [Google Colab](https://colab.research.google.com/gist/aeturrell/438fb066e4471312667268669cef2c11/specification_curve-examples.ipynb).

Here's an example of using **Specification Curve**.

```python
# import specification curve
import specification_curve as specy


# Generate some fake data
# ------------------------
import numpy as np
import pandas as pd
# Set seed for random numbers
seed_for_prng = 78557
# prng=probabilistic random number generator
prng = np.random.default_rng(seed_for_prng)
n_samples = 400
# Number of dimensions
n_dim = 4
c_rnd_vars = prng.random(size=(n_dim, n_samples))
y_1 = (0.4*c_rnd_vars[0, :] -  # THIS IS THE TRUE VALUE OF THE COEFFICIENT
       0.2*c_rnd_vars[1, :] +
       0.3*prng.standard_normal(n_samples))
# Next line causes y_2 ests to be much more noisy
y_2 = y_1 - 0.5*np.abs(prng.standard_normal(n_samples))
# Put data into dataframe
df = pd.DataFrame([y_1, y_2], ['y1', 'y2']).T
df["x_1"] = c_rnd_vars[0, :]
df["c_1"] = c_rnd_vars[1, :]
df["c_2"] = c_rnd_vars[2, :]
df["c_3"] = c_rnd_vars[3, :]

# Specification Curve Analysis
#-----------------------------
sc = specy.SpecificationCurve(
    df,
    y_endog=['y1', 'y2'],
    x_exog="x_1",
    controls=["c_1", "c_2", "c_3"])
sc.fit()
sc.plot()
```

Grey squares (black lines when there are many specifications) show whether a variable is included in a specification or not. Blue or red markers and error bars show whether the coefficient is positive and significant (at the 0.05 level) or red and significant, respectively.

## Installation

You can install **Specification Curve** via pip:

```bash
$ pip install specification-curve
```

To install the development version from git, use:

```bash
$ pip install git+https://github.com/aeturrell/specification_curve.git
```

## Citing Specification Curve

```text
@misc{aeturrell_2022_7264033,
  author       = {Arthur Turrell},
  title        = {Specification Curve: v0.3.1},
  month        = oct,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v0.3.1},
  doi          = {10.5281/zenodo.7264033},
  url          = {https://doi.org/10.5281/zenodo.7264033}
}
```

Using **Specification Curve** in your paper? Let us know by raising an issue beginning with "citation".

## License

Distributed under the terms of the [MIT license](https://opensource.org/licenses/MIT).

## Credits

The package is built with [poetry](https://python-poetry.org/), while the documentation is built with [Jupyter Book](https://jupyterbook.org). Tests are run with [nox](https://nox.thea.codes/en/stable/).

## Similar Packages

In RStats, there is [specr](https://github.com/masurp/specr) (which inspired many design choices in this package) and [spec_chart](https://github.com/ArielOrtizBobea/spec_chart). Some of the example data in this package is the same as in specr, but please note that results have not been cross-checked across packages.
