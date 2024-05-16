# Specification Curve

Specification Curve is a Python package that performs specification curve analysis; it helps with the problem of the "garden of forking paths" (many defensible choices) when doing analysis by running many regressions and summarising the estimated coefficients in an easy to digest chart.

[![PyPI](https://img.shields.io/pypi/v/specification_curve.svg)](https://pypi.org/project/specification_curve/)
[![Status](https://img.shields.io/pypi/status/specification_curve.svg)](https://pypi.org/project/specification_curve/)
[![Python Version](https://img.shields.io/pypi/pyversions/specification_curve)](https://pypi.org/project/specification_curve)
[![License](https://img.shields.io/pypi/l/specification_curve)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/aeturrell/specification_curve/workflows/Tests/badge.svg)](https://github.com/aeturrell/specification_curve/actions?workflow=Tests)
[![Codecov](https://codecov.io/gh/aeturrell/specification_curve/branch/main/graph/badge.svg)](https://codecov.io/gh/aeturrell/specification_curve)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aeturrell/specification_curve/blob/master/docs/features.ipynb)
[![DOI](https://zenodo.org/badge/282989537.svg)](https://zenodo.org/badge/latestdoi/282989537)
[![Downloads](https://static.pepy.tech/badge/specification-curve)](https://pepy.tech/project/Specification_curve)

![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![macOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)

[![Source](https://img.shields.io/badge/source%20code-github-lightgrey?style=for-the-badge)](https://github.com/aeturrell/specification_curve)

[Go to the full documentation for **Specification Curve**](https://aeturrell.github.io/specification_curve/).

## Quickstart

You can try out specification curve right now in [Google Colab](https://colab.research.google.com/github/aeturrell/specification_curve/blob/master/docs/features.ipynb). To install the package in Colab, run `!pip install specification_curve` in a new code cell.

Alternatively you can see examples on the [features page](https://aeturrell.github.io/specification_curve/features.html) of the documentation.

## Installation

You can install **Specification Curve** via pip:

```bash
pip install specification-curve
```

To install the development version from git, use:

```bash
pip install git+https://github.com/aeturrell/specification_curve.git
```

## Citing Specification Curve

You can find full citation information at the following link: [https://zenodo.org/badge/latestdoi/282989537](https://zenodo.org/badge/latestdoi/282989537).

Using **Specification Curve** in your paper? Let us know by raising an issue beginning with "citation".

## License

Distributed under the terms of the [MIT license](https://opensource.org/licenses/MIT).

## Similar Packages

In RStats, there is [specr](https://github.com/masurp/specr) (which inspired many design choices in this package) and [spec_chart](https://github.com/ArielOrtizBobea/spec_chart). Some of the example data in this package is the same as in specr, but please note that results have not been cross-checked across packages.
