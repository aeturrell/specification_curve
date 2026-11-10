---
name: specification_curve
description: >
  Specification curve analysis for econometrics, built on pyfixest. Use when writing Python code that uses the specification_curve package.
license: MIT
compatibility: Requires Python >=3.12.
---

# Specification Curve

Specification curve analysis for econometrics, built on pyfixest

## Installation

```bash
pip install specification_curve
```

## API overview

### Classes

The main entry point. Describe the specifications, estimate them, and plot the curve.

- `SpecificationCurve`: Runs and plots every variant of a regression specification

### Functions

Plot a results table directly, and select subsets of specifications.

- `plot_curve`: Draws a specification curve
- `subsets_by`: Builds a ``subsets`` argument with one sample per level of a column

### Example data

Data sets used in the documentation and tests.

- `load_example_panel`: Generates a synthetic firm-year panel
- `load_example_data1`: Retrieves example data from a file included with the package
- `load_example_data2`: Generates example data with one outcome and four regressors
- `load_example_data3`: Generates example data with two outcomes, where the true effect of x1 is 0.4

## Resources

- [Full documentation](https://aeturrell.github.io/specification_curve/)
- [llms.txt](llms.txt) — Indexed API reference for LLMs
- [llms-full.txt](llms-full.txt) — Comprehensive documentation for LLMs
