"""
Specification Curve
-------------------
A package that produces specification curve analysis.
"""

from importlib.metadata import PackageNotFoundError, version

from .curve import SpecificationCurve
from .data import (
    load_example_data1,
    load_example_data2,
    load_example_data3,
    load_example_panel,
)
from .grid import subsets_by
from .plot import plot_curve

try:
    __version__ = version("specification_curve")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    "SpecificationCurve",
    "load_example_data1",
    "load_example_data2",
    "load_example_data3",
    "load_example_panel",
    "plot_curve",
    "subsets_by",
]
