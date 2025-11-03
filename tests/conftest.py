"""Pytest configuration file for specification_curve tests."""

import matplotlib

# Set matplotlib to use non-interactive backend to avoid tkinter issues on CI
# This must be done before importing pyplot
matplotlib.use("Agg")
