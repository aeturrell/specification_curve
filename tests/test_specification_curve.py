#!/usr/bin/env python

"""Tests for `specification_curve` package."""


import unittest
import numpy as np
import pandas as pd


from specification_curve import specification_curve as sc


class TestSpecification_curve(unittest.TestCase):
    """Tests for `specification_curve` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""
        n_samples = 5000
        x_1 = np.random.random(size=n_samples)
        x_2 = np.random.random(size=n_samples)
        x_3 = np.random.randint(2, size=n_samples)
        x_4 = np.random.random(size=n_samples)
        x_5 = np.random.randint(4, size=n_samples)
        y = (0.5*x_1 + 0.8*x_2 + 0.2*x_3 + x_5*0.2 +
             + np.random.randn(n_samples))

        df = pd.DataFrame([x_1, x_2, x_3, x_4, x_5, y],
                          ['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'y']).T
        y_endog = 'y'
        x_exog = 'x_1'
        controls = ['x_2', 'x_3', 'x_4', 'x_5']
        sc.spec_curve(df, y_endog, x_exog, controls)
