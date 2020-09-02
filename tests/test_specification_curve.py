#!/usr/bin/env python

"""Tests for `specification_curve` package."""

import unittest
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from specification_curve import specification_curve as specy
from specification_curve import example as scdata


class TestSpecification_curve(unittest.TestCase):
    """Tests for `specification_curve` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_basic(self):
        """Test vanilla run."""
        df = scdata.load_example_data2()
        y_endog = 'y'
        x_exog = 'x_1'
        controls = ['x_2', 'x_3', 'x_4']
        sc = specy.SpecificationCurve(df, y_endog, x_exog, controls)
        sc.fit()
        sc.plot()
        sc.df_r.head()

    def test_001_fe_grp(self):
        """Test expand multiple FE groups. Docs feature 1."""
        df = scdata.load_example_data1()
        y_endog = 'y1'
        x_exog = 'x1'
        controls = ['c1', 'c2', 'group1', 'group2']
        sc = specy.SpecificationCurve(df, y_endog, x_exog, controls,
                                      cat_expand=['group1', 'group2'])
        sc.fit()
        sc.plot()
        sc.df_r.head()

    def test_002_docs_feat_two(self):
        """Test docs feature 2."""
        df = scdata.load_example_data1()
        y_endog = 'y1'
        x_exog = 'x1'
        controls = ['c1', 'c2', 'group1', 'group2']
        sc = specy.SpecificationCurve(df, y_endog, x_exog, controls,
                                      exclu_grps=[['c1', 'c2']])
        sc.fit()
        sc.plot()
        sc.df_r.head()

    def test_004_docs_feat_three(self):
        """Test docs feature 3: multiple dependent or independent variables"""
        df = scdata.load_example_data1()
        x_exog = ['x1', 'x2']
        y_endog = 'y1'
        controls = ['c1', 'c2', 'group1', 'group2']
        sc = specy.SpecificationCurve(df, y_endog, x_exog, controls)
        sc.fit()
        sc.plot()
        sc.df_r.head()

    def test_005_save_fig(self):
        """Test save fig.."""
        df = scdata.load_example_data1()
        x_exog = ['x1', 'x2']
        y_endog = 'y1'
        controls = ['c1', 'c2', 'group1', 'group2']
        sc = specy.SpecificationCurve(df, y_endog, x_exog, controls,
                                      cat_expand=['group1'])
        sc.fit()
        sc.plot(save_path='test_fig.pdf')
        os.remove('test_fig.pdf')

    def test_006_logit_estimator(self):
        """ Test running with different statsmodels estimators -
            here logitistic
        """
        n_samples = 1000
        x_2 = np.random.randint(2, size=n_samples)
        x_1 = np.random.random(size=n_samples)
        x_3 = np.random.randint(3, size=n_samples)
        x_4 = np.random.random(size=n_samples)
        x_5 = x_1 + 0.05*np.random.randn(n_samples)
        x_beta = -1 + 3.5*x_1 + 0.2*x_2 + 0.3*x_3
        prob = 1/(1 + np.exp(-x_beta))
        y = np.random.binomial(n=1, p=prob, size=n_samples)
        y2 = np.random.binomial(n=1, p=prob*0.98, size=n_samples)
        df = pd.DataFrame([x_1, x_2, x_3, x_4, x_5, y, y2],
                          ['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'y', 'y2']).T
        y_endog = ['y', 'y2']
        x_exog = ['x_1', 'x_5']
        controls = ['x_3', 'x_2', 'x_4']
        sc = specy.SpecificationCurve(df, y_endog, x_exog, controls,
                                      cat_expand='x_3')
        sc.fit(estimator=sm.Logit)
        sc.plot()
        sc.df_r.head()

    def test_007_probit_estimator(self):
        """
        Running with different statsmodels estimators - here probit
        """
        n_samples = 1000
        x_2 = np.random.randint(2, size=n_samples)
        x_1 = np.random.random(size=n_samples)
        x_3 = np.random.randint(3, size=n_samples)
        x_4 = np.random.random(size=n_samples)
        x_5 = x_1 + 0.05*np.random.randn(n_samples)
        x_beta = -1 + 3.5*x_1 + 0.2*x_2 + 0.3*x_3
        prob = norm.cdf(x_beta)
        y = np.random.binomial(n=1, p=prob, size=n_samples)
        y2 = np.random.binomial(n=1, p=prob*0.98, size=n_samples)
        df = pd.DataFrame([x_1, x_2, x_3, x_4, x_5, y, y2],
                          ['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'y', 'y2']).T
        y_endog = ['y', 'y2']
        x_exog = ['x_1', 'x_5']
        controls = ['x_3', 'x_2', 'x_4']
        sc = specy.SpecificationCurve(df, y_endog, x_exog, controls,
                                      cat_expand='x_3')
        sc.fit(estimator=sm.Probit)
        sc.plot()
        sc.df_r.head()

    def test_008_large_no_specifications(self):
        """
        Test a very large set of specifications
        """
        n_samples = 400
        # Number of dimensions of continuous
        # random variables
        n_dim = 8
        c_rnd_vars = np.random.random(size=(n_dim, n_samples))
        c_rnd_vars_names = [f'c_{i}' for i in range(np.shape(c_rnd_vars)[0])]
        y_1 = (0.3*c_rnd_vars[0, :] +
               0.5*c_rnd_vars[1, :])
        y_2 = y_1 + 0.05*np.random.randn(n_samples)
        df = pd.DataFrame([y_1, y_2], ['y1', 'y2']).T
        for i, col_name in enumerate(c_rnd_vars_names):
            df[col_name] = c_rnd_vars[i, :]
        controls = c_rnd_vars_names[1:]
        sc = specy.SpecificationCurve(df, ['y1', 'y2'], c_rnd_vars_names[0],
                                      controls)
        sc.fit()
        sc.plot()
