#!/usr/bin/env python

"""Tests for `specification_curve` package."""


import unittest
import os


from specification_curve import specification_curve as sc
from specification_curve import example as edata


class TestSpecification_curve(unittest.TestCase):
    """Tests for `specification_curve` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_basic(self):
        """Test vanilla run."""
        df = edata.load_example_data2()
        y_endog = 'y'
        x_exog = 'x_1'
        controls = ['x_2', 'x_3', 'x_4']
        df_r = sc.spec_curve(df, y_endog, x_exog, controls)
        df_r.head()

    def test_001_fe_grp(self):
        """Test expand multiple FE groups."""
        df = edata.load_example_data1()
        y_endog = 'y1'
        x_exog = 'x1'
        controls = ['c1', 'c2', 'group1', 'group2']
        df_r = sc.spec_curve(df, y_endog, x_exog, controls,
                             cat_expand=['group1', 'group2'])
        df_r.head()

    def test_002_docs_feat_one(self):
        """Test docs feature 1."""
        df = edata.load_example_data1()
        y_endog = 'y1'
        x_exog = 'x1'
        controls = ['c1', 'c2', 'group1', 'group2']
        df_r = sc.spec_curve(df, y_endog, x_exog, controls,
                             cat_expand=['group1', 'group2'])
        df_r.head()

    def test_003_docs_feat_two(self):
        """Test docs feature 2."""
        df = edata.load_example_data1()
        y_endog = 'y1'
        x_exog = 'x1'
        controls = ['c1', 'c2', 'group1', 'group2']
        df_r = sc.spec_curve(df, y_endog, x_exog, controls,
                             exclu_grps=[['c1', 'c2']])
        df_r.head()

    def test_004_docs_feat_three(self):
        """Test docs feature 3."""
        df = edata.load_example_data1()
        x_exog = ['x1', 'x2']
        y_endog = 'y1'
        controls = ['c1', 'c2', 'group1', 'group2']
        df_r = sc.spec_curve(df, y_endog, x_exog, controls)
        df_r.head()

    def test_005_save_fig(self):
        """Test save fig"""
        df = edata.load_example_data1()
        x_exog = ['x1', 'x2']
        y_endog = 'y1'
        controls = ['c1', 'c2', 'group1', 'group2']
        df_r = sc.spec_curve(df, y_endog, x_exog, controls,
                             cat_expand=['group1'],
                             save_path='test_fig.pdf')
        df_r.head()
        os.remove('test_fig.pdf')
