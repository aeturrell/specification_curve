===================
Specification Curve
===================


.. image:: https://img.shields.io/pypi/v/specification_curve.svg
        :target: https://pypi.python.org/pypi/specification_curve

.. image:: https://img.shields.io/travis/aeturrell/specification_curve.svg
        :target: https://travis-ci.com/aeturrell/specification_curve

.. image:: https://readthedocs.org/projects/specification-curve/badge/?version=latest
        :target: https://specification-curve.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


Specification Curve is a Python package that performs specification curve analysis.


* Free software: MIT license
* Documentation: https://specification-curve.readthedocs.io.

Quickstart
----------

Running

.. code-block:: python

   from specification_curve import specification_curve as sc
   from specification_curve import example as scdata
   df = scdata.load_example_data1()
   y_endog = 'y2'
   x_exog = ['x1', 'x2']
   controls = ['c1', 'c2', 'group1', 'group2']
   df_r = sc.spec_curve(df, y_endog, x_exog, controls,
                        cat_expand=['group2'])

produces

.. image:: https://raw.githubusercontent.com/aeturrell/specification_curve/master/docs/images/example.png
   :width: 600

Grey squares (black lines when there are many specifications) show whether
a variable is included in a specification or not. Blue markers and error bars
show whether the coefficient is significant (0.05).

Here's another example:

.. code-block:: python

   from specification_curve import specification_curve as sc
   import numpy as np
   n_samples = 300
   np.random.seed(1332)
   x_1 = np.random.random(size=n_samples)
   x_2 = np.random.random(size=n_samples)
   x_3 = np.random.random(size=n_samples)
   x_4 = np.random.randint(2, size=n_samples)
   y = (0.8*x_1 + 0.1*x_2 + 0.5*x_3 + x_4*0.6 +
        + 2*np.random.randn(n_samples))
   df = pd.DataFrame([x_1, x_2, x_3, x_4, y],
                     ['x_1', 'x_2', 'x_3', 'x_4', 'y']).T
   # Set x_4 as a categorical variable
   df['x_4'] = df['x_4'].astype('category')
   df_r = sc.spec_curve(df, y_endog, x_exog, controls,
                        cat_expand=['x_4'])


Features
--------

These examples use the first set of **example data**:

.. code-block:: python

    from specification_curve import specification_curve as sc
    from specification_curve import example as scdata
    df = scdata.load_example_data1()

* Expand fixed effects into mutually exclusive groups using ``cat_expand``

.. code-block:: python

    y_endog = 'y1'
    x_exog = 'x1'
    controls = ['c1', 'c2', 'group1', 'group2']
    df_r = sc.spec_curve(df, y_endog, x_exog, controls,
                             cat_expand=['group1', 'group2'])

* Mutually exclude two variables using ``exclu_grp``

.. code-block:: python

    y_endog = 'y1'
    x_exog = 'x1'
    controls = ['c1', 'c2', 'group1', 'group2']
    df_r = sc.spec_curve(df, y_endog, x_exog, controls,
                     exclu_grps=[['c1', 'c2']])

* Use multiple independent or dependent variables

.. code-block:: python

    x_exog = ['x1', 'x2']
    y_endog = 'y1'
    controls = ['c1', 'c2', 'group1', 'group2']
    df_r = sc.spec_curve(df, y_endog, x_exog, controls)

* Save plots to file (format is inferred from file extension)

.. code-block:: python

    df_r = sc.spec_curve(df, y_endog, x_exog, controls, save_path='chart_name.pdf')

* Specification results stored in output DataFrame

.. code-block:: python

    df_r = sc.spec_curve(df, y_endog, x_exog, controls)
    print(df_r.head)                                                                               |   Coefficient | Specification                                |      bse | pvalues                                                                                                                                                                                                                                                              | SpecificationCounts                                                     | color_coeff   |   coeff_pvals |\n|--------------------:|:---------|:----------|:----------------------------------------------------------------------------------------|--------------:|:---------------------------------------------|---------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------|:--------------|--------------:|\n|                   0 | x1       | y2        | <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x7ffbef85a320> |      0.125466 | ['c1', 'c2', 'group1', 'group2', 'x1', 'y2'] | 0.200342 | {'x1': 0.5314363422953396, 'c1': 4.023143156706627e-123, 'c2': 9.266289401133827e-82, 'group1=0': 5.18596300838285e-07, 'group1=1': 0.00016924113833934303, 'group2=A': 2.9095422148517334e-18, 'group2=B': 0.02577443774493471, 'group2=C': 1.7427249638009747e-53} | Counter({'c1': 1, 'c2': 1, 'group1': 1, 'group2': 1, 'x1': 1, 'y2': 1}) | black         |      0.531436 |\n|                   1 | x1       | y2        | <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x7ffbefd18a58> |      0.125466 | ['c1', 'c2', 'x1', 'y2']                     | 0.200342 | {'x1': 0.5314363422953396, 'c1': 4.023143156706627e-123, 'c2': 9.266289401133827e-82, 'group1=0': 5.18596300838285e-07, 'group1=1': 0.00016924113833934303, 'group2=A': 2.9095422148517334e-18, 'group2=B': 0.02577443774493471, 'group2=C': 1.7427249638009747e-53} | Counter({'c1': 1, 'c2': 1, 'x1': 1, 'y2': 1})                           | black         |      0.531436 |\n|                   2 | x1       | y2        | <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x7ffbefcc4518> |      0.125466 | ['c1', 'c2', 'group2', 'x1', 'y2']           | 0.200342 | {'x1': 0.5314363422953396, 'c1': 4.023143156706627e-123, 'c2': 9.266289401133827e-82, 'group1=0': 5.18596300838285e-07, 'group1=1': 0.00016924113833934303, 'group2=A': 2.9095422148517334e-18, 'group2=B': 0.02577443774493471, 'group2=C': 1.7427249638009747e-53} | Counter({'c1': 1, 'c2': 1, 'group2': 1, 'x1': 1, 'y2': 1})              | black         |      0.531436 |\n|                   3 | x1       | y2        | <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x7ffbefce5710> |      0.125466 | ['c1', 'c2', 'group1', 'x1', 'y2']           | 0.200342 | {'x1': 0.5314363422953396, 'c1': 4.023143156706627e-123, 'c2': 9.266289401133827e-82, 'group1=0': 5.18596300838285e-07, 'group1=1': 0.00016924113833934303, 'group2=A': 2.9095422148517334e-18, 'group2=B': 0.02577443774493471, 'group2=C': 1.7427249638009747e-53} | Counter({'c1': 1, 'c2': 1, 'group1': 1, 'x1': 1, 'y2': 1})              | black         |      0.531436 |\n|                   4 | x1       | y2        | <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x7ffbefd3c550> |      0.25906  | ['c1', 'x1', 'y2']                           | 0.290318 | {'x1': 0.37265088717342676, 'c1': 1.5169358676571366e-80, 'group1=0': 0.042722321571069366, 'group1=1': 0.0018990200285307827, 'group2=A': 3.3577597149187116e-24, 'group2=B': 0.005064446699681006, 'group2=C': 3.392729150865205e-16}                              | Counter({'c1': 1, 'x1': 1, 'y2': 1})                                    | black         |      0.372651 |"


Similar Packages
----------------

In RStats, there is specr_ (which inspired many design choices in this package) and spec_chart_. Some of the example data in this package is the same as in specr_.

.. _specr: https://github.com/masurp/specr
.. _spec_chart: https://github.com/ArielOrtizBobea/spec_chart

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
