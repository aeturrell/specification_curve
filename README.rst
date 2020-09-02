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

.. image:: https://static.pepy.tech/badge/specification-curve
        :alt: Downloads

.. image:: https://img.shields.io/pypi/pyversions/specification_curve.svg
        :target: https://pypi.python.org/pypi/specification_curve/
        :alt: Support Python versions


Specification Curve is a Python (3.6+) package that performs specification curve analysis.


* Free software: MIT license
* Documentation: https://specification-curve.readthedocs.io.

Quickstart
----------

Running

.. code-block:: python

   from specification_curve import specification_curve as specy
   from specification_curve import example as scdata
   df = scdata.load_example_data1()
   y_endog = 'y1'
   x_exog = 'x1'
   controls = ['c1', 'c2', 'group1', 'group2']
   sc = specy.SpecificationCurve(df, y_endog, x_exog, controls,
                                 cat_expand=['group2'])
   sc.fit()
   sc.plot()

produces

.. image:: https://raw.githubusercontent.com/aeturrell/specification_curve/master/docs/images/example.png
   :width: 600

Grey squares (black lines when there are many specifications) show whether
a variable is included in a specification or not. Blue markers and error bars
show whether the coefficient is significant (0.05).

Here's another example:

.. code-block:: python

   from specification_curve import specification_curve as specy
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
   sc = specy.SpecificationCurve(df, y_endog, x_exog, controls,
                                 cat_expand=['x_4'])
   sc.fit()
   sc.plot()


Features
--------

These examples use the first set of **example data**:

.. code-block:: python

    from specification_curve import specification_curve as specy
    from specification_curve import example as scdata
    df = scdata.load_example_data1()

* Expand fixed effects into mutually exclusive groups using ``cat_expand``

.. code-block:: python

    y_endog = 'y1'
    x_exog = 'x1'
    controls = ['c1', 'c2', 'group1', 'group2']
    sc = specy.SpecificationCurve(df, y_endog, x_exog, controls,
                                  cat_expand=['group1', 'group2'])
    sc.fit()
    sc.plot()

* Mutually exclude two variables using ``exclu_grp``

.. code-block:: python

    y_endog = 'y1'
    x_exog = 'x1'
    controls = ['c1', 'c2', 'group1', 'group2']
    sc = specy.SpecificationCurve(df, y_endog, x_exog, controls,
                                      exclu_grps=[['c1', 'c2']])
    sc.fit()
    sc.plot()

* Use multiple independent or dependent variables

.. code-block:: python

    x_exog = ['x1', 'x2']
    y_endog = 'y1'
    controls = ['c1', 'c2', 'group1', 'group2']
    sc = specy.SpecificationCurve(df, y_endog, x_exog, controls)
    sc.fit()
    sc.plot()

* Save plots to file (format is inferred from file extension)

.. code-block:: python

    sc = specy.SpecificationCurve(df, y_endog, x_exog, controls,
                                      cat_expand=['group1'])
    sc.fit()
    sc.plot(save_path='test_fig.pdf')

* Specification results stored in output DataFrame `df_r`

.. code-block:: python

    sc = specy.SpecificationCurve(df, y_endog, x_exog, controls)
    sc.fit()
    print(sc.df_r)

* Other `statsmodels` estimators (OLS is the default) can be used

.. code-block:: python

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
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
    sc.fit(estimator=sm.Logit)  # sm.Probit also works
    sc.plot()

* The style of specification flexes for very large numbers of specifications

.. code-block:: python

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
