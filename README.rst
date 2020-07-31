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

.. image:: docs/images/example.png
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

    df = edata.load_example_data1()

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
