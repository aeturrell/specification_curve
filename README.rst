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
--------

.. code-block:: Python

   import numpy as np
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
   
   specification_curve(df, y_endog, x_exog, controls)Some Ruby code.

Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
