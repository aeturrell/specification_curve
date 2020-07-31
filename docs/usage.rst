=====
Usage
=====

To use Specification Curve in a project::

.. code-block:: python

   from specification_curve import specification_curve as sc
   import numpy as np
   n_samples = 100
   x_1 = np.random.random(size=n_samples)
   x_2 = np.random.random(size=n_samples)
   x_3 = np.random.random(size=n_samples)
   x_4 = np.random.randint(2, size=n_samples)
   y = (0.8*x_1 + 0.1*x_2 + 0.5*x_3 + x_4*0.6 +
        + 2*np.random.randn(n_samples))
   
   df = pd.DataFrame([x_1, x_2, x_3, x_4, y],
                     ['x_1', 'x_2', 'x_3', 'x_4', 'y']).T
   y_endog = 'y'
   x_exog = 'x_1'
   controls = ['x_2', 'x_3', 'x_4']
   # Set x_4 as a categorical variable
   df['x_4'] = df['x_4'].astype('category')
   
   df_r = sc.spec_curve(df, y_endog, x_exog, controls,
                           cat_expand=['x_4'])


Documentation for the Code
**************************

.. automodule:: specification_curve

.. automodule:: specification_curve.specification_curve
   :members: