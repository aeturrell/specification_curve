=====
Usage
=====

To use Specification Curve in a project::

.. code-block:: python

   from specification_curve import specification_curve as sc
   from specification_curve import example as scdata
   df = scdata.load_example_data1()
   y_endog = 'y2'
   x_exog = ['x1', 'x2']
   controls = ['c1', 'c2', 'group1', 'group2']
   df_r = sc.spec_curve(df, y_endog, x_exog, controls, cat_expand=['group2'])

produces

.. image:: docs/images/example.png
   :width: 600

* TODO: add more documentation