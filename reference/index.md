# API Reference


## Classes


The main entry point. Describe the specifications, estimate them, and plot the curve.


[SpecificationCurve](SpecificationCurve.md#specification_curve.SpecificationCurve)  
Runs and plots every variant of a regression specification.


## Functions


Plot a results table directly, and select subsets of specifications.


[plot_curve()](plot_curve.md#specification_curve.plot_curve)  
Draws a specification curve.

[subsets_by()](subsets_by.md#specification_curve.subsets_by)  
Builds a `subsets` argument with one sample per level of a column.


## Example data


Data sets used in the documentation and tests.


[load_example_panel()](load_example_panel.md#specification_curve.load_example_panel)  
Generates a synthetic firm-year panel.

[load_example_data1()](load_example_data1.md#specification_curve.load_example_data1)  
Retrieves example data from a file included with the package.

[load_example_data2()](load_example_data2.md#specification_curve.load_example_data2)  
Generates example data with one outcome and four regressors.

[load_example_data3()](load_example_data3.md#specification_curve.load_example_data3)  
Generates example data with two outcomes, where the true effect of x1 is 0.4.
