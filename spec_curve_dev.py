import specification_curve as sc

df = sc.load_example_data2()
y_endog = "y"
x_exog = "x_1"
controls = ["x_2", "x_3", "x_4"]
sc = sc.SpecificationCurve(df, y_endog, x_exog, controls)
sc.fit()
