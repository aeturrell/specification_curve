import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import specification_curve as sc
import statsmodels.api as sm

tic = time.perf_counter()

df = sc.load_example_data3()
sco = sc.SpecificationCurve(df, formula="y1 | y2 ~ x1 + c1 | c2 | c3")
sco.fit()
sco.fit_null(n_boot=100)


n_samples = 10000
x_2 = np.random.randint(2, size=n_samples)
x_1 = np.random.random(size=n_samples)
x_3 = np.random.randint(3, size=n_samples)
x_4 = np.random.random(size=n_samples)
x_5 = x_1 + 0.05 * np.random.randn(n_samples)
x_beta = -1 + 3.5 * x_1 + 0.2 * x_2 + 0.3 * x_3
prob = 1 / (1 + np.exp(-x_beta))
y = np.random.binomial(n=1, p=prob, size=n_samples)
y2 = np.random.binomial(n=1, p=prob * 0.98, size=n_samples)
df = pd.DataFrame(
    [x_1, x_2, x_3, x_4, x_5, y, y2],
    ["x_1", "x_2", "x_3", "x_4", "x_5", "y", "y2"],
).T
y_endog = ["y", "y2"]
x_exog = ["x_1", "x_5"]
controls = ["x_3", "x_2", "x_4"]
scol = sc.SpecificationCurve(df, y_endog, x_exog, controls, cat_expand="x_3")
scol.fit(estimator=sm.Logit)

toc = time.perf_counter()
elapsed = toc - tic

time_stamp_string = datetime.now().strftime("%Y%m%d_%H%M%S")

with open(Path(f"outputs/benchmark_time_{time_stamp_string}.txt"), "w") as f:
    f.write(f"{elapsed:.2f}")
