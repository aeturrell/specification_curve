""" Examples """

import numpy as np
import pandas as pd
from specification_curve import specification_curve as sc


def load_example_data1():
    # Example data
    df = pd.read_csv('specification_curve/data/example_data.csv',
                     index_col=0)
    num_cols = [x for x in df.columns if x not in ['group1', 'group2']]
    for col in num_cols:
        df[col] = df[col].astype(np.double)
    cat_cols = [x for x in df.columns if x not in num_cols]
    for col in cat_cols:
        df[col] = df[col].astype('category')
    return df


def load_example_data2():
    n_samples = 1000
    np.random.seed(1332)
    x_1 = np.random.random(size=n_samples)
    x_2 = np.random.random(size=n_samples)
    x_3 = np.random.randint(2, size=n_samples)
    x_4 = np.random.random(size=n_samples)
    x_5 = np.random.randint(4, size=n_samples)
    y = (0.5*x_1 + 0.8*x_2 + 0.2*x_3 + x_5*0.2 +
         + np.random.randn(n_samples))

    df = pd.DataFrame([x_1, x_2, x_3, x_4, x_5, y],
                      ['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'y']).T
    return df


# Example dataset 1
df = load_example_data1()
x_exog = 'x2'
y_endog = 'y2'
ctrls = ['c1', 'c2', 'group1', 'group2']

df_r = sc.spec_curve(df, y_endog, x_exog, ctrls)
df_r = sc.spec_curve(df, y_endog, x_exog, ctrls,
                     cat_expand=['group1'])
df_r = sc.spec_curve(df, y_endog, x_exog, ctrls,
                     exclu_grps=[['c1', 'c2']])
df_r = sc.spec_curve(df, y_endog, x_exog, ctrls,
                     cat_expand=['group1', 'group2'])
df_r = sc.spec_curve(df, y_endog, x_exog, ctrls,
                     cat_expand=['group1', 'group2'],
                     exclu_grps=[['c1', 'c2']])
# Desired behaviour here is no overlap between grp1 and grp2
df_r = sc.spec_curve(df, y_endog, x_exog, ctrls,
                     cat_expand=['group1', 'group2'],
                     exclu_grps=[['group1', 'group2']])
df_r = sc.spec_curve(df, y_endog, x_exog, ctrls,
                     cat_expand=['group1', 'group2'],
                     exclu_grps=[['group2']])


# Example dataset 2
df = load_example_data2()
y_endog = 'y'
x_exog = 'x_1'
controls = ['x_2', 'x_3', 'x_4', 'x_5']
exclu_grps = [['x_4', 'x_5']]

# Front page example
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
