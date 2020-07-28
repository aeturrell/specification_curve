"""Main module."""

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
import sklearn


def _pretty_plots():
    jsonPlotSettings = {'xtick.labelsize': 16,
                        'ytick.labelsize': 20,
                        'xtick.labelsize': 20,
                        'font.size': 22,
                        'figure.figsize': (10, 5),
                        'axes.titlesize': 22,
                        'axes.labelsize': 20,
                        'lines.linewidth': 2,
                        'lines.markersize': 6,
                        'legend.fontsize': 11,
                        'mathtext.fontset': 'stix',
                        'font.family': 'STIXGeneral'}
    plt.style.use(jsonPlotSettings)


def _get_shuffled_results(df, y_endog, x_exog, all_combs,
                          num_shuffles=50):
    """Performs fake regressions for a specification curve.

    Uses OLS to perform all variants of y = beta*x + (other factors).
    Considers the full set of reasonable specifications jointly; how
    inconsistent are the results with the null hypothesis of no effect?
    Uses a permutation test which shuffles up the data and re-runs
    the regressions. It assumes exchangeability, i.e. that the rows are not
    related in any way. The shuffled datasets maintain all the other
    features of the original one (e.g., collinearity, time trends, skewness,
    etc.) except we now know there is no link between (shuffled) names and
    fatalities; the null is true by construction.

    :dataframe df: pandas dataframe in tidy format
    :string y_endog: dependent variable
    :string x_exog: independent variable
    :list[list[string]] all_combs: combinations of specifications
    :list[string] controls: continuous variables to control for
    :int num_shuffles: how many variants to do of *each* specification

    :returns: Pandas dataframe
    """
    all_results_shuffle = []
    for i in range(num_shuffles):
        df_shuffle = df.copy(deep=True)
        df_shuffle[x_exog] = sklearn.utils.shuffle(df[x_exog].values)
        results_shuffle = [sm.OLS(df_shuffle[y_endog],
                                  df_shuffle[[x_exog]+x]).fit()
                           for x in all_combs]
        all_results_shuffle.append(results_shuffle)
    df_r_shuffle = pd.DataFrame([[x.params[x_exog]
                                  for x in y] for y in all_results_shuffle])
    # There are multiple shuffled regressions for each specification, so take
    # the median of num_shuffles
    med_shuffle = df_r_shuffle.quantile(
        0.5).sort_values().reset_index().drop('index', axis=1)
    return med_shuffle


def _spec_curve_regression(df, y_endog, x_exog, controls):
    """Performs all regressions for a specification curve.

    Uses OLS to perform all control & variants of y = beta*x + (other factors).
    Assumes that all controls and fixed effects should be varied.

    :dataframe df: pandas dataframe in tidy format
    :string y_endog: dependent variable
    :string x_exog: independent variable
    :list[string] controls: variables to control for

    :returns: Statmodels RegressionResults object
    """
    all_combs = [combinations(controls, k) for k in range(len(controls) + 1)]
    # Flatten this into a single list of tuples
    all_combs = [item for sublist in all_combs for item in sublist]
    # Turn all the tuples into lists
    all_combs = [list(x) for x in all_combs]
    # Run regressions
    all_results = [sm.OLS(df[y_endog], df[[x_exog]+reg_vars]).fit()
                   for reg_vars in all_combs]
    # Get coefficient values and specifications
    df_r = pd.DataFrame([x.params[x_exog] for x in all_results],
                        columns=['Coefficient'])
    df_r['Specification'] = all_combs
    # Grab the shuffled results while everything is in the original order
    df_r['Shuffle_coeff'] = _get_shuffled_results(df, y_endog, x_exog,
                                                  all_combs,
                                                  num_shuffles=50)
    # Get std err and pvalues
    df_r['bse'] = [x.bse[x_exog] for x in all_results]
    df_r['pvalues'] = [x.pvalues for x in all_results]
    df_r['pvalues'] = df_r['pvalues'].apply(lambda x: dict(x))
    # Re-order by coefficient
    df_r = df_r.sort_values('Coefficient')
    df_r = df_r.reset_index().drop('index', axis=1)
    df_r.index.names = ['Specification No.']
    df_r['Specification'] = df_r['Specification'].apply(lambda x: sorted(x))
    df_r['SpecificationCounts'] = df_r['Specification'].apply(
        lambda x: Counter(x))
    return df_r


def _controls_matrix(df_r):
    df_ctr_mat = df_r['pvalues'].apply(pd.Series)
    df_ctr_mat = df_ctr_mat.reindex(sorted(df_ctr_mat.columns), axis=1)
    # Insignificant
    df_ctr_mat[np.abs(df_ctr_mat) > 0.05] = 1
    # Significant
    df_ctr_mat[df_ctr_mat <= 0.05] = 0.
    df_ctr_mat['Coefficient'] = df_r['Coefficient']
    return df_ctr_mat


def plot_spec_curve(df_r, df_ctr_mat, controls, save_path=None):
    """Plots a specification curve.



    :dataframe df_r: pandas dataframe with regression specification results
    :dataframe df_ctr_mat: pandas dataframe of inclusion and significance
    :list[string] controls: variables to control for
    :string save_path: File path to save figure to.

    """
    _pretty_plots()
    plt.close('all')
    fig, axarr = plt.subplots(2, sharex=True, figsize=(10, 10))
    for ax in axarr:
        ax.yaxis.major.formatter._useMathText = True
    axarr[0].scatter(df_r.index, df_r['Coefficient'],
                     lw=3.,
                     s=0.6,
                     color='b',
                     label='Coefficient')
    axarr[0].scatter(df_r.index,
                     df_r['Shuffle_coeff'],
                     lw=3.,
                     s=0.6,
                     color='r',
                     marker='d',
                     label='Coefficient under null (median over bootstraps)')
    axarr[0].axhline(color='k', lw=0.5)
    axarr[0].axhline(y=np.median(df_r['Coefficient']),
                     color='k',
                     alpha=0.3,
                     label='Median coefficient',
                     dashes=[12, 5])
    axarr[0].fill_between(df_r.index,
                          df_r['Coefficient']+df_r['bse'],
                          df_r['Coefficient']-df_r['bse'],
                          color='b',
                          alpha=0.3)
    axarr[0].legend(frameon=True, loc='lower right',
                    ncol=1, handlelength=4, markerscale=10)
    axarr[0].set_ylabel('Coefficient')
    axarr[0].set_title('Specification curve analysis')
    cmap = plt.cm.plasma
    cmap.set_bad('white', 1.)
    axarr[1].imshow(df_ctr_mat[controls].T, aspect='auto',
                    cmap=cmap, interpolation='None')
    axarr[1].set_ylabel('Controls')
    axarr[1].set_xlabel(df_r.index.name)
    axarr[1].set_yticks(range(len(controls)))
    axarr[1].set_yticklabels(controls)
    plt.subplots_adjust(wspace=0, hspace=0.05)
    if(save_path is not None):
        plt.save(save_path)
    plt.show()


def specification_curve(df, y_endog, x_exog, controls, save_path=None):
    df_r = _spec_curve_regression(df, y_endog, x_exog, controls=controls)
    df_ctr_mat = _controls_matrix(df_r)
    plot_spec_curve(df_r, df_ctr_mat, controls, save_path=save_path)
    return df_r
