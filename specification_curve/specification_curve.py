"""Main module."""

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
import sklearn
import copy
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from math import log10, floor


def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))


def _pretty_plots():
    jsonPlotSettings = {'ytick.labelsize': 20,
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


def _excl_combs(lst, r, excludes):
    """lst = [1, 2, 3, 4, 5, 6]
       excludes = [{1, 3}]
       gen = _excl_combs(lst, 2, excludes)
    """
    if(excludes != [[]]):
        return [comb for comb in combinations(lst, r)
                if not any(e.issubset(comb) for e in excludes)]
    else:
        return list(combinations(lst, r))


def _reg_func(df, y_endog, x_exog, reg_vars):
    # NB: get dummies
    # transforms by default any col that is object or cat
    xf = pd.get_dummies(df, prefix_sep='=')
    new_cols = [x for x in xf.columns if x not in df.columns]
    gone_cols = [x for x in df.columns if x not in xf.columns]
    reg_vars_here = copy.copy(reg_vars)
    reg_vars_here.extend(new_cols)
    [reg_vars_here.remove(x) for x in gone_cols if x in reg_vars_here]
    return sm.OLS(xf[y_endog], xf[[x_exog]+reg_vars_here]).fit()


def _spec_curve_regression(xdf, y_endog, x_exog, controls,
                           exclu_grps=[[]],
                           cat_expand=[]):
    """Performs all regressions for a specification curve.

    Uses OLS to perform all control & variants of y = beta*x + (other factors).
    Assumes that all controls and fixed effects should be varied.

    :dataframe df: pandas dataframe in tidy format
    :string y_endog: dependent variable
    :string x_exog: independent variable
    :list[string] controls: variables to control for

    :returns: Statmodels RegressionResults object
    """
    # Make sure exlu grps is a list of lists
    #todo
    #
    df = xdf.copy()
    controls = copy.copy(controls)
    init_cols = [y_endog] + [x_exog] + controls
    df = df[init_cols]
    new_cols = []
    if(cat_expand != []):
        df = pd.get_dummies(df, columns=cat_expand, prefix_sep='=')
        new_cols = [x for x in df.columns if x not in init_cols]
        # Now change the controls
        [controls.remove(x) for x in cat_expand]
        controls.extend(new_cols)
        # Create mapping from cat expand to new cols
        oldnew = dict(zip([x for x in cat_expand],
                        [[x for x in new_cols if y in x] for y in cat_expand]))
        # Now exclude the groups that combine any new cols
        for x in cat_expand:
            if(exclu_grps == [[]]):
                exclu_grps = [oldnew[x]]
            else:
                exclu_grps.append(oldnew[x])
    # Now change the exclu_grps names, if they exist, into new names
    # TODO

    # dict_
    # new_cols
    # if(exclu_grps != [[]]):

    # # Turn mutually exclusive groups into sets
    if(exclu_grps != [[]]):
        exclu_grps = [set(x) for x in exclu_grps]
    # Get all combinations excluding mutually excl groups
    all_combs = [_excl_combs(controls, k, exclu_grps)
                 for k in range(len(controls)+1)]
    # Flatten this into a single list of tuples
    all_combs = [item for sublist in all_combs for item in sublist]
    # Turn all the tuples into lists
    all_combs = [list(x) for x in all_combs]
    # TODO: magic to turn eg group1, group2 into dummy vars here
    # Run regressions eg using pd.get_dummies(df['group1'])
    all_results = [_reg_func(df, y_endog, x_exog, reg_vars)
                   for reg_vars in all_combs]
    # Get coefficient values and specifications
    df_r = pd.DataFrame([x.params[x_exog] for x in all_results],
                        columns=['Coefficient'])
    df_r['Specification'] = all_combs
    # Grab the shuffled results while everything is in the original order
    # df_r['Shuffle_coeff'] = _get_shuffled_results(df, y_endog, x_exog,
    #                                               all_combs,
    #                                               num_shuffles=50)
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


def plot_spec_curve(df_r, x_exog, controls, save_path=None):
    """Plots a specification curve.



    :dataframe df_r: pandas dataframe with regression specification results
    :dataframe df_ctr_mat: pandas dataframe of inclusion and significance
    :list[string] controls: variables to control for
    :string save_path: File path to save figure to.

    """
    _pretty_plots()
    plt.close('all')
    fig, axarr = plt.subplots(2, sharex=True, figsize=(12, 8))
    for ax in axarr:
        ax.yaxis.major.formatter._useMathText = True
    # axarr[0].scatter(df_r.index,
    #                  df_r['Shuffle_coeff'],
    #                  lw=3.,
    #                  s=1,
    #                  color='r',
    #                  marker='D',
    #                  label='Coefficient under null')
    # axarr[0].axhline(color='k', lw=0.5)
    axarr[0].axhline(y=np.median(df_r['Coefficient']),
                     color='k',
                     lw=0.3,
                     alpha=1,
                     label='Median coefficient',
                     dashes=[12, 5])
    # Colour the significant ones differently
    df_r['color_coeff'] = 'grey'
    df_r['coeff_pvals'] = df_r['pvalues'].apply(lambda x: x[x_exog])
    df_r.loc[df_r['coeff_pvals'] < 0.05, 'color_coeff'] = 'blue'
    for color in df_r['color_coeff'].unique():
        slice_df_r = df_r.loc[df_r['color_coeff'] == color]
        axarr[0].errorbar(slice_df_r.index, slice_df_r['Coefficient'],
                          yerr=slice_df_r['bse'],
                          ls='none', color=color, alpha=0.6, zorder=1,
                          elinewidth=5)
    axarr[0].scatter(df_r.index, df_r['Coefficient'],
                     s=5.,
                     color='grey',
                     alpha=0.6,
                     marker='o',
                     label='Coefficient',
                     zorder=3)
    axarr[0].legend(frameon=True, loc='lower right',
                    ncol=1, handlelength=2, markerscale=2)
    axarr[0].set_ylabel('Coefficient')
    axarr[0].set_title('Specification curve analysis')
    ylims = axarr[0].get_ylim()
    axarr[0].set_ylim(round_to_1(ylims[0]), round_to_1(ylims[1]))
    # drawing squares
    df_spec = df_r['SpecificationCounts'].apply(pd.Series).fillna(0.)
    df_spec = df_spec.replace(0., False).replace(1., True)
    df_spec = df_spec.T
    df_spec = df_spec.sort_index()
    # TODO If all of a cat var is always true or false, collapse down to single
    # line
    # ? pd.melt(df_spec.reset_index(), id_vars=['Specification No.',
                                        # 'c1', 'c2'])
    wid = 0.8
    hei = wid/3
    color_dict = {True: '#9B9B9B', False: '#F2F2F2'}
    for i, row_name in enumerate(df_spec.index):
        for j, col_name in enumerate(df_spec.columns):
            color = color_dict[df_spec.iloc[i, j]]
            sq = patches.Rectangle(
                (j-wid/2, i-hei/2), wid, hei, fill=True, color=color)
            axarr[1].add_patch(sq)
    axarr[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axarr[1].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axarr[1].set_yticks(range(len(list(df_spec.index.values))))
    axarr[1].set_yticklabels(list(df_spec.index.values))
    axarr[1].set_xticklabels([])
    axarr[1].set_xlim(-0.5, len(df_spec.columns))
    # All unique possible combs:
    mega_list = list(set(df_r['Specification'].sum()))
    axarr[1].set_ylim(-0.5, len(mega_list)-0.5)
    for place in ['right', 'top', 'bottom', 'left']:
        axarr[1].spines[place].set_visible(False)
    for ax in axarr:
        ax.set_xticks([], minor=True)
        ax.set_xticks([])
    plt.subplots_adjust(wspace=0, hspace=0.05)
    if(save_path is not None):
        plt.save(save_path)
    plt.show()


def specification_curve(df, y_endog, x_exog, controls,
                        exclu_grps=[[]],
                        cat_expand=[],
                        save_path=None):
    df_r = _spec_curve_regression(df, y_endog, x_exog, controls,
                                  exclu_grps=exclu_grps,
                                  cat_expand=cat_expand)
    plot_spec_curve(df_r, x_exog, controls, save_path=save_path)
    return df_r


# Example data 1
def load_example_data2():
    n_samples = 1000
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


y_endog = 'y'
x_exog = 'x_1'
controls = ['x_2', 'x_3', 'x_4', 'x_5']
exclu_grps = [['x_4', 'x_5']]


def load_example_data():
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


x_exog = 'x2'
y_endog = 'y2'
ctrls = ['c1', 'c2', 'group1', 'group2']

df = load_example_data()
df_r = specification_curve(df, y_endog, x_exog, ctrls)
df_r = specification_curve(df, y_endog, x_exog, ctrls,
                           cat_expand=['group1'])
df_r = specification_curve(df, y_endog, x_exog, ctrls,
                           exclu_grps=[['c1', 'c2']])
df_r = specification_curve(df, y_endog, x_exog, ctrls,
                           cat_expand=['group1', 'group2'])
df_r = specification_curve(df, y_endog, x_exog, ctrls,
                           cat_expand=['group1', 'group2'],
                           exclu_grps=[['c1', 'c2']])
