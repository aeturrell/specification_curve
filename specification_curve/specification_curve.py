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
                        'axes.labelsize': 18,
                        'lines.linewidth': 2,
                        'lines.markersize': 5,
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
    # Make sure exlu grps is a list of lists TODO
    df = xdf.copy()
    controls = copy.copy(controls)
    init_cols = [y_endog] + [x_exog] + controls
    df = df[init_cols]
    new_cols = []
    # Warning: hard-coded prefix
    if(cat_expand != []):
        df = pd.get_dummies(df, columns=cat_expand, prefix_sep=' = ')
        new_cols = [x for x in df.columns if x not in init_cols]
        # Now change the controls
        [controls.remove(x) for x in cat_expand]
        controls.extend(new_cols)
        # Create mapping from cat expand to new cols
        oldnew = dict(zip([x for x in cat_expand],
                          [[x for x in new_cols if y in x]
                           for y in cat_expand]))
        # Now exclude the groups that combine any new cols
        for x in cat_expand:
            if(exclu_grps == [[]]):
                exclu_grps = [oldnew[x]]
            else:
                exclu_grps.append(oldnew[x])
    # Find any subsets of excluded combs and add all variants
    for x in exclu_grps:
        if(len(x) > 2):
            sub_combs = [list(combinations(x, y)) for y in range(2, len(x))]
            sub_combs = [item for sublist in sub_combs for item in sublist]
            # Turn all the tuples into lists
            sub_combs = [list(x) for x in sub_combs]
            exclu_grps = exclu_grps + sub_combs

    # Turn mutually exclusive groups into sets
    if(exclu_grps != [[]]):
        exclu_grps = [set(x) for x in exclu_grps]
    # Get all combinations excluding mutually excl groups
    all_combs = [_excl_combs(controls, k, exclu_grps)
                 for k in range(len(controls)+1)]
    # Flatten this into a single list of tuples
    all_combs = [item for sublist in all_combs for item in sublist]
    # Turn all the tuples into lists
    all_combs = [list(x) for x in all_combs]
    # Regressions
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


def plot_spec_curve(df_r, x_exog, controls, save_path=None):
    """Plots a specification curve.



    :dataframe df_r: pandas dataframe with regression specification results
    :dataframe df_ctr_mat: pandas dataframe of inclusion and significance
    :list[string] controls: variables to control for
    :string save_path: File path to save figure to.

    """
    _pretty_plots()
    # Set up blocks for showing what effects are included
    df_spec = df_r['SpecificationCounts'].apply(pd.Series).fillna(0.)
    df_spec = df_spec.replace(0., False).replace(1., True)
    df_spec = df_spec.T
    df_spec = df_spec.sort_index()
    # Warning: hard-coded prefix
    block_ids = [x.split(' = ')[0] for x in list(df_spec.index.values)]
    block_df = pd.DataFrame(list(df_spec.index.values), index=block_ids,
                            columns=['name'])
    # The size of each block
    block_df['group_size'] = pd.Series(dict(Counter(block_ids)))
    block_df = block_df.reset_index().set_index('name')
    # Gives a unique index to each block
    block_df['group_index'] = pd.Categorical(block_df['group_size']).codes
    plt.close('all')
    fig = plt.figure(constrained_layout=False, figsize=(12, 8))
    heights = ([2] +
               [0.3*np.log(x+1)
                for x in block_df['group_size'].value_counts()][::-1])
    
    spec = fig.add_gridspec(ncols=1, nrows=len(heights),
                            height_ratios=heights, wspace=0.05)
    axarr = []
    for row in range(len(heights)):
        axarr.append(fig.add_subplot(spec[row, 0]))

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
    df_r['color_coeff'] = 'black'
    df_r['coeff_pvals'] = df_r['pvalues'].apply(lambda x: x[x_exog])
    df_r.loc[df_r['coeff_pvals'] < 0.05, 'color_coeff'] = 'blue'
    for color in df_r['color_coeff'].unique():
        slice_df_r = df_r.loc[df_r['color_coeff'] == color]
        markers, caps, bars = axarr[0].errorbar(slice_df_r.index,
                                                slice_df_r['Coefficient'],
                                                yerr=slice_df_r['bse'],
                                                ls='none', color=color,
                                                alpha=0.8, zorder=1,
                                                elinewidth=2,
                                                marker='o')
        [bar.set_alpha(0.4) for bar in bars]
        [cap.set_alpha(0.4) for cap in caps]
    # axarr[0].scatter(df_r.index, df_r['Coefficient'],
    #                  s=5.,
    #                  color='grey',
    #                  alpha=0.6,
    #                  marker='o',
    #                  label='Coefficient',
    #                  zorder=3)
    axarr[0].legend(frameon=True, loc='lower right',
                    ncol=1, handlelength=2)
    axarr[0].set_ylabel('Coefficient')
    axarr[0].set_title('Specification curve analysis')
    ylims = axarr[0].get_ylim()
    axarr[0].set_ylim(round_to_1(ylims[0]), round_to_1(ylims[1]))
    # Now do the blocks
    wid = 0.6
    hei = wid/3
    color_dict = {True: '#9B9B9B', False: '#F2F2F2'}
    for ax_num, ax in enumerate(axarr[1:]):
        block_index = block_df.loc[block_df['group_index'] == ax_num, :].index
        df_sp_sl = df_spec.loc[block_index, :].copy()
        for i, row_name in enumerate(df_sp_sl.index):
            for j, col_name in enumerate(df_sp_sl.columns):
                color = color_dict[df_sp_sl.iloc[i, j]]
                sq = patches.Rectangle(
                    (j-wid/2, i-hei/2), wid, hei, fill=True, color=color)
                ax.add_patch(sq)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_yticks(range(len(list(df_sp_sl.index.values))))
        ax.set_yticklabels(list(df_sp_sl.index.values))
        ax.set_xticklabels([])
        ax.set_ylim(-hei, len(df_sp_sl)-hei*2)
        ax.set_xlim(-wid, len(df_sp_sl.columns))
        for place in ['right', 'top', 'bottom']:
            ax.spines[place].set_visible(False)
    for ax in axarr:
        ax.set_xticks([], minor=True)
        ax.set_xticks([])
        ax.set_xlim(-wid, len(df_spec.columns))
    if(save_path is not None):
        plt.save(save_path)
    plt.show()


def spec_curve(df, y_endog, x_exog, controls,
               exclu_grps=[[]],
               cat_expand=[],
               save_path=None):
    df_r = _spec_curve_regression(df, y_endog, x_exog, controls,
                                  exclu_grps=exclu_grps,
                                  cat_expand=cat_expand)
    plot_spec_curve(df_r, x_exog, controls, save_path=save_path)
    return df_r
