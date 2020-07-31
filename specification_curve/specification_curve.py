"""
Specification Curve
-------------------
A package that produces specification curve analysis.
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
import copy
import itertools
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from math import log10, floor


def _round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))


def _double_list_check(XX):
    """
    If a list, return as list of lists
    """
    if(not(any(isinstance(el, list) for el in XX))):
        XX = [XX]
    return XX


def _single_list_check_str(X):
    """
    If type is string, return string in list
    """
    if(type(X) == str):
        X = [X]
    return X


def _pretty_plots():
    jsonPlotSettings = {'ytick.labelsize': 16,
                        'xtick.labelsize': 16,
                        'font.size': 22,
                        'figure.figsize': (10, 5),
                        'axes.titlesize': 22,
                        'axes.labelsize': 18,
                        'lines.linewidth': 2,
                        'lines.markersize': 3,
                        'legend.fontsize': 11,
                        'mathtext.fontset': 'stix',
                        'font.family': 'STIXGeneral'}
    plt.style.use(jsonPlotSettings)


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


def _flatn_list(nested_list):
    return list(itertools.chain.from_iterable(nested_list))


def _spec_curve_regression(xdf, y_endog, x_exog, controls,
                           exclu_grps=[[]],
                           cat_expand=[]):
    """Performs all regressions for a specification curve.

    Uses OLS to perform all control & variants of y = beta*x + (other factors).
    Assumes that all controls and fixed effects should be varied.

    :dataframe xdf: pandas dataframe in tidy format
    :list[string] y_endog: dependent variable(s)
    :list[string] x_exog: independent variable(s)
    :list[string] controls: variables to control for
    :list[list[string]] exclu_grps: each list should have controls
    that are mutually exclusive in.
    :list[string] cat_expand: categorical variables that are
    mutually exclusive

    :returns: pandas dataframe of results
    """
    df = xdf.copy()
    controls = copy.copy(controls)
    init_cols = y_endog + x_exog + controls
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
    ctrl_combs = [_excl_combs(controls, k, exclu_grps)
                  for k in range(len(controls)+1)]
    # Flatten this into a single list of tuples
    ctrl_combs = _flatn_list(ctrl_combs)
    # Turn all the tuples into lists
    ctrl_combs = [list(x) for x in ctrl_combs]
    # Regressions - order of loop matters here
    reg_results = [[[_reg_func(df, y, x, ctrl_vars)
                     for ctrl_vars in ctrl_combs]
                    for x in x_exog]
                   for y in y_endog]
    reg_results = _flatn_list(_flatn_list(reg_results))
    # Order matters here: x ends up as first var,
    # y as second
    combs = [[[[x] + [y] + ctrl_vars for ctrl_vars in ctrl_combs]
              for x in x_exog]
             for y in y_endog]
    combs = _flatn_list(_flatn_list(combs))
    df_r = pd.DataFrame(combs)
    df_r = df_r.rename(columns={0: 'x_exog', 1: 'y_endog'})
    df_r['Results'] = reg_results
    df_r['Coefficient'] = df_r.apply(
        lambda row: row['Results'].params[row['x_exog']], axis=1)
    df_r['Specification'] = combs
    df_r['bse'] = df_r.apply(
        lambda row: row['Results'].bse[row['x_exog']], axis=1)
    df_r['pvalues'] = [x.pvalues for x in reg_results]
    df_r['pvalues'] = df_r['pvalues'].apply(lambda x: dict(x))
    # Re-order by coefficient
    df_r = df_r.sort_values('Coefficient')
    cols_to_keep = ['Results', 'Coefficient',
                    'Specification', 'bse', 'pvalues',
                    'x_exog', 'y_endog']
    df_r = (df_r.drop(
        [x for x in df_r.columns if x not in cols_to_keep], axis=1))
    df_r = df_r.reset_index().drop('index', axis=1)
    df_r.index.names = ['Specification No.']
    df_r['Specification'] = df_r['Specification'].apply(lambda x: sorted(x))
    df_r['SpecificationCounts'] = df_r['Specification'].apply(
        lambda x: Counter(x))
    return df_r


def plot_spec_curve(df_r, y_endog, x_exog, save_path=None):
    """Plots a specification curve.


    :dataframe df_r: pandas dataframe with regression specification results
    :list[string] y_endog: dependent variable(s)
    :list[string] x_exog: independent variable(s)
    :string save_path: File path to save figure to.

    """
    _pretty_plots()
    # Set up blocks for showing what effects are included
    df_spec = df_r['SpecificationCounts'].apply(pd.Series).fillna(0.)
    df_spec = df_spec.replace(0., False).replace(1., True)
    df_spec = df_spec.T
    df_spec = df_spec.sort_index()
    # This is quite hacky
    new_ctrl_names = list(set(_flatn_list(df_r['Specification'].values)))
    new_ctrl_names = [x for x in new_ctrl_names if x not in x_exog+y_endog]
    new_ctrl_names.sort()
    name = x_exog + y_endog + new_ctrl_names
    group = (['x_exog' for x in range(len(x_exog))] +
             ['y_endog' for y in range(len(y_endog))] +
             [var.split(' = ')[0] for var in new_ctrl_names])
    block_df = pd.DataFrame(group, index=name, columns=['group'])
    group_map = dict(zip(block_df['group'], block_df['group']))
    group_map.update(dict(zip([x for x in group if x in new_ctrl_names],
                              ['control'
                               for x in group if x in new_ctrl_names])))
    block_df['group'] = block_df['group'].apply(lambda x: group_map[x])
    counts = (block_df.reset_index().groupby('group').count().rename(
        columns={'index': 'counts'}))
    block_df = pd.merge(block_df.reset_index(),
                        counts.reset_index(),
                        on=['group']).set_index('index')
    block_df = block_df.loc[block_df['counts'] > 1, :]
    index_dict = dict(zip(block_df['group'].unique(),
                          range(len(block_df['group'].unique()))))
    block_df['group_index'] = block_df['group'].apply(lambda x: index_dict[x])
    heights = ([2] +
               [0.3*np.log(x+1)
                for x in block_df['group_index'].value_counts(sort=False)])
    # Make the plot
    plt.close('all')
    fig = plt.figure(constrained_layout=False, figsize=(12, 8))
    spec = fig.add_gridspec(ncols=1, nrows=len(heights),
                            height_ratios=heights, wspace=0.05)
    axarr = []
    for row in range(len(heights)):
        axarr.append(fig.add_subplot(spec[row, 0]))
    # Do the coefficient first
    axarr[0].axhline(y=np.median(df_r['Coefficient']),
                     color='k',
                     lw=0.3,
                     alpha=1,
                     label='Median coefficient',
                     dashes=[12, 5])
    # Colour the significant ones differently
    df_r['color_coeff'] = 'black'
    df_r['coeff_pvals'] = (df_r.apply(lambda row:
                                      row['pvalues'][row['x_exog']],
                                      axis=1))
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
    axarr[0].legend(frameon=True, loc='lower right',
                    ncol=1, handlelength=2)
    axarr[0].set_ylabel('Coefficient')
    axarr[0].set_title('Specification curve analysis')
    ylims = axarr[0].get_ylim()
    axarr[0].set_ylim(_round_to_1(ylims[0]), _round_to_1(ylims[1]))
    # Now do the blocks - each group get its own array
    wid = 0.6
    hei = wid/3
    color_dict = {True: '#9B9B9B', False: '#F2F2F2'}
    if(len(df_r) > 160):
        wid = 0.01
        color_dict = {True: 'k', False: '#FFFFFF'}
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
        ax.set_ylim(-hei, len(df_sp_sl)-hei*4)
        ax.set_xlim(-wid, len(df_sp_sl.columns))
        for place in ['right', 'top', 'bottom']:
            ax.spines[place].set_visible(False)
    for ax in axarr:
        ax.set_xticks([], minor=True)
        ax.set_xticks([])
        ax.set_xlim(-wid, len(df_spec.columns))
    if(save_path is not None):
        plt.savefig(save_path, dpi=300)
    plt.show()


def spec_curve(df, y_endog, x_exog, controls,
               exclu_grps=[[]],
               cat_expand=[],
               save_path=None):
    """Creates a specification curve from given data

    Uses OLS to perform all variants of y = beta*x + (other factors).
    Stores the results of those regressions in a tidy format pandas dataframe.
    Plots the regressions in chart that can optionally be saved.
    Will iterate over multiple inputs for exog. and endog. variables.
    Note that categorical variables that are expanded cannot be mutually
    excluded from other categorical variables that are expanded.

    :dataframe df: pandas dataframe in tidy format
    :list[string] y_endog: dependent variable(s)
    :list[string] x_exog: independent variable(s)
    :list[string] controls: variables to control for
    :list[list[string]] exclu_grps: each list should have controls that are
    mutually exclusive in.
    :list[string] cat_expand: categorical variables that are mutually exclusive
    :string save_path: where to store output

    :returns: pandas dataframe of results

    Example
    -------
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
    """
    controls = _single_list_check_str(controls)
    cat_expand = _single_list_check_str(cat_expand)
    exclu_grps = _double_list_check(exclu_grps)
    y_endog = _single_list_check_str(y_endog)
    x_exog = _single_list_check_str(x_exog)
    df_r = _spec_curve_regression(df, y_endog, x_exog, controls,
                                  exclu_grps=exclu_grps,
                                  cat_expand=cat_expand)
    plot_spec_curve(df_r, y_endog, x_exog, save_path=save_path)
    return df_r
