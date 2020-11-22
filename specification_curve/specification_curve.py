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
    """
    Rounds numbers to 1 s.f.
    """
    return round(x, -int(floor(log10(abs(x)))) + 1)


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


def _remove_overlapping_vars(list_to_check, includes_list):
    """ Checks, and removes, any variable in list_to_check that is also in
    includes_list, returning list_to_check without overlapping variable
    names.
    """
    return [x for x in list_to_check if x not in includes_list]


def _pretty_plots():
    """
    Uses specification curve package's pretty plot style.
    Overrides existing style.
    """
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


def _flatn_list(nested_list):
    """ Flattens nested list """
    return list(itertools.chain.from_iterable(nested_list))


class SpecificationCurve():
    """
    Specification curve object

    Uses a model to perform all variants of a specification.
    Stores the results of those regressions in a tidy format pandas dataframe.
    Plots the regressions in chart that can optionally be saved.
    Will iterate over multiple inputs for exog. and endog. variables.
    Note that categorical variables that are expanded cannot be mutually
    excluded from other categorical variables that are expanded.

    """
    def __init__(self, df, y_endog, x_exog, controls, exclu_grps=[[]],
                 cat_expand=[], always_include=[]):
        """
        :list[string] y_endog: dependent variable(s)
        :list[string] x_exog: independent variable(s)
        :list[string] controls: variables to control for
        :list[list[string]] exclu_grps: each list should have controls
        that are mutually exclusive in.
        :list[string] cat_expand: categorical variables to test one at a time
        :list[string] always_include: regressors to always include
        """
        self.df = df.copy()
        self.y_endog = y_endog
        self.x_exog = x_exog
        self.controls = controls
        self.exclu_grps = exclu_grps
        self.cat_expand = cat_expand
        self.always_include = always_include

    def fit(self, estimator=sm.OLS):
        """
        Fits a specification curve.

        :statsmodels.regression.linear_model estimator: statsmodels estimator
        object. Default is OLS.
        """
        self.estimator = estimator
        self.controls = _single_list_check_str(self.controls)
        self.cat_expand = _single_list_check_str(self.cat_expand)
        self.exclu_grps = _double_list_check(self.exclu_grps)
        self.always_include = _single_list_check_str(self.always_include)
        self.y_endog = _single_list_check_str(self.y_endog)
        self.x_exog = _single_list_check_str(self.x_exog)
        # If any of always include in any other list, remove it from other list
        self.controls = _remove_overlapping_vars(self.controls,
                                                 self.always_include)
        self.x_exog = _remove_overlapping_vars(self.x_exog,
                                               self.always_include)
        self.ctrl_combs = self._compute_combinations()
        self.df_r = self._spec_curve_regression()
        print('Fit complete')

    def _reg_func(self, y_endog, x_exog, reg_vars):
        # NB: get dummies
        # transforms by default any col that is object or cat
        xf = pd.get_dummies(self.df, prefix_sep='=')
        new_cols = [x for x in xf.columns if x not in self.df.columns]
        gone_cols = [x for x in self.df.columns if x not in xf.columns]
        reg_vars_here = copy.copy(reg_vars)
        reg_vars_here.extend(new_cols)
        [reg_vars_here.remove(x) for x in gone_cols if x in reg_vars_here]
        return self.estimator(xf[y_endog], xf[[x_exog]+reg_vars_here]).fit()

    def _compute_combinations(self):
        """
        Finds all possible combinations of variables.
        Changes df to have dummies for cat expand columns.
        """
        init_cols = (self.y_endog + self.x_exog +
                     self.controls + self.always_include)
        self.df = self.df[init_cols]
        new_cols = []
        # Warning: hard-coded prefix
        if(self.cat_expand != []):
            self.df = pd.get_dummies(self.df, columns=self.cat_expand,
                                     prefix_sep=' = ')
            new_cols = [x for x in self.df.columns if x not in init_cols]
            # Now change the controls
            [self.controls.remove(x) for x in self.cat_expand]
            self.controls.extend(new_cols)
            # Create mapping from cat expand to new cols
            oldnew = dict(zip([x for x in self.cat_expand],
                              [[x for x in new_cols if y in x]
                               for y in self.cat_expand]))
            # Now exclude the groups that combine any new cols
            for x in self.cat_expand:
                if(self.exclu_grps == [[]]):
                    self.exclu_grps = [oldnew[x]]
                else:
                    self.exclu_grps.append(oldnew[x])
        # Find any subsets of excluded combs and add all variants
        for x in self.exclu_grps:
            if(len(x) > 2):
                sub_combs = [list(combinations(x, y))
                             for y in range(2, len(x))]
                sub_combs = [item for sublist in sub_combs for item in sublist]
                # Turn all the tuples into lists
                sub_combs = [list(x) for x in sub_combs]
                self.exclu_grps = self.exclu_grps + sub_combs
        # Turn mutually exclusive groups into sets
        if(self.exclu_grps != [[]]):
            self.exclu_grps = [set(x) for x in self.exclu_grps]
        # Get all combinations excluding mutually excl groups
        ctrl_combs = [_excl_combs(self.controls, k, self.exclu_grps)
                      for k in range(len(self.controls)+1)]
        # Flatten this into a single list of tuples
        ctrl_combs = _flatn_list(ctrl_combs)
        # Turn all the tuples into lists
        ctrl_combs = [list(x) for x in ctrl_combs]
        # Add in always included regressors
        ctrl_combs = [x + self.always_include for x in ctrl_combs]
        return ctrl_combs

    def _spec_curve_regression(self):
        """Performs all regressions for a specification curve.

        Estimates model with estimator as given in fitting function.
        Assumes that all controls and fixed effects should be varied.

        :returns: pandas dataframe of results
        """
        # Regressions - order of loop matters here
        reg_results = [[[self._reg_func(y, x, ctrl_vars)
                         for ctrl_vars in self.ctrl_combs]
                        for x in self.x_exog]
                       for y in self.y_endog]
        reg_results = _flatn_list(_flatn_list(reg_results))
        # Order matters here: x ends up as first var,
        # y as second
        combs = [[[[x] + [y] + ctrl_vars for ctrl_vars in self.ctrl_combs]
                  for x in self.x_exog]
                 for y in self.y_endog]
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
        df_r['Specification'] = df_r['Specification'].apply(
            lambda x: sorted(x))
        df_r['SpecificationCounts'] = df_r['Specification'].apply(
            lambda x: Counter(x))
        return df_r

    def plot(self, save_path=None, pretty_plots=True,
             preferred_spec=[]):
        """
        Makes plots of fitted specification curve.

        :string save_path: exported fig filename
        :bool pretty_plots: whether to use this package's figure formatting
        :list[str] preferred_spec: preferred specification
        """
        if(pretty_plots):
            _pretty_plots()
        # Set up blocks for showing what effects are included
        df_spec = self.df_r['SpecificationCounts'].apply(pd.Series).fillna(0.)
        df_spec = df_spec.replace(0., False).replace(1., True)
        df_spec = df_spec.T
        df_spec = df_spec.sort_index()
        # Label the preferred specification as True
        self.df_r['preferred'] = False
        if preferred_spec:
            self.df_r['preferred'] = self.df_r['Specification'].apply(
                lambda x: Counter(x) == Counter(preferred_spec))
        # This is quite hacky
        new_ctrl_names = list(
            set(_flatn_list(self.df_r['Specification'].values)))
        new_ctrl_names = [x for x in new_ctrl_names if
                          x not in self.x_exog + self.y_endog]
        new_ctrl_names.sort()
        name = self.x_exog + self.y_endog + new_ctrl_names
        group = (['x_exog' for x in range(len(self.x_exog))] +
                 ['y_endog' for y in range(len(self.y_endog))] +
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
        block_df['group_index'] = block_df['group'].apply(
            lambda x: index_dict[x])
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
        axarr[0].axhline(y=np.median(self.df_r['Coefficient']),
                         color='k',
                         lw=0.3,
                         alpha=1,
                         label='Median coefficient',
                         dashes=[12, 5])
        # Colour the significant ones differently
        self.df_r['color_coeff'] = 'black'
        self.df_r['coeff_pvals'] = (self
                                    .df_r
                                    .apply(lambda row:
                                           row['pvalues'][row['x_exog']],
                                           axis=1))
        self.df_r.loc[self.df_r['coeff_pvals'] < 0.05, 'color_coeff'] = 'blue'
        for color in self.df_r['color_coeff'].unique():
            slice_df_r = self.df_r.loc[self.df_r['color_coeff'] == color]
            markers, caps, bars = axarr[0].errorbar(slice_df_r.index,
                                                    slice_df_r['Coefficient'],
                                                    yerr=slice_df_r['bse'],
                                                    ls='none', color=color,
                                                    alpha=0.8, zorder=1,
                                                    elinewidth=2,
                                                    marker='o')
            [bar.set_alpha(0.4) for bar in bars]
            [cap.set_alpha(0.4) for cap in caps]
        # If there is a preferred spec, label it.
        if preferred_spec:
            loc_y = (self.df_r.loc[self.df_r['preferred'],
                                   'Coefficient'])
            loc_x = loc_y.index[0]
            loc_y = loc_y.values[0]
            cn_styl = "angle3,angleA=0,angleB=-90"
            axarr[0].annotate('Preferred specification',
                              xy=(loc_x, loc_y), xycoords='data',
                              xytext=(3, 100), textcoords='offset points',
                              fontsize='x-small',
                              arrowprops=dict(arrowstyle="fancy",
                                              fc="0.4", ec="none",
                                              connectionstyle=cn_styl))
        axarr[0].legend(frameon=True, loc='lower right',
                        ncol=1, handlelength=2)
        axarr[0].set_ylabel('Coefficient')
        axarr[0].set_title('Specification curve analysis')
        max_height = (self.df_r['Coefficient'] + self.df_r['bse']).max()
        min_height = (self.df_r['Coefficient'] - self.df_r['bse']).min()
        ylims = (min_height/1.2, 1.2*max_height)
        axarr[0].set_ylim(_round_to_1(ylims[0]),
                          _round_to_1(ylims[1]))
        # Now do the blocks - each group get its own array
        wid = 0.6
        hei = wid/3
        color_dict = {True: '#9B9B9B', False: '#F2F2F2'}
        if(len(self.df_r) > 160):
            wid = 0.01
            color_dict = {True: 'k', False: '#FFFFFF'}
        for ax_num, ax in enumerate(axarr[1:]):
            block_index = block_df.loc[block_df['group_index']
                                       == ax_num, :].index
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
