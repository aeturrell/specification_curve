"""
Specification Curve
-------------------
A package that produces specification curve analysis.
"""
import copy
import itertools
import os
import typing
from collections import Counter
from collections import defaultdict
from itertools import combinations
from math import floor
from math import log10
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pkg_resources
import statsmodels.api as sm


EXAMPLE_FILE = pkg_resources.resource_filename(
    "specification_curve", os.path.join("data", "example_data.csv")
)


def _round_to_1(x: float) -> float:
    """Rounds numbers to 1 s.f.
    Args:
        x (float): input number
    Returns:
        float: number rounded
    """
    return round(x, -int(floor(log10(abs(x)))) + 1)


@typing.no_type_check
def _double_list_check(XX: Union[List[str], List[List[str]]]) -> List[List[str]]:
    """Ensures that input is returned as nested list.
    Args:
        XX (Union[list[str], list[list[str]]]): Input list of (list of) strings
    Returns:
        list[list[str]]: List of list of strings
    """
    if not (any(isinstance(el, list) for el in XX)):
        XX = [XX]
    return XX


@typing.no_type_check
def _single_list_check_str(X: Union[str, List[str]]) -> List[str]:
    """Ensures a list of strings.
    Args:
        X (Union[str, list[str]]): input string of list of strings
    Returns:
        list[str]: List of strings.
    """
    if type(X) == str:
        X = [X]
    return X


def _remove_overlapping_vars(
    list_to_check: List[str], includes_list: List[str]
) -> List[str]:
    """Removes any variable in list_to_check that is also in includes_list.
    Args:
        list_to_check (list[str]): _description_
        includes_listlist (List[str]): _description_
    Returns:
        list[str]: without overlapping variable names.
    """
    return [x for x in list_to_check if x not in includes_list]


def _pretty_plots() -> None:
    """Uses specification curve package's pretty plot style."""
    json_plot_settings = {
        "ytick.labelsize": 16,
        "xtick.labelsize": 16,
        "font.size": 22,
        "figure.figsize": (10, 5),
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "lines.linewidth": 2,
        "lines.markersize": 3,
        "legend.fontsize": 11,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    }
    plt.style.use(json_plot_settings)


def _excl_combs(lst, r, excludes):
    """From a given list of combinations, excludes those in `excludes`.
    Args:
        lst (list[str]): combinations
        r (int): combination integer
        excludes (list[str]): combinations to exclude
    Returns:
        list[str]: combinations with excluded combinations remove
    """
    if excludes != [[]]:
        return [
            comb
            for comb in combinations(lst, r)
            if not any(e.issubset(comb) for e in excludes)
        ]
    else:
        return list(combinations(lst, r))


def _flatn_list(nested_list: Union[str, List[str], List[List[str]]]) -> List[str]:
    """Flattens nested list.
    Args:
        nested_list (Union[List[str], List[List[str]]]): nested list
    Returns:
        List[str]: flattened list
    """
    return list(itertools.chain.from_iterable(nested_list))


class SpecificationCurve:
    """Specification curve object.
    Uses a model to perform all variants of a specification.
    Stores the results of those regressions in a tidy format pandas dataframe.
    Plots the regressions in chart that can optionally be saved.
    Will iterate over multiple inputs for exog. and endog. variables.
    Note that categorical variables that are expanded cannot be mutually
    excluded from other categorical variables that are expanded.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y_endog: Union[str, List[str]],
        x_exog: Union[str, List[str]],
        controls: List[str],
        exclu_grps: List[List[None]] = [[]],
        cat_expand: Union[str, List[None]] = [],
        always_include: List[str] = [],
    ) -> None:
        """Specification curve object constructor.
        Args:
            df (pd.DataFrame): Data for regressions
            y_endog (Union[str, List[str]]): Endogeneous variables
            x_exog (Union[str, List[str]]): Exogeneous variables
            controls (List[str]): Conditioning variables
            exclu_grps (List[List[None]], optional): Combinations to exclude. Defaults to [[]].
            cat_expand (List[None], optional): Fixed effects whose categories to run separately. Defaults to [].
            always_include (List[str], optional): Any controls to always include. Defaults to [].
        """
        self.df = df.copy()
        self.y_endog = _single_list_check_str(y_endog)
        self.x_exog = _single_list_check_str(x_exog)
        self.controls = _single_list_check_str(controls)
        self.exclu_grps = _double_list_check(exclu_grps)
        self.cat_expand = _single_list_check_str(cat_expand)
        self.always_include = _single_list_check_str(always_include)

    def fit(self, estimator=sm.OLS) -> None:
        """Fits a specification curve by performing regressions.
        Args:
            estimator (statsmodels.regression.linear_model or statsmodels.discrete.discrete_model, optional): statsmodels estimator. Defaults to sm.OLS.
        """
        self.estimator = estimator
        # self.cat_expand = _single_list_check_str(self.cat_expand)
        # self.exclu_grps = _double_list_check(self.exclu_grps)
        # self.always_include = _single_list_check_str(self.always_include)
        # If any of always include in any other list, remove it from other list
        self.controls = _remove_overlapping_vars(self.controls, self.always_include)
        self.x_exog = _remove_overlapping_vars(self.x_exog, self.always_include)
        self.ctrl_combs = self._compute_combinations()
        self.df_r = self._spec_curve_regression()
        print("Fit complete")

    def _reg_func(
        self, y_endog: List[str], x_exog: str, reg_vars: List[str]
    ) -> sm.regression.linear_model.RegressionResults:
        """Performs the regression.
        Args:
            y_endog (List[str]): Endogeneous variables
            x_exog (str): Exogeneous variables
            reg_vars (List[str]): Controls
        Returns:
            sm.regression.linear_model.RegressionResults: coefficients from reg
        """
        # NB: get dummies
        # transforms by default any col that is object or cat
        xf = pd.get_dummies(self.df, prefix_sep="=")
        new_cols = [str(x) for x in xf.columns if x not in self.df.columns]
        gone_cols = [str(x) for x in self.df.columns if x not in xf.columns]
        reg_vars_here = copy.copy(reg_vars)
        reg_vars_here.extend(new_cols)
        # mypy prefers this formulation to a list comprehension
        for x in gone_cols:
            if x in reg_vars_here:
                reg_vars_here.remove(x)
        # Ensure new cols are int so that statsmodels will run on them.
        # This is because statsmodels requires all values to be of either int or float dtype.
        # first get columns series with true or false depending on if not int or float stem to data type
        non_int_or_float_cols = (
            ~xf[reg_vars_here]
            .dtypes.astype("string")
            .str.split("[1-9][0-9]", regex=True)
            .str[0]
            .isin(["int", "float"])
        )
        # now take only the trues from
        cols_to_convert_to_int = list(
            non_int_or_float_cols[non_int_or_float_cols].index
        )
        # convert just these columns to int
        for col_name in cols_to_convert_to_int:
            xf[col_name] = xf[col_name].astype(int)
        return self.estimator(xf[y_endog], xf[[x_exog] + reg_vars_here]).fit()

    def _compute_combinations(self):
        """
        Finds all possible combinations of variables.
        Changes df to have dummies for cat expand columns.
        """
        init_cols = self.y_endog + self.x_exog + self.controls + self.always_include
        self.df = self.df[init_cols]
        new_cols = []
        # Warning: hard-coded prefix
        if self.cat_expand != []:
            self.df = pd.get_dummies(self.df, columns=self.cat_expand, prefix_sep=" = ")
            new_cols = [x for x in self.df.columns if x not in init_cols]
            # Now change the controls
            [self.controls.remove(x) for x in self.cat_expand]
            self.controls.extend(new_cols)
            # Create mapping from cat expand to new cols
            oldnew = dict(
                zip(
                    [x for x in self.cat_expand],
                    [[x for x in new_cols if y in x] for y in self.cat_expand],
                )
            )
            # Now exclude the groups that combine any new cols
            for x in self.cat_expand:
                if self.exclu_grps == [[]]:
                    self.exclu_grps = [oldnew[x]]
                else:
                    self.exclu_grps.append(oldnew[x])
        # Find any subsets of excluded combs and add all variants
        for x in self.exclu_grps:
            if len(x) > 2:
                sub_combs = [list(combinations(x, y)) for y in range(2, len(x))]
                sub_combs = [item for sublist in sub_combs for item in sublist]
                # Turn all the tuples into lists
                sub_combs = [list(x) for x in sub_combs]
                self.exclu_grps = self.exclu_grps + sub_combs
        # Turn mutually exclusive groups into sets
        if self.exclu_grps != [[]]:
            self.exclu_grps = [set(x) for x in self.exclu_grps]
        # Get all combinations excluding mutually excl groups
        ctrl_combs = [
            _excl_combs(self.controls, k, self.exclu_grps)
            for k in range(len(self.controls) + 1)
        ]
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
        reg_results = [
            [
                [self._reg_func(y, x, ctrl_vars) for ctrl_vars in self.ctrl_combs]
                for x in self.x_exog
            ]
            for y in self.y_endog
        ]
        reg_results = _flatn_list(_flatn_list(reg_results))
        # Order matters here: x ends up as first var,
        # y as second
        combs = [
            [
                [[x] + [y] + ctrl_vars for ctrl_vars in self.ctrl_combs]
                for x in self.x_exog
            ]
            for y in self.y_endog
        ]
        combs = _flatn_list(_flatn_list(combs))
        df_r = pd.DataFrame(combs)
        df_r = df_r.rename(columns={0: "x_exog", 1: "y_endog"})
        df_r["Results"] = reg_results
        df_r["Coefficient"] = df_r.apply(
            lambda row: row["Results"].params[row["x_exog"]], axis=1
        )
        df_r["Specification"] = combs
        df_r["bse"] = df_r.apply(lambda row: row["Results"].bse[row["x_exog"]], axis=1)
        # Pull out the confidence interval around the exogeneous variable
        df_r["conf_int"] = df_r.apply(
            lambda row: np.array(row["Results"].conf_int().loc[row["x_exog"]]), axis=1
        )
        df_r["pvalues"] = [x.pvalues for x in reg_results]
        df_r["pvalues"] = df_r["pvalues"].apply(lambda x: dict(x))
        # Re-order by coefficient: makes plots look more continuous
        df_r = df_r.sort_values("Coefficient")
        cols_to_keep = [
            "Results",
            "Coefficient",
            "Specification",
            "bse",
            "pvalues",
            "conf_int",
            "x_exog",
            "y_endog",
        ]
        df_r = df_r.drop([x for x in df_r.columns if x not in cols_to_keep], axis=1)
        df_r = df_r.reset_index().drop("index", axis=1)
        df_r.index.names = ["Specification No."]
        df_r["Specification"] = df_r["Specification"].apply(lambda x: sorted(x))
        df_r["SpecificationCounts"] = df_r["Specification"].apply(lambda x: Counter(x))
        return df_r

    def plot(
        self, save_path=None, pretty_plots: bool = True, preferred_spec: List[None] = []
    ) -> None:
        """Makes plots of fitted specification curve.
        Args:
            save_path (_type_, optional): Exported fig filename. Defaults to None.
            pretty_plots (bool, optional): whether to use this package's figure formatting. Defaults to True.
            preferred_spec (list, optional): preferred specification. Defaults to [].
        """
        if pretty_plots:
            _pretty_plots()
        # Set up blocks for showing what effects are included
        df_spec = self.df_r["SpecificationCounts"].apply(pd.Series).fillna(0.0)
        df_spec = df_spec.replace(0.0, False).replace(1.0, True)
        df_spec = df_spec.T
        df_spec = df_spec.sort_index()
        # Label the preferred specification as True
        self.df_r["preferred"] = False
        if preferred_spec:
            self.df_r["preferred"] = self.df_r["Specification"].apply(
                lambda x: Counter(x) == Counter(preferred_spec)
            )
        # This is quite hacky
        new_ctrl_names = list(set(_flatn_list(self.df_r["Specification"].values)))
        new_ctrl_names = [
            x for x in new_ctrl_names if x not in self.x_exog + self.y_endog
        ]
        new_ctrl_names.sort()
        name = self.x_exog + self.y_endog + new_ctrl_names
        group = (
            ["x_exog" for x in range(len(self.x_exog))]
            + ["y_endog" for y in range(len(self.y_endog))]
            + [var.split(" = ")[0] for var in new_ctrl_names]
        )
        block_df = pd.DataFrame(group, index=name, columns=["group"])
        group_map = dict(zip(block_df["group"], block_df["group"]))
        group_map.update(
            dict(
                zip(
                    [x for x in group if x in new_ctrl_names],
                    ["control" for x in group if x in new_ctrl_names],
                )
            )
        )
        block_df["group"] = block_df["group"].apply(lambda x: group_map[x])
        counts = (
            block_df.reset_index()
            .groupby("group")
            .count()
            .rename(columns={"index": "counts"})
        )
        block_df = pd.merge(
            block_df.reset_index(), counts.reset_index(), on=["group"]
        ).set_index("index")
        block_df = block_df.loc[block_df["counts"] > 1, :]
        index_dict = dict(
            zip(block_df["group"].unique(), range(len(block_df["group"].unique())))
        )
        block_df["group_index"] = block_df["group"].apply(lambda x: index_dict[x])
        heights = [2] + [
            0.3 * np.log(x + 1)
            for x in block_df["group_index"].value_counts(sort=False)
        ]
        # Make the plot
        plt.close("all")
        fig = plt.figure(constrained_layout=False, figsize=(12, 8))
        spec = fig.add_gridspec(
            ncols=1, nrows=len(heights), height_ratios=heights, wspace=0.05
        )
        axarr = []
        for row in range(len(heights)):
            axarr.append(fig.add_subplot(spec[row, 0]))
        # Add a line showing where the median coefficient is
        axarr[0].axhline(
            y=np.median(self.df_r["Coefficient"]),
            color="k",
            lw=0.3,
            alpha=1,
            label="Median coefficient",
            dashes=[12, 5],
        )
        # Annotate the median coefficient line with text
        axarr[0].text(
            x=0.1,
            y=np.median(self.df_r["Coefficient"]) * 1.02,
            s="Median coefficient",
            fontsize=12,
            color="gray",
            zorder=5,
        )
        # Colour the significant estimate values differently
        self.df_r["color_coeff"] = "black"
        self.df_r["coeff_pvals"] = self.df_r.apply(
            lambda row: row["pvalues"][row["x_exog"]], axis=1
        )
        self.df_r.loc[self.df_r["coeff_pvals"] < 0.05, "color_coeff"] = "blue"
        red_condition = (self.df_r["Coefficient"] < 0) & (
            self.df_r["coeff_pvals"] < 0.05
        )
        self.df_r.loc[red_condition, "color_coeff"] = "red"
        for color in self.df_r["color_coeff"].unique():
            slice_df_r = self.df_r.loc[self.df_r["color_coeff"] == color]
            a = (
                slice_df_r["Coefficient"]
                - np.stack(slice_df_r["conf_int"].to_numpy())[:, 0]
            )
            b = (
                np.stack(slice_df_r["conf_int"].to_numpy())[:, 1]
                - slice_df_r["Coefficient"]
            )
            y_err_correct_shape = np.stack((a, b))
            markers, caps, bars = axarr[0].errorbar(
                slice_df_r.index,
                slice_df_r["Coefficient"],
                yerr=y_err_correct_shape,
                ls="none",
                color=color,
                alpha=0.8,
                zorder=1,
                elinewidth=2,
                marker="o",
            )
            [bar.set_alpha(0.4) for bar in bars]
            [cap.set_alpha(0.4) for cap in caps]
        # If there is a preferred spec, label it.
        if preferred_spec:
            loc_y = self.df_r.loc[self.df_r["preferred"], "Coefficient"]
            loc_x = loc_y.index[0]
            loc_y = loc_y.values[0]
            cn_styl = "angle3,angleA=0,angleB=-90"
            axarr[0].annotate(
                f"Preferred specification: {loc_y:+.2f}",
                xy=(loc_x, loc_y),
                xycoords="data",
                xytext=(-30, 60),
                textcoords="offset points",
                fontsize=10,
                arrowprops=dict(
                    arrowstyle="fancy", fc="0.4", ec="none", connectionstyle=cn_styl
                ),
            )
        axarr[0].set_ylabel("Coefficient")
        axarr[0].set_title("Specification curve analysis")
        max_height = self.df_r["conf_int"].apply(lambda x: x.max()).max()
        min_height = self.df_r["conf_int"].apply(lambda x: x.min()).min()
        ylims = (min_height / 1.2, 1.2 * max_height)
        axarr[0].set_ylim(_round_to_1(ylims[0]), _round_to_1(ylims[1]))
        # Now do the blocks - each group get its own array
        wid = 0.5
        hei = wid / 2.5
        color_dict = {True: "#9B9B9B", False: "#F2F2F2"}
        if len(self.df_r) > 160:
            wid = 0.01
            color_dict = {True: "k", False: "#FFFFFF"}
        # Define group names to put on RHS of plot

        def return_string():
            """Convenience function to setup default dict.
            Returns:
                str: "subset"
            """
            return "subset"

        block_name_dict: DefaultDict[str, str] = defaultdict(return_string)
        block_name_dict.update({"x_exog": "x", "y_endog": "y", "control": "controls"})
        for ax_num, ax in enumerate(axarr[1:]):
            block_index = block_df.loc[block_df["group_index"] == ax_num, :].index
            df_sp_sl = df_spec.loc[block_index, :].copy()
            for i, row_name in enumerate(df_sp_sl.index):
                for j, col_name in enumerate(df_sp_sl.columns):
                    color = color_dict[df_sp_sl.iloc[i, j]]
                    sq = patches.Rectangle(
                        (j - wid / 2, i - hei / 2), wid, hei, fill=True, color=color
                    )
                    ax.add_patch(sq)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.set_yticks(range(len(list(df_sp_sl.index.values))))
            # Add text on the RHS that describes what each block is
            ax.text(
                x=len(df_sp_sl.columns) + 1,
                y=np.mean(ax.get_yticks()),
                s=block_name_dict[
                    block_df.loc[block_df["group_index"] == ax_num, "group"].iloc[0]
                ],
                rotation=-90,
                fontsize=11,
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax.set_yticklabels(list(df_sp_sl.index.values), fontsize=12)
            ax.set_xticklabels([])
            ax.set_ylim(-hei, len(df_sp_sl) - hei * 4)
            ax.set_xlim(-wid, len(df_sp_sl.columns))
            for place in ["right", "top", "bottom"]:
                ax.spines[place].set_visible(False)
        for ax in axarr:
            ax.set_xticks([], minor=True)
            ax.set_xticks([])
            ax.set_xlim(-wid, len(df_spec.columns))
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        plt.show()


def load_example_data1() -> pd.DataFrame:
    """Retrieves example data from a file included with the package.
    Returns:
        pd.DataFrame: Example data suitable for regression.
    """
    # Example data
    df = pd.read_csv(EXAMPLE_FILE, index_col=0)
    num_cols = [x for x in df.columns if x not in ["group1", "group2"]]
    for col in num_cols:
        df[col] = df[col].astype(np.double)
    cat_cols = [x for x in df.columns if x not in num_cols]
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df


def load_example_data2() -> pd.DataFrame:
    """Generates fake data.
    Returns:
        pd.DataFrame: Example data suitable for regression.
    """
    # Set seed for random numbers
    seed_for_prng = 78557
    # prng=probabilistic random number generator
    prng = np.random.default_rng(seed_for_prng)
    n_samples = 500
    x_1 = prng.random(size=n_samples)
    x_2 = prng.random(size=n_samples)
    x_3 = prng.random(size=n_samples)
    x_4 = prng.integers(2, size=n_samples)
    y = (
        0.8 * x_1
        + 0.1 * x_2
        + 0.5 * x_3
        + x_4 * 0.6
        + +2 * prng.standard_normal(n_samples)
    )
    df = pd.DataFrame([x_1, x_2, x_3, x_4, y], ["x_1", "x_2", "x_3", "x_4", "y"]).T
    # Set x_4 as a categorical variable
    df["x_4"] = df["x_4"].astype("category")
    return df
