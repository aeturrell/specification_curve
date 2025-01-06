"""
Specification Curve
-------------------
A package that produces specification curve analysis.
"""

import copy
import itertools
import os
from collections import Counter, defaultdict
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from math import floor, log10
from typing import DefaultDict, List, Optional, Union

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
from matplotlib.ticker import AutoMinorLocator
from tqdm.auto import trange
from typeguard import typeguard_ignore

try:
    __version__ = version("specification_curve")
except PackageNotFoundError:
    __version__ = "unknown"


# Set seed for random numbers
seed_for_prng = 78557
# prng=probabilistic random number generator
prng = np.random.default_rng(seed_for_prng)


def _round_to_2(x: float) -> float:
    """Rounds numbers to 2 s.f.
    Args:
        x (float): input number
    Returns:
        float: number rounded
    """
    return round(x, -int(floor(log10(abs(x)))) + 2)


def _remove_overlapping_vars(
    list_to_check: List[str], includes_list: List[str]
) -> List[str]:
    """Removes any variable in list_to_check that is also in includes_list.
    Args:
        list_to_check (list[str]): List that we wish to be sure doesn't have entries that are also in includes_list
        includes_list (List[str]):
    Returns:
        list[str]: list_to_check without overlapping variable names.
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


def _undummify(df: pd.DataFrame, prefix_sep: str = " = ") -> pd.DataFrame:
    """Reverts the pd.get_dummies() operation. Useful for reverting 'cat_expand'.

    Args:
        df (pd.DataFrame): Dataframe with separate columns as dummies
        prefix_sep (str, optional): String used to expand, eg "fe" with values 0 or 1 becoming "fe = 0", "fe = 1". Defaults to " = ".

    Returns:
        pd.DataFrame: Input data frame with dummies re-packed into single column.
    """
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df


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


@typeguard_ignore
def _flatn_list(nested_list: Union[str, List[str], List[List[str]]]) -> List[str]:
    """Flattens nested list.
    Args:
        nested_list
    Returns:
        List[str]: flattened list
    """
    return list(itertools.chain.from_iterable(nested_list))


def _parse_formula(formula_string: str) -> dict[str, List[str]]:
    """
    Parse a formula string of the format "y | y1 ~ x | x1 + c + c2 | c3"
    into separate lists of variables.

    Parameters:
    formula_string (str): The formula string to parse

    Returns:
    dict: Dictionary containing lists of variables categorized as:
          - endog: dependent variables (before ~)
          - exog: independent variables (after ~ and before first +)
          - always_include: standalone variables (no | symbol)
          - controls: variables that appear after | symbols
    """
    # Initialize result lists
    result: dict[str, List[str]] = {
        "x_exog": [],
        "y_endog": [],
        "always_include": [],
        "controls": [],
    }

    # Split into left and right sides of the tilde
    left_side, right_side = formula_string.split("~")

    # Process endog variables (left side)
    left_vars = left_side.strip().split("|")
    result["y_endog"].extend(var.strip() for var in left_vars if var.strip())

    # Split right side into components by plus sign
    right_components = right_side.strip().split("+")

    # Process the first component (exog variables)
    endog_vars = right_components[0].strip().split("|")
    result["x_exog"].extend(var.strip() for var in endog_vars if var.strip())

    # Process remaining components
    for component in right_components[1:]:
        component = component.strip()
        if "|" in component:
            # If component contains |, split and add all parts to controls
            vars_in_component = component.split("|")
            result["controls"].extend(
                var.strip() for var in vars_in_component if var.strip()
            )
        else:
            # If component is standalone (no |), add to always_include
            result["always_include"].append(component)

    return result


class SpecificationCurve:
    """Specification curve object.
    Uses a model to perform all variants of a specification.
    Stores the results of those regressions in a tidy format pandas dataframe.
    Plots the regressions in chart that can optionally be saved.
    Will iterate over multiple inputs for exog. and endog. variables.
    Note that categorical variables that are expanded cannot be mutually
    excluded from other categorical variables that are expanded.

    The class can be initialized in two mutually exclusive ways:
    1. Using a formula string (e.g., "y ~ x1 + x2")
    2. Using separate y_endog, x_exog, controls, and always_include parameters

    Args:
        df (pd.DataFrame): Input DataFrame
        formula (str, optional): R-style formula string (e.g., "y ~ x1 + x2")
        y_endog (Union[str, List[str]], optional): Dependent variable(s)
        x_exog (Union[str, List[str]], optional): Independent variable(s)
        controls (List[str], optional): Control variables
        exclu_grps (Union[List[List[None]], List[str], str, List[List[str]]], optional):
            Groups of variables to exclude. Defaults to [[None]]
        cat_expand (Union[str, List[None], List[str], List[List[str]]], optional):
            Categorical variables to expand. Defaults to []
        always_include (Union[str, List[str]], optional): Variables to always include

    Raises:
        ValueError: If neither formula nor (y_endog, x_exog, controls) are provided
        ValueError: If both formula and (y_endog, x_exog, controls) are provided
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y_endog: Optional[Union[str, List[str]]] = None,
        x_exog: Optional[Union[str, List[str]]] = None,
        controls: Optional[List[str]] = None,
        exclu_grps: Optional[
            Union[List[List[None]], List[str], str, List[List[str]]]
        ] = [[None]],
        cat_expand: Optional[Union[str, List[None], List[str], List[List[str]]]] = [],
        always_include: Optional[Union[str, List[str]]] = None,
        formula: Optional[str] = None,
    ) -> None:
        if formula is not None:
            if any(
                param is not None
                for param in [y_endog, x_exog, controls, always_include]
            ):
                raise ValueError(
                    "Cannot provide both formula and individual components "
                    "(y_endog, x_exog, controls, always_include)"
                )
            self.formula = formula
            spec_via_eqn = _parse_formula(formula)
            self.controls = spec_via_eqn["controls"]
            self.always_include = spec_via_eqn["always_include"]
            self.y_endog = spec_via_eqn["y_endog"]
            self.x_exog = spec_via_eqn["x_exog"]
        else:
            if any(param is None for param in [y_endog, x_exog, controls]):
                raise ValueError(
                    "Must provide either formula or all of: y_endog, x_exog, controls"
                )
            # Use the provided components directly
            self.y_endog = y_endog if isinstance(y_endog, list) else [str(y_endog)]
            self.x_exog = x_exog if isinstance(x_exog, list) else [str(x_exog)]
            self.controls = controls if isinstance(controls, list) else [str(controls)]
            self.always_include = (
                always_include
                if isinstance(always_include, list)
                else [str(always_include)]
                if always_include is not None
                else []
            )
            self.formula = None  # type: ignore

        self.df = df
        self.exclu_grps = exclu_grps
        self.cat_expand = cat_expand
        self.orig_cat_expand = cat_expand
        self.orig_controls = self.controls.copy()  # type: ignore
        self.orig_exclu_grps = self.exclu_grps.copy()  # type: ignore
        self.init_cols = (
            self.y_endog + self.x_exog + self.controls + self.always_include
        ).copy()

        # Raise error if always include and category expand have any of the same columns.
        if (self.cat_expand is not None) and (self.always_include is not None):
            if [x for x in self.always_include if x in self.cat_expand]:
                raise ValueError(
                    "Cannot always include a variable that is being expanded."
                )

    def __repr__(self) -> str:
        """Represents a Specification Curve object as a string in the terminal.
        Gives summary statistics if .fit() or .fit_null() have been run.

        Returns:
            str: Summary message.
        """
        message = "--------------------------\nSpecification Curve object\n--------------------------\n"
        if self.formula is not None:
            message = message + f"{self.formula=}\n" + "\n"
        else:
            message = (
                message
                + "y vars: "
                + ", ".join([x for x in self.y_endog])
                + "\n"
                + "x vars: "
                + ", ".join([x for x in self.x_exog])
                + "\n"
                + "controls: "
                + ", ".join([x for x in self.controls])
                + "\n"
                + "always included: "
                + " ".join([x for x in self.always_include])
                + "\n"
                + "\n"
            )
        if hasattr(self, "estimator"):
            # Model has been fit if estimator exists
            message = (
                message
                + "Specifications have run\n-----------------------\n"
                + "Estimator: "
                + repr(sm.OLS)
                + "\n"
                + "No. specifications: "
                + str(len(self.df_r))
                + "\n"
                + "\nCoefficient stats:\n"
                + pd.DataFrame(
                    self.df_r["Coefficient"]
                    .agg(["min", "median", "max"])
                    .apply(lambda x: _round_to_2(x))
                )
                .rename(columns={"Coefficient": ""})
                .T.to_string()
                + "\n\n"
            )
        if hasattr(self, "null_df"):
            message = (
                message
                + "Coeffs under null have run\n--------------------------\n"
                + self.null_stats_summary.to_string()
            )
        return message

    def fit(self, estimator=sm.OLS) -> None:
        """Fits a specification curve by performing regressions.

        Args:
            estimator (statsmodels.regression.linear_model or statsmodels.discrete.discrete_model, optional): statsmodels estimator. Defaults to sm.OLS.

        """
        self.estimator = estimator
        # If any of always include in any other list, remove it from other list
        self.controls = _remove_overlapping_vars(self.controls, self.always_include)
        self.x_exog = _remove_overlapping_vars(self.x_exog, self.always_include)
        # Now work out all specifications
        self.ctrl_combs = self._compute_combinations()
        self.df_r = self._spec_curve_regression()

    def _reg_func(
        self, y_endog: Union[str, List[str]], x_exog: str, reg_vars: List[str]
    ) -> Union[
        sm.regression.linear_model.RegressionResults,
        sm.regression.linear_model.RegressionResultsWrapper,
    ]:
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
        new_cols_from_cat_expand = [
            str(x) for x in xf.columns if x not in self.df.columns
        ]
        gone_cols = [str(x) for x in self.df.columns if x not in xf.columns]
        reg_vars_here = copy.copy(reg_vars)
        reg_vars_here.extend(new_cols_from_cat_expand)
        # mypy prefers this formulation to a list comprehension
        for x in gone_cols:
            if x in reg_vars_here:
                reg_vars_here.remove(x)
        # Ensure new cols are int so that statsmodels will run on them.
        # This is because statsmodels requires all values to be of either int or float dtype.
        # first get columns series with true or false depending on if not int or float stem to data type
        non_int_or_float_cols = ~xf[reg_vars_here].dtypes.astype("string").str.split(
            "[1-9][0-9]", regex=True
        ).str[0].isin(["int", "float"])
        # now take only the trues from
        cols_to_convert_to_int = list(
            non_int_or_float_cols[non_int_or_float_cols].index
        )
        # convert just these columns to int
        for col_name in cols_to_convert_to_int:
            xf[col_name] = xf[col_name].astype(int)
        xf_with_constant = sm.tools.tools.add_constant(xf[[x_exog] + reg_vars_here])
        return self.estimator(xf[y_endog], xf_with_constant).fit()

    def _compute_combinations(self):
        """
        Finds all possible combinations of variables.
        Changes df to have dummies for cat expand columns.
        """
        init_cols = self.y_endog + self.x_exog + self.controls + self.always_include
        # Only keep relevant columns
        self.df = self.df[init_cols]
        new_cols_from_cat_expand = []
        # Expand any variables in category expand into full dummies so that they can be subsets.
        if self.cat_expand != []:
            cat_expand_local = (
                self.cat_expand
                if isinstance(self.cat_expand, list)
                else [str(self.cat_expand)]
            )
            self.df = pd.get_dummies(
                self.df, columns=cat_expand_local, prefix_sep=" = "
            )
            new_cols_from_cat_expand = [
                x for x in self.df.columns if x not in init_cols
            ]
            # Now change the controls: remove the cat_expand columns
            [self.controls.remove(x) for x in cat_expand_local if x in self.controls]
            # Add the dummy versions of the same columns
            self.controls.extend(new_cols_from_cat_expand)
            # Create mapping from cat expand to their new dummy new cols
            dict_cat_expand_to_dummies = dict(
                zip(
                    [x for x in self.cat_expand],
                    [
                        [x for x in new_cols_from_cat_expand if y in x]
                        for y in self.cat_expand
                    ],
                )
            )
            # Now stop the new dummies from being part of specifications together
            # as they are meant to be mutually exclusive subsets.
            for x in self.cat_expand:
                if self.exclu_grps == [[]]:
                    self.exclu_grps = [dict_cat_expand_to_dummies[x]]
                else:
                    self.exclu_grps.append(dict_cat_expand_to_dummies[x])
        # Find any subsets of excluded combs and add all variants thereof
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

    def _spec_curve_regression(self) -> pd.DataFrame:
        """Runs all of the regressions on the available combinations.

        Returns:
            pd.DataFrame: DataFrame of specifications and their results.
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

    def fit_null(self, n_boot: int = 30, f_sample: float = 0.1) -> None:
        """Refits all of the specifications under the null of $y_{i(k)}^* = y_{i(k)} - b_k*x_{i(k)}$
        where i is over rows and k is over specifications and i is a function of k as y and x rows
        can change depending on the k when there are multiple y_endog and multiple x_exog.
        Each bootstrap sees a fraction of rows f_sample taken and then a new specification curve fit under the null.
        Then summary statistics are created by specification, and statistical tests.

        Args:
            n_boot (int, optional): Number of bootstraps. Defaults to 30.
            f_sample (float, optional): Fraction of rows to sample in each bootstrap. Defaults to 0.1.

        Raises:
            ValueError: If .fit() has not been run first.

        Returns:
            None: Results saved in self.null_stats_summary.

        """
        if hasattr(self, "df_r"):
            # construct the null, y_(i(k))* = y_(i(k)) - b_k*x_(i(k))
            y_star = (
                self.df[self.df_r["y_endog"]]
                - self.df_r["Coefficient"].values * self.df[self.df_r["x_exog"]].values
            )
            y_star.columns = self.df_r.index

            all_boot_df = pd.DataFrame()

            for i in trange(
                n_boot, desc=f"Bootstraps ({y_star.shape[1]} specifications)"
            ):
                # Choose num_selections rows from the len(df["y"]) possible rows of data
                # Dim.: num_selections * num_specifications
                # which rows do we want for this bootstrap?
                num_rows = int(floor(f_sample * y_star.shape[0]))
                index_of_rows = prng.integers(
                    low=0, high=y_star.shape[0], size=num_rows
                )
                y_star_chosen_rows = y_star.loc[index_of_rows, :]

                df_spec_coef_results = pd.DataFrame()
                for y_star_k in range(
                    y_star_chosen_rows.shape[1]
                ):  # iterating over specifications
                    df_new = self.df.iloc[index_of_rows, :].copy()  # selects rows
                    df_new["y_star"] = y_star_chosen_rows.loc[
                        :, y_star_k
                    ]  # goes one spec at a time
                    cat_expanded_columns = [
                        x for x in self.df.columns if x not in self.init_cols
                    ]
                    for cat_expand_col in self.orig_cat_expand:  # type: ignore
                        lst = [x.split(" = ")[0] for x in self.df.columns]
                        indices = [
                            icat
                            for icat, xcat in enumerate(lst)
                            if xcat == cat_expand_col
                        ]
                        cols_to_pick_out = self.df.columns[indices]
                        orig_df_cols = _undummify(self.df[cols_to_pick_out])
                        orig_df_cols = orig_df_cols.iloc[index_of_rows, :].copy()
                        df_new = pd.concat([df_new, orig_df_cols], axis=1)

                    df_new = df_new.drop(cat_expanded_columns, axis=1)
                    sc_boot = SpecificationCurve(
                        df_new,
                        y_endog="y_star",
                        x_exog=self.x_exog,
                        controls=self.orig_controls,
                        cat_expand=self.orig_cat_expand,
                        always_include=self.always_include,
                        exclu_grps=self.orig_exclu_grps,
                    )
                    sc_boot.fit()
                    coeff_this_bootstrap = pd.DataFrame()
                    coeff_this_bootstrap["Coefficient"] = sc_boot.df_r["Coefficient"]
                    coeff_this_bootstrap["Bootstrap"] = i
                    coeff_this_bootstrap["Specification No."] = y_star_k
                    coeff_this_bootstrap.index.name = "boot_spec_no"
                    df_spec_coef_results = pd.concat(
                        [
                            df_spec_coef_results,
                            coeff_this_bootstrap,
                        ],
                        axis=0,
                    )

                all_boot_df = pd.concat([all_boot_df, df_spec_coef_results], axis=0)
            quantiles_for_bootstrap = np.array([0.025, 0.5, 0.975])
            self.null_df = (
                all_boot_df.groupby("Specification No.")["Coefficient"]
                .quantile(quantiles_for_bootstrap)
                .unstack(level=1)
            )
            median_by_spec = all_boot_df.groupby("Specification No.")[
                "Coefficient"
            ].median()
            median_by_spec_p_value = pg.ttest(
                median_by_spec, self.df_r["Coefficient"]
            ).loc["T-test", "p-val"]
            number_specs = len(self.df_r["Coefficient"])
            if any(np.isin([number_specs, 0], (median_by_spec > 0).sum())):
                share_pls_p_value = np.nan
            else:
                share_pls_p_value = pg.ttest((median_by_spec > 0), True).loc[
                    "T-test", "p-val"
                ]
            if any(np.isin([number_specs, 0], (median_by_spec < 0).sum())):
                share_neg_p_value = np.nan
            else:
                share_neg_p_value = pg.ttest((median_by_spec < 0), True).loc[
                    "T-test", "p-val"
                ]

            def _nice_pval_text(num: float) -> str:
                """Turns p-value numbers into nice strings.

                Args:
                    num (float): p-value to stringify

                Returns:
                    str: Reading friendly version
                """
                if num <= 0.001:
                    return "<0.001"
                elif np.isinf(num) or np.isnan(num):
                    return "NA"
                else:
                    return str(_round_to_2(num))

            median_by_spec_p_text = _nice_pval_text(median_by_spec_p_value)
            share_pls_p_text = _nice_pval_text(share_pls_p_value)
            share_neg_p_text = _nice_pval_text(share_neg_p_value)
            self.null_stats_summary = pd.DataFrame(
                columns=["estimate", "p-value"],
                index=["median", "share positive", "share negative"],
                data=[
                    [f"{_round_to_2(median_by_spec.median())}", median_by_spec_p_text],
                    [
                        f"{(median_by_spec>0).sum()} of {number_specs}",
                        share_pls_p_text,
                    ],
                    [
                        f"{(median_by_spec<0).sum()} of {number_specs}",
                        share_neg_p_text,
                    ],
                ],
            )
        else:
            raise ValueError("Must have run .fit() before .fit_null()")

    def _create_plot_internal(
        self,
        preferred_spec: Union[List[str], List[None]],
        show_null_curve: bool,
    ) -> tuple[mpl.figure.Figure, List[mpl.axes._axes.Axes]]:
        """Makes plot of fitted specification curve.

        Args:
            preferred_spec (Union[List[str], List[None]]): Preferred specification. Defaults to [].
            show_null_curve (bool): whether to include the curve under the null.

        Returns:
            tuple[mpl.figure.Figure, List[mpl.axes._axes.Axes]]: fig, axes with chart on.
        """
        TEXT_ANNOTATION_OFFSET_POINTS = 5
        # Set up blocks for showing what effects are included
        df_spec = self.df_r["SpecificationCounts"].apply(pd.Series).fillna(0.0)
        pd.set_option("future.no_silent_downcasting", True)  # for the line below
        df_spec = df_spec.replace(0.0, False).replace(1.0, True)
        df_spec = df_spec.T
        df_spec = df_spec.sort_index()
        # The above produces a dataframe of the form:
        # Specification No.	0	1	2	3	4	5	6	7
        # x_1	True	True	True	True	True	True	True	True
        # x_2	False	False	True	True	False	False	True	True
        # x_3	True	True	True	True	False	False	False	False
        # x_4	False	True	False	True	False	True	False	True
        # y	True	True	True	True	True	True	True	True
        # in df_spec
        # Label the preferred specification as True
        self.df_r["preferred"] = False
        if preferred_spec:
            self.df_r["preferred"] = self.df_r["Specification"].apply(
                lambda x: Counter(x) == Counter(preferred_spec)
            )
        # This is quite hacky. It takes full list of variables and just keeps
        # those that we will be varying over.
        new_ctrl_names = list(set(_flatn_list(self.df_r["Specification"].values)))
        new_ctrl_names = [
            x for x in new_ctrl_names if x not in self.x_exog + self.y_endog
        ]
        new_ctrl_names.sort()
        # These are the names of the blocks under the chart
        name = self.x_exog + self.y_endog + new_ctrl_names
        # These are the groups in the blocks under the chart.
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
        # The above maps each individual variable to its group. So it might
        # look something like:
        # {'x_exog': 'x_exog',
        # 'y_endog': 'y_endog',
        # 'x_2': 'control',
        # 'x_3': 'control',
        # 'x_4': 'control'}
        block_df["group"] = block_df["group"].apply(lambda x: group_map[x])
        # The above puts this into a data frame describing the block struture.
        # eg it might look like
        #       group
        # x_1	x_exog
        # y	y_endog
        # x_2	control
        # x_3	control
        # x_4	control
        # where the first column is the index
        # The below does counts of variables by group. eg in the example above
        # there are three controls, one exog, one endog.
        counts = (
            block_df.reset_index()
            .groupby("group")
            .count()
            .rename(columns={"index": "counts"})
        )
        # For any group with a count of more than one, record it in block df
        block_df = pd.merge(
            block_df.reset_index(), counts.reset_index(), on=["group"]
        ).set_index("index")
        block_df = block_df.loc[block_df["counts"] > 1, :]
        index_dict = dict(
            zip(block_df["group"].unique(), range(len(block_df["group"].unique())))
        )
        block_df["group_index"] = block_df["group"].apply(lambda x: index_dict[x])
        # above gives, as an example,
        # 	    group	counts	group_index
        # index
        # x_2	control	3	0
        # x_3	control	3	0
        # x_4	control	3	0
        # heights of the blocks. Gets bigger as there are more entries.
        heights = [2] + [
            0.3 * np.log(x + 1)
            for x in block_df["group_index"].value_counts(sort=False)
        ]
        # --------------------------------------------------------------------
        # Make the plot
        # --------------------------------------------------------------------
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
            label="Median",
            dashes=[12, 5],
        )
        # Annotate the median coefficient line with text
        axarr[0].annotate(
            text="Median",
            xy=(0.1, np.median(self.df_r["Coefficient"])),
            xytext=(0, TEXT_ANNOTATION_OFFSET_POINTS),
            xycoords=("figure fraction", "data"),
            textcoords="offset points",
            fontsize=12,
            color="k",
            alpha=0.6,
            zorder=5,
        )
        # Colour the significant estimate values differently
        self.df_r["color_coeff"] = "black"
        self.df_r["coeff_pvals"] = self.df_r.apply(
            lambda row: row["pvalues"][row["x_exog"]], axis=1
        )
        self.df_r.loc[self.df_r["coeff_pvals"] < 0.05, "color_coeff"] = (
            "blue"  # "#91d1f1"
        )
        red_condition = (self.df_r["Coefficient"] < 0) & (
            self.df_r["coeff_pvals"] < 0.05
        )
        self.df_r.loc[red_condition, "color_coeff"] = "red"  # "#f94026"
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
                alpha=0.6,
                zorder=1,
                elinewidth=2,
                marker="o",
            )
            [bar.set_alpha(0.2) for bar in bars]
            [cap.set_alpha(0.2) for cap in caps]
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
        axarr[0].set_ylabel("Coefficient", fontsize=13)
        max_height = self.df_r["conf_int"].apply(lambda x: x.max()).max()
        min_height = self.df_r["conf_int"].apply(lambda x: x.min()).min()
        if show_null_curve:
            ns_df = self.null_df
            for column in ns_df.columns[::2]:
                axarr[0].plot(
                    ns_df[column].index, ns_df[column], ls="--", color="k", alpha=0.1
                )
            for column in ns_df.columns[1::2]:
                axarr[0].plot(
                    ns_df[column].index,
                    ns_df[column],
                    ls="dashdot",
                    color="k",
                    alpha=0.3,
                )

            max_height = max(max_height, ns_df.max().max())
            min_height = min(min_height, ns_df.min().min())

            # Annotate the null line
            axarr[0].annotate(
                xy=(0.6, ns_df.iloc[-1, 1]),  # type: ignore
                xycoords=("figure fraction", "data"),  # type: ignore
                xytext=(0, TEXT_ANNOTATION_OFFSET_POINTS),
                textcoords="offset points",
                text="Median under null",
                fontsize=11,
                color="gray",
                zorder=5,
                ha="left",
            )

        def get_chart_axes_limits(height: float, max: bool) -> float:
            """For positive numbers and max, returns height*axes_width_multiple.
            For negative numbers and max, returns height/axes_width_multiple
            max and max height > 0:
            np.sign(height) = 1
            height*exp(1*1*log(m)) = height*m
            max and max height < 0:
            np.sign(height) = -1
            height*exp(-1*1*log(m)) = height/m
            min and min height > 0:
            np.sign(height) = 1
            height*exp(1*-1*log(m)) = height/m
            min and min height < 0
            np.sign(height) = -1
            height*exp(-1*-1*log(m)) = height*m

            Args:
                height (float): The height of the error bar

            Returns:
                float: limit
            """
            axes_width_multiple = 1.2
            max_or_min_factor = 1 if max else -1
            return height * np.exp(
                np.sign(height) * max_or_min_factor * np.log(axes_width_multiple)
            )

        ylims = (
            get_chart_axes_limits(min_height, False),
            get_chart_axes_limits(max_height, True),
        )
        axarr[0].yaxis.set_minor_locator(AutoMinorLocator())
        axarr[0].set_ylim(_round_to_2(ylims[0]), _round_to_2(ylims[1]))
        for place in ["right", "top"]:
            axarr[0].spines[place].set_visible(False)

        # Now do the blocks - each group get its own array
        wid = 0.5
        hei = wid / 2.5
        spec_not_activated_color = "#F2F2F2"
        # This applies a tighter and more 'zoomed out' plot style when there
        # are a large number of specifications
        if len(self.df_r) > 160:
            wid = 0.01
            spec_not_activated_color = "#FFFFFF"
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
            # Loop over variables in the group (eg x_2, x_3)
            for i, row_name in enumerate(df_sp_sl.index):
                # Loop over specification numbers
                for j, col_name in enumerate(df_sp_sl.columns):
                    # retrieve colour based on significance
                    sig_colour = self.df_r.loc[j, "color_coeff"]
                    if df_sp_sl.iloc[i, j]:
                        color = sig_colour
                        alpha_choice = 0.4
                    else:
                        color = spec_not_activated_color
                        alpha_choice = 1.0

                    sq = patches.Rectangle(
                        (j - wid / 2, i - hei / 2),
                        wid,
                        hei,
                        fill=True,
                        color=color,
                        alpha=alpha_choice,
                    )
                    ax.add_patch(sq)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.set_yticks(range(len(list(df_sp_sl.index.values))))
            # Add text on the RHS that describes what each block is
            spacing_factor = 0.02  # Adjust this value to control spacing
            figure_width_adj = 0.5
            figure_width = len(df_sp_sl.columns) - figure_width_adj
            text_x_pos = figure_width * (1 + spacing_factor)
            ax.text(
                x=text_x_pos,
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
            ax.set_xlim(-wid, figure_width * (1 + spacing_factor * 2))
            for place in ["right", "top", "bottom"]:
                ax.spines[place].set_visible(False)
        for ax in axarr:
            ax.set_xticks([], minor=True)
            ax.set_xticks([])
            ax.set_xlim(-wid, len(df_spec.columns) - figure_width_adj)
        return fig, axarr

    def plot(
        self,
        save_path=None,
        pretty_plots: bool = True,
        preferred_spec: Union[List[str], List[None]] = [],
        show_null_curve: bool = False,
        return_fig: bool = False,
        **kwargs,
    ) -> Union[None, tuple[mpl.figure.Figure, List[mpl.axes._axes.Axes]]]:
        """Makes plot of fitted specification curve. Optionally returns figure and axes for onward adjustment.

        Args:
            save_path (string or path, optional): Exported fig filename. Defaults to None.
            pretty_plots (bool, optional): Whether to use this package's figure formatting. Defaults to True.
            preferred_spec (list, optional): Preferred specification. Defaults to [].
            show_null_curve (bool, optional): Whether to include the curve under the null. Defaults to False.
            pretty_plots (bool, optional): Whether to use this package's figure formatting. Defaults to False.
            return_fig (bool, optional): Whether to return the figure and axes objects. Defaults to False.
            **kwargs (dict, optional): Additional arguments passed to .fit_null() when show_null_curve is True. Parameters:
                n_boot (int): Number of bootstrap iterations for null curve calculation. eg the argument would be `**{"n_boot": 5}`.
                f_sample (float): Fraction of rows to sample in each bootstrap. Defaults to 0.1.

        Returns:
            Union[None, tuple[mpl.figure.Figure, List[mpl.axes._axes.Axes]]]: None or the fig and axes with chart on.

        Raises:
            ValueError: If .plot() is called before .fit() - the fit must be run first.

        """
        if pretty_plots:
            _pretty_plots()
        if not hasattr(self, "df_r"):
            raise ValueError("Nothing to plot. Must run .fit() before .plot()")
        if show_null_curve and "n_boot" in kwargs.keys():  # use new n_boots
            self.fit_null(**kwargs)
        if show_null_curve and not hasattr(self, "null_df"):
            self.fit_null(**kwargs)
        fig, axes = self._create_plot_internal(preferred_spec, show_null_curve)
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        if return_fig:
            return fig, axes
        else:
            plt.show()
            return None


def load_example_data1() -> pd.DataFrame:
    """Retrieves example data from a file included with the package.

    Returns:
        pd.DataFrame: Example data suitable for regression.
    """
    # Example data
    ref = resources.files("specification_curve") / os.path.join(
        "data", "example_data.csv"
    )
    with resources.as_file(ref) as path:
        df = pd.read_csv(path, index_col=0)

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


def load_example_data3() -> pd.DataFrame:
    """Generates fake data.

    Returns:
        pd.DataFrame: Example data suitable for regression.
    """
    n_samples = 20000
    # Number of dimensions
    n_dim = 4
    c_rnd_vars = prng.random(size=(n_dim, n_samples))
    x_bool = prng.integers(2, size=n_samples)
    y_1 = (
        0.4 * c_rnd_vars[0, :]  # THIS IS THE TRUE VALUE OF THE COEFFICIENT
        - 0.2 * c_rnd_vars[1, :]
        + 0.3 * prng.standard_normal(n_samples)
        + 0.6
        + 0.2 * x_bool
    )
    # Next line causes y_2 ests to be much more noisy
    y_2 = y_1 - 0.5 * np.abs(prng.standard_normal(n_samples))
    # Put data into dataframe
    df = pd.DataFrame([y_1, y_2], ["y1", "y2"]).T
    df["x1"] = c_rnd_vars[0, :]
    df["c1"] = c_rnd_vars[1, :]
    df["c2"] = c_rnd_vars[2, :]
    df["c3"] = c_rnd_vars[3, :]
    df["ccat"] = x_bool
    return df
