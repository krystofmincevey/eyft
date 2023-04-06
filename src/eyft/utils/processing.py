import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List
from scipy import stats

from ..utils import logger


# TODO: Check which things are done in place
#   and potentially remove return statements.


def remove_spaces(
    df: pd.DataFrame,
    col: str,
) -> pd.DataFrame:
    """
    :param df: dataset we use
    :param col: you can choose which column you want to clean
    :return: self (i.e. pd.Dataframe changed in place)
    """
    df[col] = df[col].str.re.replace(" +", " ").str.trim()
    return df


def merge_data(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_key: str,
    right_key: str,
    columns: list,
    merge_type: str = "left",
) -> pd.DataFrame:
    """
    :param df_left: dataset we want to enrich
    :param df_right: dataset used to add information to the left dataset
    :param left_key: field used for the matching in the target table
    :param right_key: field used for the matching in the right table
    :param columns: columns from the right table we want to add in the left table
    :param merge_type: what type of join you want to do
    """
    for col in columns:
        if col in df_left.columns:
            logger.warning(
                f"{col} already in df_left and "
                f"hence not joined."
            )

    df_right = df_right[right_key, *columns]

    df = df_left.merge(
        df_right,
        left_on=left_key,
        right_on=right_key,
        how=merge_type
    )
    return df


# TODO: Not finished
def sum_statistics(
    df: pd.DataFrame,
    col: str,
):
    """
    :param df: dataset we use
    :param col: you can choose the column where you want to know the summary statistics from
    :return: the summary statistics of the chosen column
    """
    df[col] = df[col].describe


def boxplot(
    df: pd.DataFrame,
    col: str,
    by_user=str,
):
    """
    :param df: dataset we use
    :param col: you can choose the column you want to use to calculate a boxplot
    :param by_user: optional choice to perform a grouping of the targeted column
    :return: boxplot of the chosen column
    """
    plt.boxplot(df[col], by=by_user)
    plt.show()


def histogram(
    df: pd.DataFrame,
    col: str,
    bins: List[float, int] = None,
) -> List[float, int]:
    """
    :param df: dataset we use
    :param col: you can choose the column you want to use to calculate the histogram
    :param bins: numerical limits of the bins in the histogram
    :return: histogram of the chosen column
    """
    if bins is None:
        bins = np.histogram_bin_edges(df[col], bins="fd")
    plt.hist(df[col], bins=bins)
    plt.show()

    return bins


def remove_symbols(
    df: pd.DataFrame,
    col: str,
    list_symbols: list,
    cleaning_list: list,
) -> pd.DataFrame:
    """
    :param df: dataset we use
    :param col: you can choose the column you want to clean
    :param list_symbols: symbols you want to remove
    :param cleaning_list: what value needs to be put
        instead of the removed symbol
    """
    if len(list_symbols) != len(cleaning_list):
        raise "List of symbols length do not match with the cleaning values length"
    for symbol, clean_symbol in zip(list_symbols, cleaning_list):
        df[col] = df[col].str.replace(symbol, clean_symbol)
    return df


def normalise(df: pd.DataFrame, cols: list, is_plot=False):
    """
    Transform continuous covariates that are not
    gaussian.
    """
    for col in cols:
        stat, p = stats.shapiro(df[col].values)
        if p < 0.01:
            print(f'\n{col} may not be Gaussian.')
            print(f'Pre Transformation; {col}:')
            print(df[col].describe())

            if is_plot:
                df[col].plot.hist(title=col)
                plt.show()

            # BOX COX NEEDS + VALS
            min_val = df[col].min()
            if min_val <= 0:
                normaliser = np.abs(min_val) + 1
                print(f"Normaliser: {normaliser}")
            else:
                normaliser = 0

            tf_vals, _lambda = stats.boxcox(df[col].values + normaliser)
            df.drop(labels=[col], axis="columns", inplace=True)

            new_col = f'{col}_plus_with_box_cox'

            df[new_col] = tf_vals
            print(f'Post transformation; {new_col} properties:')
            print(df[new_col].describe())

            if is_plot:
                df[new_col].plot.hist(title=new_col)
                plt.show()

    return df


def explore(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    max_cats: int = 15,
):
    """
    Analyses column x_col in df, generating both descriptive
    statistics, and a seaborn plot (x_col vs y_col).
    Note that if the number of unique values in x_col exceeds
    max_cats, the column is treated as numerical.
    Otherwise, the column is treated as a categorical
    col.
    """

    print('---'*10)
    print(f'Analysing X: {x_col} | Y: {y_col}')
    print(df[x_col].describe())

    if df[x_col].nunique() > max_cats:
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        sns.scatterplot(x=df[x_col], y=df[y_col])
    else:
        try:
            sns.boxplot(x=x_col, y=y_col, data=df)
        except ValueError:
            logger.info(
                f'Unable to plot X: {x_col} vs Y: {y_col}.'
            )
            pass

    plt.show()



