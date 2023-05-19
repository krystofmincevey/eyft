import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests

from typing import List, Dict, Union
from scipy import stats


from ..utils import logger


# TODO: Check which things are done in place
#   and potentially remove return statements.


def translate_text(
    text: str,
    source_lang: str = 'fr',
    target_lang: str = 'en',
):
    url = "https://api.mymemory.translated.net/get"
    params = {
        "q": text,
        "langpair": f"{source_lang}|{target_lang}"
    }
    response = requests.get(url, params=params)
    translation = response.json()["responseData"]["translatedText"]
    return translation


def process_str(
    txt: str,
    mapping: Dict[str, str]
) -> str:

    for key, val in mapping.items():
        txt = txt.replace(key, val)
    txt = re.sub(" +", " ", txt).strip()
    return txt


def merge(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_key: str,
    right_key: str,
    columns: list = None,
    merge_type: str = "left",
) -> pd.DataFrame:
    """
    :param df_left: dataset we want to enrich
    :param df_right: dataset used to add information to the left dataset
    :param left_key: field used for the matching in the y_col table
    :param right_key: field used for the matching in the right table
    :param columns: columns from the right table we want to add into the left table
    :param merge_type: what type of join you want to do
    """

    # Handle Inputs -----------------------------------
    if merge_type not in ["inner", "left"]:
        raise ValueError(
            f"{merge.__name__} only supports "
            f"inner and left joins not {merge_type}."
        )

    if columns is None:
        columns = list(df_right.columns)
        columns.remove(right_key)

    # REMOVE NAN ROWS IN PRIMARY KEY TO SPEED UP JOIN
    df_right = df_right[~df_right[right_key].isna()]
    # -------------------------------------------------

    for col in columns.copy():
        if col in df_left.columns:
            columns.remove(col)
            logger.warning(
                f"{col} already in df_left and "
                f"hence not joined."
            )
    df_right = df_right[[*columns, right_key]]

    df = df_left.merge(
        df_right,
        left_on=left_key,
        right_on=right_key,
        how=merge_type
    )

    # Sanity check ------------------------------------
    if merge_type == 'left' and df.shape[0] != df_left.shape[0]:
        logger.warning(
            f"Left join expanded the size (nrows) of the"
            f"dataframe from {df_left.shape[0]} to {df.shape[0]}."
        )
    # ------------------------------------------------
    return df


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
    bins: List[Union[float, int]] = None,
) -> List[Union[float, int]]:
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


def normalise(df: pd.DataFrame, cols: list, is_plot=False):
    """
    Transform continuous covariates that are not
    gaussian to be more normal.
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
