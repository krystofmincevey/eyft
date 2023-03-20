import pandas as pd

from typing import List

from src.eyft import logger


# TODO: Can potentially replace cutoff
#  with n - number of features to keep


def select(
        df: pd.DataFrame,
        cols: List[str]
) -> pd.DataFrame:
    return df[cols]


def step_wise(
        df: pd.DataFrame,
        cutoff: float
) -> List[str]:
    """
    Returns list of most important features (p_val < cutoff).
    The list is compiled by building a linear model
    using p vals in a step-wise manner.
    """
    raise NotImplementedError
    # logger.info(f'{col}: {p_val}')
    # return features


def random_forrest(
        df: pd.DataFrame,
        cutoff: float,
) -> List[str]:
    """
    Keep most important features
    (as determined by random forest)
    satisfying cutoff
    """
    raise NotImplementedError


def lasso(
        df: pd.DataFrame,
        cutoff: float,
) -> List[str]:
    """
    Keep weights that are larger than cutoff.
    """
    raise NotImplementedError


def pearson(
    df: pd.DataFrame,
    cutoff: float = 0.8,
) -> List[str]:
    """
    Keep features that have a pearson
    correlation lower than the cutoff.
    """
    raise NotImplementedError


def vif(
        df: pd.DataFrame,
        cutoff: float = 5,
) -> List[str]:
    """
    Keep features that have a VIF (Variance Inflation Factor)

    """
    raise NotImplementedError


# TODO: Integrate feature selection into either random or bayesian
#  search during model selection.

