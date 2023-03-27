import pandas as pd

from typing import List

from ..feature_engineering import logger


# TODO: Can potentially replace cutoff
#  with n - number of features to keep


def select(
        df: pd.DataFrame,
        cols: List[str]
) -> pd.DataFrame:
    return df[cols]


def forward_select(
    df: pd.DataFrame,
    cutoff: float
) -> List[str]:
    """
    In forward selection, we start with a null model and then start
    fitting the model with each individual feature one at a time
    and select the feature with the minimum p-value. Now fit a model
    with two features by trying combinations of the earlier selected
    feature with all other remaining features. Again select the
    feature with the minimum p-value. Now fit a model with three
    features by trying combinations of two previously selected features
    with other remaining features. Repeat this process until we
    have a set of selected features with a p-value of individual features
    less than the significance level.
    """
    raise NotImplementedError
    # logger.info(f'{col}: {p_val}')
    # return features


def backward_eliminate(
        df: pd.DataFrame,
        cutoff: float
) -> List[str]:
    """
    In backward elimination, we start with the full model
    (including all the independent variables) and then remove
    the insignificant feature with the highest p-value(> significance level).
    This process repeats again and again until we have the final
    set of significant features.
    """
    raise NotImplementedError
    # logger.info(f'{col}: {p_val}')
    # return features


def step_wise_select(
        df: pd.DataFrame,
        cutoff: float
) -> List[str]:
    """
    It is similar to forward selection but the difference is
    while adding a new feature it also checks the significance
    of already added features and if it finds any of the already
    selected features insignificant then it simply removes
    that particular feature through backward elimination.

    Hence, It is a combination of forward selection and backward elimination.

    Returns list of most important features (p_val > cutoff).
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
    smaller than cutoff
    """
    raise NotImplementedError


# TODO: Integrate feature selection into either random or bayesian
#  search during model selection.

