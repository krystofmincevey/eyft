import pandas as pd
import statsmodels.api as sm

from typing import List

from ..feature_engineering import logger


# TODO: Can potentially (in some cases) replace cutoff
#  with n - number of features to keep


def select(
        df: pd.DataFrame,
        cols: List[str]
) -> pd.DataFrame:
    return df[cols]


def forward_select(
    df: pd.DataFrame,
    y_col: str,
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
    initial_features = df.columns.tolist()
    best_features = []
    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(y_col, sm.add_constant(df[best_features + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if min_p_value < cutoff:
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features


def backward_eliminate(
        df: pd.DataFrame,
        y_col: str,
        cutoff: float
) -> List[str]:
    """
    In backward elimination, we start with the full model
    (including all the independent variables) and then remove
    the insignificant feature with the highest p-value(> significance level).
    This process repeats again and again until we have the final
    set of significant features.
    """
    features = df.columns.tolist()
    while len(features) > 0:
        features_with_constant = sm.add_constant(df[features])
        p_values = sm.OLS(y_col, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if max_p_value >= cutoff:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features


def step_wise_select(
        df: pd.DataFrame,
        y_col: str,
        cutoff_in: float = 0.05,
        cutoff_out: float = 0.05,
) -> List[str]:
    """
    It is similar to forward selection but the difference is
    while adding a new feature it also checks the significance
    of already added features and if it finds any of the already
    selected features insignificant then it simply removes
    that particular feature through backward elimination.

    Hence, It is a combination of forward selection and backward elimination.
    """
    initial_features = df.columns.tolist()
    best_features = []
    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(y_col, sm.add_constant(df[best_features + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if min_p_value < cutoff_in:
            best_features.append(new_pval.idxmin())
            while len(best_features) > 0:
                best_features_with_constant = sm.add_constant(df[best_features])
                p_values = sm.OLS(y_col, best_features_with_constant).fit().pvalues[1:]
                max_p_value = p_values.max()
                if max_p_value >= cutoff_out:
                    excluded_feature = p_values.idxmax()
                    best_features.remove(excluded_feature)
                else:
                    break
        else:
            break
    return best_features


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

