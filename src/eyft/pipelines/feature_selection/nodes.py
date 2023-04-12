import numpy as np
import pandas as pd
import xgboost as xgb
import statsmodels.api as sm

from typing import List
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ..feature_engineering import logger
from ...utils.models import _seed
from ...utils.constants import SEED, RF_PARAMS


# TODO: Integrate feature selection into either random or bayesian
#  search during model selection.
# TODO: Arthur: Test feature selection functions. ERRORS may exist in functions which you may have to fix!
# TODO: Implement CV feature selection


def select(
    df: pd.DataFrame,
    cols: List[str],
) -> pd.DataFrame:
    return df[cols]


def _sm_model(x_df: pd.DataFrame, y_df: pd.DataFrame):
    if len(np.unique(y_df.values)) <= 2:
        model = sm.Logit(y_df, x_df)
    else:
        model = sm.OLS(y_df, sm.add_constant(x_df))
    return model.fit()


def forward_select(
    df: pd.DataFrame,
    y_col: str,
    cutoff: float = 0.05,
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

    logger.info(
        f"Performing forward feature selection using the "
        f"cutoff: {cutoff}."
    )

    y = df[y_col].values

    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = _sm_model(x_df=df[best_features + [new_column]], y_df=y)
            new_pval[new_column] = model.pvalues[new_column]

        min_p_value = new_pval.min()
        if min_p_value < cutoff:
            best_features.append(new_pval.idxmin())
            logger.info(
                f"Selected {best_features[-1]} whose p-value is: {min_p_value}."
            )
        else:
            break
    return best_features


def backward_eliminate(
    df: pd.DataFrame,
    y_col: str,
    cutoff: float = 0.05
) -> List[str]:
    """
    In backward elimination, we start with the full model
    (including all the independent variables) and then remove
    the insignificant feature with the highest p-value(> significance level).
    This process repeats again and again until we have the final
    set of significant features.
    """
    features = df.columns.tolist()

    logger.info(
        f"Performing backward feature selection using the "
        f"cutoff: {cutoff}."
    )

    y = df[y_col]
    while len(features) > 0:
        model = _sm_model(x_df=df[features], y_df=y)
        p_values = model.pvalues[features]
        max_p_value = p_values.max()
        if max_p_value >= cutoff:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
            logger.info(
                f"Removed {excluded_feature} whose p-value was: {max_p_value}."
            )
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

    logger.info(
        f"Performing stepwiseselection using the "
        f"forward cutoff: {cutoff_in} and backward "
        f"cutoff: {cutoff_out}."
    )

    y = df[y_col]
    while len(initial_features) > 0:

        # FORWARD -----------------------------------------------------------
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = _sm_model(x_df=df[best_features + [new_column]], y_df=y)
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if min_p_value < cutoff_in:
            best_features.append(new_pval.idxmin())
            logger.info(
                f"Selected {best_features[-1]} whose p-value is: {min_p_value}."
            )

            # BACKWARD ------------------------------------------------------
            while len(best_features) > 0:
                model = _sm_model(x_df=df[best_features], y_df=y)
                p_values = model.pvalues[best_features]
                max_p_value = p_values.max()
                if max_p_value >= cutoff_out:
                    excluded_feature = p_values.idxmax()
                    best_features.remove(excluded_feature)
                    logger.info(
                        f"Removed {excluded_feature} whose p-value was: {max_p_value}."
                    )
                else:
                    break
        else:
            break
    return best_features


@_seed
def random_forrest(
    df: pd.DataFrame,
    y_col: str,
    cutoff: float,
) -> List[str]:
    """
    Keep most important features
    (as determined by random forest)
    satisfying cutoff
    """
    df_x = df.loc[:, ~df.columns.isin([y_col])]
    y = df[y_col].values

    # MODEL
    if len(np.unique(y)) < 2:
        model = xgb.XGBClassifier(random_state=SEED, max_depth=3)
    else:
        model = xgb.XGBRegressor(**RF_PARAMS)

    model.fit(df_x, y)
    importance = model.get_booster().get_score(
        importance_type='weight', fmap=''
    )

    # SELECT BEST FEATURES
    imp_tuples = list(importance.items())
    best_features = [
        feature for feature, imp in imp_tuples if imp > cutoff
    ]
    return best_features


def lasso(
        df: pd.DataFrame,
        cutoff: float,
) -> List[str]:
    """
    Keep weights that are larger than cutoff,
    using lasso regressions to determine weights.
    """

    lasso(df, cutoff=)

    # TODO: Arthur
    raise NotImplementedError
    # return features


def pearson(
    df: pd.DataFrame,
    cutoff: float = 0.80,
) -> List[str]:
    """
    Keep features that have a pearson
    correlation lower than the cutoff.

    IMAGINE IF in y = mx + c + error
    in inputs you also had 2x
    x and 2x  -> this can pose issues when modelling

    check correlations and only keep either 2x or x but not both!
    """
    corr_matrix = df.corr()
    upper = corr_matrix.where((np.triu(np.ones(corr_matrix.shape), k=1) +
                               np.tril(np.ones(corr_matrix.shape), k=-1)).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
    df.drop(to_drop, axis=1, inplace=True)

    raise NotImplementedError
    # return features


def vif(
    df: pd.DataFrame,
    y_col: str,
    cutoff: float = 5,
    exclude_cols: List[str] = [],
) -> List[str]:
    """
    Keep features that have a VIF (Variance Inflation Factor)
    smaller than cutoff
    """

    i = 1
    max_vif_feature = 'None'  # kept as string so that while loop commences
    collinear_variables = []
    x_df = df.loc[:, ~df.columns.isin([y_col])]
    features = [col for col in x_df.columns if col not in exclude_cols]

    logger.info('\nRemoving collinear variables...')
    while max_vif_feature is not None:

        # DETERMINE VIF:
        _x_df = sm.add_constant(df[features])
        vif = [
            variance_inflation_factor(_x_df.values, i)
            for i in range(_x_df.shape[1])
        ]
        vif_df = pd.DataFrame({'vif': vif[1:]}, index=features).T

        # DETERMINE MAX VIF:
        max_vif = cutoff
        max_vif_feature = None
        for column in vif_df.columns:
            _vif = float(vif_df[column]['vif'])
            # logger.info(f"ROUND {round}: {column} vif = {vif}")
            if _vif > max_vif:
                max_vif = _vif
                max_vif_feature = column

        # REMOVE MAX VIF FEATURE:
        if max_vif_feature is not None:
            df = df.drop([max_vif_feature], axis=1)
            features.remove(max_vif_feature)
            logger.info(f'ROUND {i}: {max_vif_feature} is dropped with vif {max_vif}')
            collinear_variables.append(max_vif_feature)
        i += 1

    return features

# TODO: Finalise scafold for feature selection
# @_seed
# def calculate_dt_feature_imps(
#     df: pd.DataFrame,
#     n_eval: int = N_EVAL,
#     train_size: float = TRAIN_SIZE,
#     target_col: str = TARGET_COL,
# ) -> pd.DataFrame:
#
#     print('\nCalculating feature importances using XGBoot...')
#
#     n_obs = df.shape[0]
#     n_train = int(n_obs * train_size)
#     ranked_features = defaultdict(list)
#     for _ in range(n_eval):
#
#         train_idx = np.random.choice(range(n_obs), n_train, replace=False)
#         exp = Explore(df=df.iloc[train_idx], target_col=target_col)
#         exp.feature_select('rf', is_eval=False, is_plot=False)
#         _ranked_features = exp.get_ranked_features()
#
#         for col in df.columns:
#             ranked_features[col].append(_ranked_features[col])
#
#     f_imps = pd.DataFrame.from_dict(ranked_features)
#
#     median_f_imps = f_imps.median()
#     median_f_imps.sort_values(ascending=False, inplace=True)
#
#     print('\nFeature Importances:')
#     print(median_f_imps)
#     print('\n'*3)
#
#     return median_f_imps
#
#
# class Explore(object):
#
#     def __init__(self, df: pd.DataFrame, target_col: str):
#         self._df = df
#         self._target_col = target_col
#
#         self._ranked_features = None
#         self._f_importance = None
#
#     def get_ranked_features(self) -> dict:
#         ranked_features_dict = defaultdict(lambda: 0)
#         for feature, imp in zip(self._ranked_features, self._f_importance):
#             ranked_features_dict[feature] = imp
#         return ranked_features_dict
#
#     @_seed
#     def feature_select(self, key: str, **kwargs):
#         # Potential for future expansion
#         key2fselect = {
#             'rf': self.rf_select
#         }
#
#         _key = key.lower()
#         if _key not in key2fselect:
#             raise KeyError(f'feature_select does not support {key}.')
#
#         fs_func = key2fselect[_key]
#         return fs_func(**kwargs)
#
#     def rf_select(self, is_plot=True, is_eval=True):
#         X = self._df.loc[:, ~self._df.columns.isin([self._target_col])]
#         y = self._df[self._target_col].values
#
#         if len(np.unique(y)) < 2:
#             model = xgb.XGBClassifier(random_state=SEED, max_depth=3)
#         else:
#             model = xgb.XGBRegressor(**RF_PARAMS)
#
#         model.fit(X, y)
#
#         importance = model.get_booster().get_score(
#             importance_type='weight', fmap=''
#         )
#         tuples = list(importance.items())
#         tuples = sorted(tuples, key=lambda x: x[1])
#         labels, values = zip(*tuples)
#         self._ranked_features = labels
#         self._f_importance = values
#
#         return
