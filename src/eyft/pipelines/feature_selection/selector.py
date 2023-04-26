import numpy as np
import pandas as pd
import xgboost as xgb
import statsmodels.api as sm

from typing import List, Dict
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ..feature_engineering import logger
from ...utils.models import seed
from ...utils.constants import SEED, RF_PARAMS
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso

# TODO: Integrate feature selection into either random or bayesian
#  search during model selection.
# TODO: Arthur: Test feature selection functions. ERRORS may exist in functions which you may have to fix!
# TODO: Implement CV feature selection


def _sm_model(x_df: pd.DataFrame, y_df: pd.DataFrame):
    """
    Either returns OLS or Logit GLMs. Logit
    is returned when y is categorical.
    """
    if len(np.unique(y_df)) <= 2:
        model = sm.Logit(y_df, x_df)
    else:
        model = sm.OLS(y_df, sm.add_constant(x_df))
    return model.fit()


def forward_select(
    df: pd.DataFrame,
    y_col: str,
    cutoff: float = 0.05,
    exclude_cols: List[str] = None
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
    logger.info(
        f"Performing forward feature selection using the "
        f"cutoff: {cutoff}."
    )

    y = df[y_col].values
    df = df.loc[:, ~df.columns.isin([y_col])]
    features = [] if exclude_cols is None else exclude_cols
    initial_features = df.columns.to_list()

    while len(initial_features) > 0:

        remaining_features = list(set(initial_features) - set(features))

        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = _sm_model(x_df=df[features + [new_column]], y_df=y)
            new_pval[new_column] = model.pvalues[new_column]

        min_p_value = new_pval.min()
        if min_p_value < cutoff:
            min_feature = new_pval.idxmin()
            features.append(min_feature)
            logger.info(
                f"Selected {min_feature} whose p-value is: {min_p_value}."
            )
        else:
            break

    return features


def backward_eliminate(
    df: pd.DataFrame,
    y_col: str,
    cutoff: float = 0.05,
    exclude_cols: List[str] = None,
) -> List[str]:
    """
    In backward elimination, we start with the full model
    (including all the independent variables) and then remove
    the insignificant feature with the highest p-value(> significance level).
    This process repeats again and again until we have the final
    set of significant features.
    """
    logger.info(
        f"Performing backward feature selection using the "
        f"cutoff: {cutoff}."
    )

    y = df[y_col].values
    df = df.loc[:, ~df.columns.isin([y_col])]
    features = df.columns.tolist()
    exclude_cols = [] if exclude_cols is None else exclude_cols
    remaining_features = list(set(features) - set(exclude_cols))

    while len(remaining_features) > 0:

        model = _sm_model(x_df=df[features], y_df=y)
        p_values: pd.Series = model.pvalues[remaining_features]

        max_p_value = p_values.max()
        if max_p_value >= cutoff:
            remove_feature: str = p_values.max()
            features.remove(remove_feature)
            remaining_features.remove(remove_feature)
            logger.info(
                f"Removed {remove_feature} whose p-value was: {max_p_value}."
            )
        else:
            break

    return features


def step_wise_select(
    df: pd.DataFrame,
    y_col: str,
    cutoff_in: float = 0.05,
    cutoff_out: float = 0.05,
    exclude_cols: List[str] = None,
) -> List[str]:
    """
    It is similar to forward selection but the difference is
    while adding a new feature it also checks the significance
    of already added features and if it finds any of the already
    selected features insignificant then it simply removes
    that particular feature through backward elimination.

    Hence, It is a combination of forward selection and backward elimination.
    """
    logger.info(
        f"Performing stepwiseselection using the "
        f"forward cutoff: {cutoff_in} and backward "
        f"cutoff: {cutoff_out}."
    )

    y = df[y_col].values
    df = df.loc[:, ~df.columns.isin([y_col])]
    features = [] if exclude_cols is None else exclude_cols
    initial_features = df.columns.tolist()

    while len(initial_features) > 0:

        # FORWARD -----------------------------------------------------------
        remaining_features = list(set(initial_features) - set(features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = _sm_model(x_df=df[features + [new_column]], y_df=y)
            new_pval[new_column] = model.pvalues[new_column]

        min_p_value = new_pval.min()
        if min_p_value < cutoff_in:
            features.append(new_pval.idxmin())
            logger.info(
                f"Selected {features[-1]} whose p-value is: {min_p_value}."
            )

            # BACKWARD ------------------------------------------------------
            while len(features) > 0:
                model = _sm_model(x_df=df[features], y_df=y)
                p_values = model.pvalues[features]
                max_p_value = p_values.max()
                if max_p_value >= cutoff_out:
                    remove_feature = p_values.idxmax()
                    features.remove(remove_feature)
                    logger.info(
                        f"Removed {remove_feature} whose p-value was: {max_p_value}."
                    )
                else:
                    break
        else:
            break

    return features


@seed
def random_forrest(
    df: pd.DataFrame,
    y_col: str,
    cutoff: float = None,
    exclude_cols: List[str] = None,  # TODO: Krystof
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
    imp_tuples.sort(key=lambda x: x[1], reverse=True)

    # Set cutoff - if none use the mean importance.
    cutoff = cutoff if cutoff is not None else- sum(importance.values())/len(imp_tuples)

    features = []
    for _feature, imp in imp_tuples:
        if imp > cutoff:
            logger.info(
                f'Selecting {_feature} whose feature importance: {imp} '
                f'is greater than cutoff: {cutoff}'
            )
            features.append(_feature)
        else:
            break # since sorted in descending order

    return features


def lasso(
    df: pd.DataFrame,
    y_col: str,
    cutoff: float,
    exclude_cols: List[str] = None,  # TODO: Krystof
) -> List[str]:
    """
    Keep weights that are larger than cutoff,
    using lasso regressions to determine weights.
    """
    X,y = df(return_X_y=True)
    features = df()['col']
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso())
    ])

    search = GridSearchCV(pipeline,
                          {'model_alpha': np.arange(0.1, 10, 0.1)},
                          cv=5, scoring="neg_mean_squared_error", verbose=3
                          )

    search.best_params_()
    coefficients = search.best_estimator_.named_steps['model'].coef
    importance = np.abs(coefficients)

    np.array(features)[importance > 0]
    np.array(features)[importance == 0]

    return features


    # TODO: Arthur
    raise NotImplementedError
    # return features


def pearson(
    df: pd.DataFrame,
    y_col: str,
    cutoff: float = 0.80,
    exclude_cols: List[str] = None,  # TODO: Krystof
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
    exclude_cols: List[str] = None,
) -> List[str]:
    """
    Keep features that have a VIF (Variance Inflation Factor)
    smaller than cutoff
    """

    i = 1
    exclude_cols = exclude_cols if exclude_cols is not None else []
    max_vif_feature = 'None'  # kept as string so that while loop starts
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

        i += 1

    return features


class Selector(object):

    MAPPER = {
        "rf": random_forrest,
        "forward": forward_select,
        "backward": backward_eliminate,
        "step": step_wise_select,
        "lasso": lasso,
        "vif": vif,
        "pearson": pearson,
    }

    def __init__(self, df: pd.DataFrame, target_col: str):
        self._df = df
        self.target_col = target_col

        self._best_features = None

    @property
    def target_col(self):
        return self._target_col

    @target_col.setter
    def target_col(self, col: str):
        if col not in self._df.columns:
            raise KeyError(
                f"Target col: {col} not in "
                f"df.columns: {self._df.columns}"
            )
        self._target_col = col

    def get_best_features(self):
        return self._best_features

    def feature_select(
        self, key,
        is_cv: bool = False,
        **kwargs
    ):
        if is_cv in ['True', True]:
            self.cv_select(key, **kwargs)
        else:
            self.single_select(key, **kwargs)

    @seed
    def single_select(self, key: str, **kwargs: Dict[str, str]):
        _key = key.lower()
        if _key not in self.MAPPER:
            raise KeyError(f'feature_select does not support {key}.')

        fs_func = self.MAPPER[_key]
        logger.info(f"Selecting important features using: {fs_func.__name__}")
        self._best_features = fs_func(
            self._df, y_col=self.target_col, **kwargs
        )
        return self

    @seed
    def cv_select(
        self,
        key: str,
        n_eval: int = 10,
        train_prc: float = 0.8,
        cutoff = None,
        **kwargs: Dict[str, str]
    ):
        """
        Performs Cross Validation (CV) feature selection.
        Across n_evals a feature selection method identified using
        key is run to select a list of important features.
        Only features selected more than cutoff times
        are then returned (as important features) by cv_select.
        """

        # ERROR HANDLING ---------------------------------------------
        if train_prc > 1 or train_prc <= 0:
            raise ValueError(
                f'train_prc must be (0-1] and not {train_prc}'
            )

        _key = key.lower()
        if _key not in self.MAPPER:
            raise KeyError(f'cv_select does not support {key}.')
        # ------------------------------------------------------------

        n_obs = self._df.shape[0]
        n_train = int(n_obs * train_prc)
        cutoff = cutoff if cutoff is not None else int(n_eval/2)

        fs_func = self.MAPPER[_key]
        feature_ranks = {
            col: 0 for col in self._df.columns if col != self._target_col
        }

        logger.info(
            f"Selecting important features using {n_eval}fold "
            f"cross validation and {fs_func.__name__}. "
        )
        for i in range(n_eval):

            train_idx = np.random.choice(range(n_obs), n_train, replace=False)
            features = fs_func(
                self._df.iloc[train_idx],
                y_col=self.target_col,
                **kwargs
            )
            logger.info(f"EVAL: {i} | Selected features: {features}")

            for _feature in features:
                feature_ranks[_feature] += 1

        fimps = list(feature_ranks.items())
        fimps.sort(key=lambda x: x[1], reverse=True)
        self._best_features = [
            _feature for _feature, imp in fimps if imp > cutoff
        ]
        logger.info(
            f"FINAL CV SELECTION COUNTS: {fimps} | Selecting "
            f"{self._best_features}."
        )
        return self
