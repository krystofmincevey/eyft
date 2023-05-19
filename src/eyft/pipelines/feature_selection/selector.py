import numpy as np
import pandas as pd
import xgboost as xgb
import statsmodels.api as sm

from typing import List, Dict, Hashable, Any, Optional
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Lasso
from sklearn.feature_selection import mutual_info_classif

from ..feature_engineering import logger
from ...utils.models import seed, is_binary, is_categorical
from ...utils.constants import SEED, RF_PARAMS


# TODO: Integrate feature selection into either random or bayesian
#  search during model selection.
# TODO: make sure these feature selection tools work for both
#  continuous and categorical variables.


def _set_list(vals) -> List[Any]:
    """Ensure that vals are list"""
    if vals is None:
        return []
    elif type(vals) != list:
        return [vals]
    else:
        return vals


def _sm_model(df_x: pd.DataFrame, y: np.ndarray):
    """
    Either returns OLS or Logit GLMs. Logit
    is returned when y is categorical.
    """
    if is_binary(y):
        model = sm.Logit(y, df_x)
    else:
        model = sm.OLS(y, sm.add_constant(df_x))
    return model.fit()


def _rf_model(df_x: pd.DataFrame, y: np.ndarray):
    """
    Either returns Classifier or Regressor.
    Classifier is returned when y is categorical.
    """

    # MODEL
    if is_categorical(y):
        model = xgb.XGBClassifier(random_state=SEED, max_depth=3)
    else:
        model = xgb.XGBRegressor(**RF_PARAMS)

    model.fit(df_x, y)
    return model


def forward_select(
    df: pd.DataFrame,
    y_col: str,
    cutoff: float = 0.05,
    keep_cols: List[str] = None
) -> List[Hashable]:
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
    df_x = df.loc[:, ~df.columns.isin([y_col])]

    features = _set_list(keep_cols)
    initial_features = df_x.columns.to_list()

    while len(initial_features) > 0:

        remaining_features = list(set(initial_features) - set(features))

        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = _sm_model(df_x=df_x[features + [new_column]], y=y)

            coef = model.params[new_column]
            pval = model.pvalues[new_column]
            # To allow for the possibility of floating point precision error:
            if abs(coef) < 1e-6:
                pval = 1.
            new_pval[new_column] = pval

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
    keep_cols: List[str] = None,
) -> List[Hashable]:
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
    df_x = df.loc[:, ~df.columns.isin([y_col])]

    features = df_x.columns.tolist()
    keep_cols = _set_list(keep_cols)
    remaining_features = list(set(features) - set(keep_cols))

    while len(remaining_features) > 0:

        model = _sm_model(df_x=df_x[features], y=y)
        pvalues: pd.Series = model.pvalues[remaining_features]
        coeffs: pd.Series = model.params[remaining_features]
        # To allow for the possibility of floating point precision error:
        pvalues.loc[coeffs.abs() < 1e-6] = 1

        max_p_value = pvalues.max()
        if max_p_value >= cutoff:
            remove_feature = pvalues.idxmax()
            features.remove(remove_feature)
            remaining_features.remove(remove_feature)
            logger.info(
                f"Removed {remove_feature} whose p-value was: {max_p_value}."
            )
        else:
            break

    remaining_features.extend(keep_cols)
    return remaining_features


def step_wise_select(
    df: pd.DataFrame,
    y_col: str,
    cutoff_in: float = 0.05,
    cutoff_out: float = 0.05,
    keep_cols: List[str] = None,
) -> List[Hashable]:
    """
    It is similar to forward selection but the difference is
    while adding a new feature it also checks the significance
    of already added features and if it finds any of the already
    selected features insignificant then it simply removes
    that particular feature through backward elimination.

    Hence, it is a combination of forward selection and backward elimination.
    """
    logger.info(
        f"Performing stepwise selection using the "
        f"forward cutoff: {cutoff_in} and backward "
        f"cutoff: {cutoff_out}."
    )

    y = df[y_col].values
    df_x = df.loc[:, ~df.columns.isin([y_col])]

    keep_cols = _set_list(keep_cols)
    features = keep_cols.copy()
    initial_features = df_x.columns.to_list()

    while len(initial_features) > 0:

        # FORWARD -----------------------------------------------------------
        remaining_features = list(set(initial_features) - set(features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = _sm_model(df_x=df_x[features + [new_column]], y=y)

            coef = model.params[new_column]
            pval = model.pvalues[new_column]
            # To allow for the possibility of floating point precision error:
            if abs(coef) < 1e-6:
                pval = 1.
            new_pval[new_column] = pval

        min_p_value = new_pval.min()
        if min_p_value < cutoff_in:
            features.append(new_pval.idxmin())
            logger.info(
                f"FORWARD PASS: Selected {features[-1]} "
                f"whose p-value is: {min_p_value}."
            )

            # BACKWARD ------------------------------------------------------
            remaining_features = list(set(features) - set(keep_cols))

            while len(remaining_features) > 0:

                model = _sm_model(df_x=df_x[features], y=y)
                pvalues: pd.Series = model.pvalues[remaining_features]
                coeffs: pd.Series = model.params[remaining_features]
                # To allow for the possibility of floating point precision error:
                pvalues.loc[coeffs.abs() < 1e-6] = 1

                max_pvalue = pvalues.max()
                if max_pvalue >= cutoff_out:
                    remove_feature = pvalues.idxmax()
                    features.remove(remove_feature)
                    remaining_features.remove(remove_feature)
                    logger.info(
                        f"BACKWARD PASS: Removed {remove_feature} "
                        f"whose p-value was: {max_pvalue}."
                    )
                else:
                    break
        else:
            break

    return features


@seed
def random_forest(
    df: pd.DataFrame,
    y_col: str,
    cutoff: float = None,
    keep_cols: List[str] = None,
    importance_type: str = 'gain',
) -> List[str]:
    """
    Keep most important features
    (as determined by random forest)
    satisfying cutoff.

    Note: importance type can be defined as:

        * 'weight': the number of times a feature is used to split the data across all trees.
        * 'gain': the average gain across all splits the feature is used in.
        * 'cover': the average coverage across all splits the feature is used in.
        * 'total_gain': the total gain across all splits the feature is used in.
        * 'total_cover': the total coverage across all splits the feature is used in.

    WARNING: When setting cutoff, pay attention to what importance_type
        is used.
    """

    y = df[y_col].values
    keep_cols = _set_list(keep_cols)
    features = list(
        set(df.columns.tolist()) - {*keep_cols, y_col}
    )
    df_x = df.loc[:, features]

    # MODEL
    model = _rf_model(df_x=df_x, y=y)
    importance = model.get_booster().get_score(
        importance_type=importance_type, fmap=''
    )

    # SELECT BEST FEATURES
    imp_tuples = list(importance.items())
    imp_tuples.sort(key=lambda x: x[1], reverse=True)

    # Set cutoff - if none use the mean importance.
    avg_importance = sum(importance.values())/len(imp_tuples)
    cutoff = cutoff if cutoff is not None else avg_importance

    imp_features = []
    for _feature, imp in imp_tuples:
        if imp > cutoff:
            logger.info(
                f'Selecting {_feature} whose feature importance: {imp} '
                f'is greater than cutoff: {cutoff} with importance type: '
                f'{importance_type}'
            )
            imp_features.append(_feature)
        else:
            break  # since sorted in descending order

    imp_features.extend(keep_cols)  # include required cols
    return imp_features


def lasso(
        df: pd.DataFrame,
        y_col: str,
        cutoff: float = 0,
        keep_cols: List[str] = None,
        alpha: float = 0.05
) -> List[str]:
    """
    Keep weights that are larger than cutoff,
    using lasso regressions to determine weights.
    """

    logger.info(
        f"Performing lasso feature selection using "
        f"alpha: {alpha}"
    )

    y = df[y_col].values
    keep_cols = _set_list(keep_cols)
    features = list(
        set(df.columns.tolist()) - {*keep_cols, y_col}
    )
    df_x = df.loc[:, features]

    lasreg = Lasso(alpha=alpha)
    lasreg.fit(X=df_x, y=y)

    coefficients = lasreg.coef_
    importance = np.abs(coefficients)

    selected_features = []
    for i in range(len(coefficients)):
        if importance[i] > cutoff:
            selected_features.append(df_x.columns[i])

    selected_features.extend(keep_cols)  # include required cols
    return selected_features


def yx_correlation(
    df: pd.DataFrame,
    y_col: str,
    cutoff: float = 0.5,
    keep_cols: List[str] = None,
    correlation_measure: str = 'pearson',
) -> List[str]:
    """
    Keep features that have a Pearson
    correlation higher than the cutoff with
    the y_col.
    """

    if correlation_measure not in ['pearson', 'kendall', 'spearman']:
        raise ValueError("Invalid correlation measure. Must be 'pearson', 'kendall', or 'spearman'.")

    # Select all columns if keep_cols is not specified
    if not keep_cols:
        keep_cols = []

    # Compute the correlation matrix using Pearson correlation
    corr_matrix = df.drop(keep_cols, axis=1).corr(method=correlation_measure)

    # Select the features that are highly correlated with the y_col column
    corr_with_target = corr_matrix[y_col]
    high_corr_features = corr_with_target[abs(corr_with_target) > cutoff].index.tolist()
    high_corr_features.remove(y_col)

    # Add back any columns specified in keep_cols
    high_corr_features += keep_cols

    return high_corr_features


def xs_correlation(
    df: pd.DataFrame,
    y_col: str,
    cutoff: float = 0.80,
    keep_cols: List[str] = None,
    correlation_measure: str = 'pearson'
) -> List[str]:
    """
    Remove features that are highly correlated with each other
    using the specified correlation measure.

    Returns a list of the remaining (not correlated) features.
    """

    if correlation_measure not in ['pearson', 'kendall', 'spearman']:
        raise ValueError("Invalid correlation measure. Must be 'pearson', 'kendall', or 'spearman'.")

    # Compute the correlation matrix using Pearson correlation
    corr_matrix = df.drop(y_col, axis=1).corr(method=correlation_measure)

    # Find pairs of variables with a correlation coefficient higher than the cutoff
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > cutoff:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

    # Determine which variable to drop from each high-correlation pair
    to_drop = set()
    for col1, col2 in high_corr_pairs:
        if col1 not in to_drop and col2 not in to_drop:
            corr_with_target1 = abs(df[col1].corr(df[y_col], method=correlation_measure))
            corr_with_target2 = abs(df[col2].corr(df[y_col], method=correlation_measure))
            if corr_with_target1 > corr_with_target2:
                to_drop.add(col2)
            else:
                to_drop.add(col1)

    # Remove keep columns (if present) from the list of columns to be dropped
    if keep_cols:
        to_drop = to_drop - set(keep_cols)

    # Return the remaining feature names without the Y column
    return list(set(df.columns) - to_drop - {y_col})


def complete_correlation(
    df: pd.DataFrame,
    y_col: str,
    x_cutoff: float = 0.80,
    y_cutoff: float = 0.5,
    keep_cols: List[str] = None,
    correlation_measure: str = 'pearson'
) -> List[str]:
    """
    Combined XS and YS correlation feature check.
    """

    features = xs_correlation(
        df, y_col=y_col, cutoff=x_cutoff,
        keep_cols=keep_cols,
        correlation_measure=correlation_measure,
    )

    df_subset = df[[*features, y_col]]

    features = yx_correlation(
        df_subset, y_col=y_col,
        cutoff=y_cutoff,
        keep_cols=keep_cols,
        correlation_measure=correlation_measure,
    )

    return features


def vif(
    df: pd.DataFrame,
    y_col: str,
    cutoff: float = 5,
    keep_cols: List[str] = None,
) -> List[str]:
    """
    Keep features that have a VIF (Variance Inflation Factor)
    smaller than cutoff.
    """

    rd = 1
    keep_cols = _set_list(keep_cols)
    max_vif_feature = 'None'  # kept as string so that while loop starts
    features = [col for col in df.columns if col != y_col]

    logger.info('\nRemoving collinear variables...')
    while max_vif_feature is not None:

        # DETERMINE VIF:
        _x_df = sm.add_constant(df[features])
        vif_values = [
            variance_inflation_factor(_x_df.values, j)
            for j in range(_x_df.shape[1])
        ]
        # start from 1: to ignore constant column
        vif_df = pd.DataFrame({'vif': vif_values[1:]}, index=features).T

        # DETERMINE MAX VIF:
        max_vif = cutoff
        max_vif_feature = None
        remaining_cols = [col for col in vif_df.columns if col not in keep_cols]
        for column in remaining_cols:
            _vif = float(vif_df[column]['vif'])
            logger.info(f"ROUND {round}: {column} vif = {vif}")

            if (_vif > max_vif) or (np.isinf(_vif)):
                max_vif = _vif
                max_vif_feature = column

        # REMOVE MAX VIF FEATURE:
        if max_vif_feature is not None:
            df = df.drop([max_vif_feature], axis=1)
            features.remove(max_vif_feature)
            logger.info(f'ROUND {rd}: {max_vif_feature} is dropped with vif {max_vif}')

        rd += 1

    return features


def mutual_info(
        df: pd.DataFrame,
        y_col: str,
        mi_cutoff: float = 0.2,
        keep_cols: Optional[List[str]] = None
) -> List[str]:
    """
    Selects features from a pandas DataFrame based on mutual information with the target variable.

    Args:
        df (pd.DataFrame): Input pandas DataFrame containing features and target variable.
        y_col (str): The name of the target variable in the DataFrame.
        mi_cutoff (float): The minimum mutual information score required to keep a feature.
        keep_cols (Optional[List[str]], optional): List of column names to always keep. Defaults to None.

    Returns:
        List[str]: A list of selected feature names.
    """

    # Separate features and y_col variable
    keep_cols = _set_list(keep_cols)
    features = [
        col for col in df.columns if col not in [*keep_cols, y_col]
    ]

    # Calculate mutual information for each feature
    mi_scores = mutual_info_classif(df[features], df[y_col])

    # Create a DataFrame with feature names and their corresponding MI scores
    mi_df = pd.DataFrame({"Feature": features, "MI_Score": mi_scores})

    # Sort features by MI scores in descending order
    mi_df = mi_df.sort_values("MI_Score", ascending=False)

    # Keep features with MI scores above the specified cutoff
    selected_features = mi_df[mi_df["MI_Score"] > mi_cutoff]["Feature"].tolist()

    # Add specified columns to always keep (if provided)
    selected_features += keep_cols

    return selected_features


def missing_ratio(
    df: pd.DataFrame,
    y_col: str,
    cutoff: float = 0.70,
    keep_cols: List[str] = None
) -> List[str]:
    """
    Remove features with a missing ratio higher
    than the specified cutoff (e.g. if more
    than 70% are null remove feature for instance)

    Returns a list of the remaining features.
    """

    # Calculate the missing ratio for each feature
    keep_cols = _set_list(keep_cols)
    features = [col for col in df.columns if col not in [*keep_cols, y_col]]
    missing_ratios = df[features].isnull().mean()

    selected_features = [
        col for col, _missing_ratio in missing_ratios.items()
        if _missing_ratio <= cutoff
    ]

    selected_features += keep_cols

    return selected_features


def variance(
    df: pd.DataFrame,
    y_col: str,
    variance_cutoff: float = 0.01,
    keep_cols: List[str] = None
) -> List[str]:
    """
    Remove features with a variance lower than the specified cutoff.

    Returns a list of the remaining features.
    """

    # Calculate the variance for each feature
    keep_cols = _set_list(keep_cols)
    features = [col for col in df.columns if col not in [*keep_cols, y_col]]
    variances = df[features].var()

    # Filter features with a variance higher than the cutoff
    selected_features = [
        col for col, _variance in variances.items()
        if _variance >= variance_cutoff
    ]

    selected_features += keep_cols

    return selected_features


class Selector(object):

    MAPPER = {
        "variance": variance,
        "missing": missing_ratio,
        "mutual_info": mutual_info,
        "rf": random_forest,
        "forward": forward_select,
        "backward": backward_eliminate,
        "step": step_wise_select,
        "lasso": lasso,
        "vif": vif,
        "complete_correlation": complete_correlation,
        "ys_correlation": yx_correlation,
        "xs_correlation": xs_correlation,
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
        cv_cutoff=None,
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
        cv_cutoff = cv_cutoff if cv_cutoff is not None else int(n_eval/2)

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
            _feature for _feature, imp in fimps if imp > cv_cutoff
        ]
        logger.info(
            f"FINAL CV SELECTION COUNTS: {fimps} | Selecting "
            f"{self._best_features}."
        )
        return self
