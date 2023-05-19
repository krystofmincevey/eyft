import numpy as np
import pandas as pd

from typing import Dict, Any, Tuple
from itertools import combinations

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV

from ...utils.models import seed, is_categorical
from ..model_selection import logger


# Create a custom transformer for feature selection
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_mask=None):
        self.feature_mask = feature_mask

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray, y=None):
        return X[:, self.feature_mask]


class SklearnModeler(object):

    _order_col: str
    _target_col: str
    _is_categorical: bool
    REGRESSION_MODELS = {
        "LinearRegression": (
            LinearRegression(),
            {}  # No hyperparameters to tune for Linear Regression
        ), 'DecisionTreeRegressor': (
            DecisionTreeRegressor(),
            {'max_depth': [None, 5, 10]}
        ), 'RandomForestRegressor': (
            RandomForestRegressor(),
            {'n_estimators': [100, 200, 300]}
        ),
    }
    CLASSIFICATION_MODELS = {
        'LogisticRegression': (
            LogisticRegression(),
            {'C': [0.1, 1, 10]}
        ), 'DecisionTreeClassifier': (
            DecisionTreeClassifier(),
            {'max_depth': [None, 5, 10]}
        ), 'RandomForestClassifier': (
            RandomForestClassifier(),
            {'n_estimators': [100, 200, 300]}
        ),
    }

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        is_feature_select: bool = False,
        order_col: str = None,
    ):
        """
        Model selection class capable of performing
        hyper-parameter tuning and model selection.
        When is_feature_select is True, features are
        included in the hyper-parameter search.

        WARNING: when an order column is set
        we assume that the data is a timeseries.
        This impacts the train procedure - as
        normal CV cannot be used on a time series.
        Note that the order_column is not included
        in the features used to train the models/pipelines.
        """

        for col in [target_col, order_col]:
            if col is not None:
                assert col in df.columns, \
                    f"{col} not in input data columns: {df.columns}"

        self._df = df
        self._is_feature_select = is_feature_select
        self.set_target(target_col)
        self.set_order_col(order_col)
        self._search_results = None

    def set_target(self, target_col: str):
        """
        Sets both self._target_col and self._is_categorical
        """
        assert target_col in self._df.columns, \
            f"{target_col} not in input data columns: {self._df.columns}"

        self._target_col = target_col
        if is_categorical(self._df[self._target_col]):
            self._is_categorical = True
        else:
            self._is_categorical = False

    def set_order_col(self, order_col: str = None):
        """
        Adds support for timeseries where CV split
        may require ordering.
        """
        if order_col is not None:

            assert order_col in self._df.columns, \
                f"{order_col} not in input data columns: {self._df.columns}"

            self._df.sort_values(by=[order_col], inplace=True)

        self._order_col = order_col

    def get_search_results(self) -> Dict[str, Tuple[Any, float, Dict[str, Any]]]:
        return self._search_results

    def get_best_results(self) -> Tuple[Any, float, Dict[str, Any]]:
        """
        Assumes (as with sklearn models) that a low score
        is bad. Eg. negative mean squared error where results
        closer to 0 are better than more negative results.
        """

        if self._search_results is None:
            raise KeyError(
                f"You must run search prior to fetching "
                f"best search resutls"
            )
        best_model, best_score, best_params = None, -1e10, None
        for _model, _score, _params in self._search_results.values():
            if _score > best_score:
                best_model, best_score, best_params = _model, _score, _params
        return best_model, best_score, best_params

    def search(
        self,
        model2params: Dict[str, Dict[str, Any]] = None,
        scoring: Any = None,
        n_iter: int = 10,
        cv: int = 5,
    ):
        """
        Searches supported models (either regression
        or classification) and spits out the best one.
        """
        if model2params is None:

            if self._is_categorical:
                model2params = {
                    model_name: params
                    for model_name, (_, params) in self.CLASSIFICATION_MODELS
                }
            else:
                model2params = {
                    model_name: params
                    for model_name, (_, params) in self.REGRESSION_MODELS
                }

        self._search(
            model2params,
            scoring=scoring,
            n_iter=n_iter,
            cv=cv,
        )
        return self

    def train(
            self, model_name: str,
            parameters: Dict[str, Any] = None,
            scoring: str = None,
            n_iter: int = 10,
            cv: int = 5,
    ) -> Tuple[Any, float, Dict[str, Any]]:
        if self._is_categorical:
            best_model, best_score, best_params = self.train_classification_model(
                model_name=model_name,
                parameters=parameters,
                scoring=scoring,
                n_iter=n_iter,
                cv=cv,
            )
        else:  # regression
            best_model, best_score, best_params = self.train_regression_model(
                model_name=model_name,
                parameters=parameters,
                scoring=scoring,
                n_iter=n_iter,
                cv=cv,
            )
        return best_model, best_score, best_params

    def train_regression_model(
        self, model_name: str,
        parameters: Dict[str, Any] = None,
        scoring: str = None,
        n_iter: int = 10,
        cv: int = 5,
    ) -> Tuple[Any, float, Dict[str, Any]]:
        """
        Train and evaluate a regression model.

        Returns: best_model, best_score, best_params
        """

        if scoring is None:
            scoring = 'neg_mean_squared_error'

        return self._train(
            model_name=model_name,
            supported_models=self.REGRESSION_MODELS,
            scoring=scoring,
            parameters=parameters,
            n_iter=n_iter,
            cv=cv,
        )

    def train_classification_model(
        self, model_name: str,
        parameters: Dict[str, Any] = None,
        scoring: str = None,
        n_iter: int = 10,
        cv: int = 5,
    ) -> Tuple[Any, float, Dict[str, Any]]:
        """
        Train and evaluate a classification model.

        Returns: best_model, best_score, best_params
        """

        if scoring is None:
            scoring = 'accuracy'

        return self._train(
            model_name=model_name,
            supported_models=self.CLASSIFICATION_MODELS,
            scoring=scoring,
            parameters=parameters,
            n_iter=n_iter,
            cv=cv,
        )

    def _search(
        self,
        model2params: Dict[str, Dict[str, Any]],
        scoring: Any = None,
        n_iter: int = 10,
        cv: int = 5,
    ):
        """
        Searches supported models (either regression
        or classification) and saves results in self._search_results.
        """
        results = {}
        for model_name, params in model2params.items():
            results[model_name] = self.train(
                model_name=model_name,
                parameters=params,
                scoring=scoring,
                n_iter=n_iter,
                cv=cv,
            )
        self._search_results = results
        return self

    @seed
    def _train(
        self, model_name: str,
        supported_models: Dict[str, Tuple[Any, Dict[str, Any]]],
        scoring: Any,
        parameters: Dict[str, Any] = None,
        n_iter: int = 10,
        cv: int = 5,
    ) -> Tuple[Any, float, Dict[str, Any]]:
        """
        Train and evaluate a model. Note, supports
        time series CV when self._order_col is set, and also
        feature selection when self._is_feature_select = True.

        Returns: best_model, best_score, best_params
        """
        if model_name in supported_models:
            model, default_parameters = supported_models[model_name]
        else:
            raise ValueError(
                f"{model_name} not among supported models: "
                f"{supported_models.keys()}"
            )

        # Get defaults if params or scoring not set
        if parameters is None:
            parameters = default_parameters

        # Hyperparameter tuning using RandomizedSearchCV
        key_model = "model"
        key_f_selection = "feature_selection"
        features = [
            col for col in self._df.columns
            if col not in [self._target_col, self._order_col]
        ]

        if self._order_col is not None:
            cv = TimeSeriesSplit(n_splits=cv)  # avoid data leakage with time series

        if self._is_feature_select:
            # Update model; create a pipeline with feature selection and model:
            model = Pipeline([
                (key_f_selection, FeatureSelector()),
                (key_model, model)
            ])

            # Generate all possible feature combinations
            all_feature_combinations = []
            for r in range(1, len(features) + 1):
                all_feature_combinations.extend(combinations(range(len(features)), r=r))

            # Update parameters to work with pipeline:
            parameters = {f"{key_model}__{key}": value for key, value in parameters.items()}
            parameters.update({
                f'{key_f_selection}__feature_mask': all_feature_combinations
            })

        logger.info(
            f'Performing model selection for {model_name} '
            f'over params: {parameters}. '
            f'The scoring functions used is: {scoring}.'
        )
        randomized_search = RandomizedSearchCV(
            model, parameters, scoring=scoring, cv=cv, n_iter=n_iter
        )

        randomized_search.fit(
            self._df[features].values,
            self._df[self._target_col].values
        )

        best_model = randomized_search.best_estimator_
        best_score = randomized_search.best_score_
        best_params = randomized_search.best_params_

        if self._is_feature_select:
            best_features_mask = best_params.pop(
                f'{key_f_selection}__feature_mask'
            )
            best_features = [features[i] for i in best_features_mask]

            # REMOVE pipeline PREFIX AND ADD FEATURES
            best_params = {
                key.replace(f'{key_model}__', ''): value for key, value in best_params.items()
            }
            best_params['features'] = best_features

        logger.info(
            f"For {model_name} the best params were found to "
            f"be {best_params} yielding a score ({scoring}) of "
            f"{best_score}."
        )
        return best_model, best_score, best_params
