import pytest
from sklearn.metrics import make_scorer, accuracy_score


def test_init(modeler):
    assert modeler is not None
    assert modeler._target_col == 'target'
    assert isinstance(modeler._is_categorical, bool)


def test_set_target(modeler):
    modeler.set_target('feature_1')
    assert modeler._target_col == 'feature_1'


def test_set_order_col(modeler):
    modeler.set_order_col('feature_1')
    assert modeler._order_col == 'feature_1'


def test_search_results_before_search(modeler):
    with pytest.raises(KeyError):
        modeler.get_best_results()


def test_search_results_after_search(modeler):
    modeler.search(
        model2params={'LogisticRegression': {'C': [0.1]}},
        scoring=make_scorer(accuracy_score),
        n_iter=1,
        cv=2
    )
    search_results = modeler.get_search_results()
    assert search_results is not None
    assert isinstance(search_results, dict)
    assert 'LogisticRegression' in search_results.keys()


def test_train(modeler):
    best_model, best_score, best_params = modeler.train(
        model_name='LogisticRegression',
        parameters={'C': [0.1]},
        scoring='accuracy',
        n_iter=1,
        cv=2
    )
    assert best_model is not None
    assert isinstance(best_score, float)
    assert isinstance(best_params, dict)


def test_train_unsupported_model(modeler):
    with pytest.raises(ValueError):
        modeler.train(
            model_name='UnsupportedModel',
            parameters={'C': [0.1]},
            scoring='accuracy',
            n_iter=1,
            cv=2
        )


def test_train_with_feature_selection(iris_modeler):
    best_model, best_score, best_params = iris_modeler.train(
        model_name='LogisticRegression',
        parameters={'C': [0.1]},
        scoring='accuracy',
        n_iter=1,
        cv=2
    )
    assert best_model is not None
    assert isinstance(best_score, float)
    assert isinstance(best_params, dict)
    assert 'features' in best_params


def test_init_2(timeseries_modeler):
    assert timeseries_modeler is not None
    assert timeseries_modeler._target_col == 'target'
    assert isinstance(timeseries_modeler._is_categorical, bool)
    assert timeseries_modeler._is_categorical  # True


def test_train_with_timeseries(timeseries_modeler):
    best_model, best_score, best_params = timeseries_modeler.train(
        model_name='LogisticRegression',
        parameters={'C': [1]},
        scoring='accuracy',
        n_iter=1,
        cv=2
    )
    assert best_model is not None
    assert isinstance(best_score, float)
    assert isinstance(best_params, dict)


def test_train_classification(timeseries_modeler):
    best_model, best_score, best_params = timeseries_modeler.train_classification_model(
        model_name='LogisticRegression',
        parameters={'C': [0.1]},
        scoring='accuracy',
        n_iter=1,
        cv=2
    )
    assert best_model is not None
    assert isinstance(best_score, float)
    assert isinstance(best_params, dict)


def test_train_regression(timeseries_modeler):
    best_model, best_score, best_params = timeseries_modeler.train_regression_model(
        model_name='LinearRegression',
        parameters={},
        scoring='neg_mean_squared_error',
        n_iter=1,
        cv=2
    )
    assert best_model is not None
    assert isinstance(best_score, float)
    assert isinstance(best_params, dict)


def test_search_with_different_models(timeseries_modeler):
    timeseries_modeler.search(
        model2params={
            'LogisticRegression': {'C': [0.1]},
            'DecisionTreeClassifier': {'max_depth': [None, 5]}
        },
        scoring=make_scorer(accuracy_score),
        n_iter=1,
        cv=2
    )
    search_results = timeseries_modeler.get_search_results()
    assert search_results is not None
    assert isinstance(search_results, dict)
    assert 'LogisticRegression' in search_results.keys()
    assert 'DecisionTreeClassifier' in search_results.keys()
