import pytest
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris

from eyft.pipelines.model_selection.modeler import SklearnModeler


@pytest.fixture
def modeler():
    df = pd.DataFrame({
        'feature_1': np.random.rand(100),
        'feature_2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })
    return SklearnModeler(df, 'target')


@pytest.fixture
def iris_modeler():
    iris = load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['target'])
    return SklearnModeler(df, 'target', is_feature_select=True)


@pytest.fixture
def timeseries_modeler():
    dates = pd.date_range(start='1/1/2020', periods=100)
    df = pd.DataFrame({
        'order_col': dates,
        'feature_1': np.random.rand(100),
        'feature_2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100),
    })
    return SklearnModeler(df, 'target', order_col='order_col')
