import pytest
import pandas as pd


@pytest.fixture
def processed_inputs():

    data = [
        [300, 3, 2, 2.59e+02],
        [200, 2, 2, 6.44e+02],
        [400, 2, 4, 6.44e+02],
        [50, 2, 2, 2.06e+02],
        [500, 2, 1, 2.29e+02],
        [300, 3, 2, 4.40e+01],
        [1000, 1, 2, 0.00e+00],
        [2000, 1, 4, 0.00e+00],
    ]
    columns = ['Price', 'Bedrooms', 'Facades', 'EPC']

    return pd.DataFrame(data, columns=columns)
