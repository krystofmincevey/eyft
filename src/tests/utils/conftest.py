import pytest
import pandas as pd


@pytest.fixture
def inputs():

    data = [
        [3.25e+05, 3, None, 'Hendrik De Braekeleerlaan 68', 2.59e+02],
        [1.74e+05, 2, 2, 'Antwerpsesteenweg 50', 6.44e+02],
        [1.74e+05, 2, 4, 'Antwerpsesteenweg 5', 6.44e+02],
        [1.99e+05, 2, 2, 'Hoevelei 194', 2.06e+02],
    ]
    columns = ['Price', 'Bedrooms', 'Facades', 'Street', 'EPC']

    return pd.DataFrame(data, columns=columns)
