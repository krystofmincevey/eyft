import pytest
import pandas as pd


@pytest.fixture
def epc_input():

    data = [
        [3.25e+05, 3, None, 'Hendrik De Braekeleerlaan 68', 2.59e+02, 0, 1],
        [1.74e+05, 2, 2, 'Antwerpsesteenweg 50', 6.44e+02, 0, 1],
        [1.74e+05, 2, 4, 'Antwerpsesteenweg 5', 6.44e+02, 0, 1],
        [1.99e+05, 2, 2, 'Hoevelei 194', 2.06e+02, 0, 1],
        [2.39e+05, 2, 1, 'Leon Gilliotlaan 40', 2.29e+02, 0, -1],
        [2.85e+05, 3, None, 'De Cranelei 9',  4.40e+01, 0, -1],
        [3.69e+05, 1, 2, 'Kapellestraat 159', 0.00e+00, 0, -1],
        [3.69e+05, 1, 3, 'Kapellestraat 159 B', 0.00e+00, 0, -1],
    ]
    columns = ['Price', 'Bedrooms', 'Facades', 'Street', 'EPC', 'Zeros', 'Ones']

    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def epc_input_2():

    data = [
        [3.25e+05, 1, None, 'Hendrik De Braekeleerlaan 68', 2.59e+02, 0, 1],
        [1.74e+05, 5, 2, 'Antwerpsesteenweg 50', 6.44e+02, 0, 1],
        [1.74e+05, 5, 4, 'Antwerpsesteenweg 5', 6.44e+02, 0, 1],
        [1.99e+05, 5, 2, 'Hoevelei 194', 2.06e+02, 0, 1],
        [2.39e+05, 0, 1, 'Leon Gilliotlaan 40', 2.29e+02, 0, -1],
        [2.85e+05, 1, None, 'De Cranelei 9',  4.40e+01, 0, -1],
        [3.69e+05, 1, 2, 'Kapellestraat 159', 0.00e+00, 0, -1],
        [3.69e+05, 0, 3, 'Kapellestraat 159 B', 0.00e+00, 0, -1],
    ]
    columns = ['Price', 'Bedrooms', 'Facades', 'Street', 'EPC', 'Zeros', 'Ones']

    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def epc_input_3():

    data = [
        [3.25e+05, 1, None],
        [1.74e+05, 5, 'a'],
        [1.74e+05, 5, 'b'],
        [1.74e+05, None, 'b'],
    ]
    columns = ['Price', 'Bedrooms', 'EPC']

    return pd.DataFrame(data, columns=columns)
