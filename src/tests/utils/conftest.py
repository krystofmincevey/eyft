import pytest
import pandas as pd


@pytest.fixture
def left_merge_input():

    data = [
        ['a', 10],
        ['b', 20],
        [None, 30]
    ]
    columns = ['Key', 'Price']

    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def right_merge_input():

    data = [
        ['a', 1],
        ['c', 2],
        [None, 3]
    ]
    columns = ['Key', 'EPC']

    return pd.DataFrame(data, columns=columns)
