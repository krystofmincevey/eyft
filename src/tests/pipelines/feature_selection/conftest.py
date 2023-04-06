import pytest
import pandas as pd


@pytest.fixture
def processed_inputs():

    data = [
        [1, 0.9, 0, ...],
        [1, 0.9, -10, ...],
        [0, 0.001, 100, ...],
        [-1, -1.1, 30, ...],
        [],
        [],
        [],
    ]
    columns = [
        'Price',  # y_col
        'COL1', 'COL2', 'COL3', "COL4", "COL5", "COL6", "COL7"
    ]

    return pd.DataFrame(data, columns=columns)

# TESTING PRINCIPLES
# y = mx1 + m2x2 + c + error
#
# choose some m   m1 = 1.1   m2 = 1.5
#
# x3, x4, x5, x6   RANDOM NUMBERS so should not be selected
#
# USE x1, x2, .... x6 as inputs and y as target.
# FROM ABOVE YOU SHOULD KNOW THAT x1 and x2 get selected.

