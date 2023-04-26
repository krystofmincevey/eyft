import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal


@pytest.fixture
def fs_inputs():
    data = [
        [1.4, 0.9, 0, 5, 70, 0, 20, (1.8 * 1.4 + 0.5 * 0.9 + 4 * 0 + -2 * 5 + 0.05)],
        [1.1, 0.9, -10, -5, 140, 10, 80, (1.8 * 1.1 + 0.5 * 0.9 + 4 * -10 + -2 * -5 + 0.05)],
        [0.9, 0.60, 100, 15, 60, 20, 45, (1.8 * 0.9 + 0.5 * 0.60 + 4 * 100 + -2 * 15 + 0.05)],
        [-1.01, -1.1, 30, -45, -90, 10, 5, (1.8 * -1.01 + 0.5 * -1.1 + 4 * 30 + -2 * -45 - 0.05)],
        [-1.10, -1.5, 45, -55, 10, -5, 15, (1.8 * -1.10 + 0.5 * -1.5 + 4 * 45 + -2 * -55 - 0.05)],
        [6, 0.51, 5, 25, 40, -10, 60, (1.8 * 6 + 0.5 * 0.51 + 4 * 5 + -2 * 25 - 0.05)],
        [-0.21, -0.45, -20, 35, 15, -20, 20, (1.8 * -0.21 + 0.5 * -0.45 + 4 * -20 + -2 * 35 + 0.05)],
    ]
    columns = ['COL1', 'COL2', 'COL3', 'COL4', 'COL5', 'COL6', 'COL7', 'Y']
    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def feat_sel_inputs():
    data = [
        [0.9, 70, 20, (0.5 * 0.9 + 4 * 20)],
        [0.9, 140, 80, (0.5 * 0.9 + 4 * 80)],
        [0.60, 60, 45, (0.5 * 0.60 + 4 * 45)],
        [-1.1, -90, 5, (0.5 * -1.1 + 4 * 5)],
        [-1.5, 10, 15, (0.5 * -1.5 + 4 * 15)],
        [0.51, 40, 60, (0.5 * 0.51 + 4 * 60)],
        [-0.45, 15, 20, (0.5 * -0.45 + 4 * 20)],
    ]
    columns = ['COL1', 'COL2', 'COL3', 'Y']
    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def corr_inputs():
    data = [
        [1.8, 3.6, 122, -8.50, (1.8 + 0.005)],
        [1.6, 3.2, -101, 84, (1.6 + 0.005)],
        [-1.1, -2.2, -2.85, 5, (-1.1 - 0.005)],
        [0.7, 1.4, 0.5, -7.3, (0.7 - 0.005)],
        [-2.3, -4.6, -27, -10.2, (-2.3 - 0.005)],
        [0.2, 0.4, -0.15, 6, (0.2 + 0.005)],
        [-0.2, -0.4, -12, 4.5, (-0.2 - 0.005)],
    ]
    columns = ['COL1', 'COL2', 'COL3', 'COL4', 'Y']
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
