import random

import pytest
import pandas as pd


N_VALS = 10000


@pytest.fixture
def fs_inputs():
    data = [[random.gauss(0., 1) for _ in range(7)] for __ in range(N_VALS)]
    columns = ['COL1', 'COL2', 'COL3', 'COL4', 'COL5', 'COL6', 'COL7']
    df = pd.DataFrame(data, columns=columns)
    df['Y'] = 2*df['COL1'] + 3*df['COL2'] + 1.5*df['COL4'] + random.gauss(0., 0.001)
    return df


@pytest.fixture
def rf_inputs():
    data = [
        [0, 0 + random.gauss(0., 0.01), random.gauss(0., 0.5), 1, 0],
        [1, -1 - random.gauss(0., 0.01), 1, random.gauss(0., 0.5), 0],
        [2, -2 + random.gauss(0., 0.01), random.gauss(0., 0.5), random.gauss(0., 0.5), 0],
        [3, -3 - random.gauss(0., 0.01), random.gauss(0., 0.5), random.gauss(0., 0.5), 1],
        [4, -4 + random.gauss(0., 0.01), 1, 1, 1],
        [5, -5 - random.gauss(0., 0.01), random.gauss(0., 0.5), random.gauss(0., 0.5), 1],
        [6, -6 + random.gauss(0., 0.01), random.gauss(0., 0.5), random.gauss(0., 0.5), 2],
        [7, -7 - random.gauss(0., 0.01), random.gauss(0., 0.5), random.gauss(0., 0.5), 2],
        [10, -10 + random.gauss(0., 0.01), 1, random.gauss(0., 0.5), 3],
        [11, -11 - random.gauss(0., 0.01), random.gauss(0., 0.5), 1, 3],
    ]
    columns = ['COL1', 'COL2', 'COL3', 'COL4', 'Y']
    df = pd.DataFrame(data, columns=columns)
    return df


@pytest.fixture
def corr_inputs_2():
    data = [
        [
            x + random.gauss(0., 0.01),
            2*(x + random.gauss(0., 0.01)) + random.gauss(0., 0.01),
            random.gauss(0., 0.5),
            x + random.gauss(0., 1)
        ]
        for x in range(N_VALS)
    ]
    columns = ['COL1', 'COL2', 'COL3', 'Y']
    return pd.DataFrame(data, columns=columns)
