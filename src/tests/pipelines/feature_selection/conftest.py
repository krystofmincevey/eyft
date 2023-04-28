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


@pytest.fixture()
def high_vif_inputs():
    data = [
        [1.0, 2.0, 3.0 + random.gauss(0., 0.01), 4.0 - random.gauss(0., 0.01), 5.0],
        [2.0, 4.0, 6.0 + random.gauss(0., 0.01), 8.0 - random.gauss(0., 0.01), 10.0],
        [3.0, 6.0, 9.0 + random.gauss(0., 0.01), 12.0 + random.gauss(0., 0.01), 15.0],
        [4.0, 8.0, 12.0 - random.gauss(0., 0.01), 16.0 + random.gauss(0., 0.01), 20.0],
        [5.0, 10.0, 15.0 - random.gauss(0., 0.01), 20.0 + random.gauss(0., 0.01), 25.0]
    ]
    columns = ['COL1', 'COL2', 'COL3', 'COL4', 'Y']
    return pd.DataFrame(data, columns=columns)
