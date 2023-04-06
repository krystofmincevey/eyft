import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import numpy as np

from eyft.pipelines.feature_engineering.transform import (
    Transform, log_transform, nan_dummies
)


class TestTransform(object):

    def test_multiply_by(self, processed_inputs):
        trsf = Transform(df=processed_inputs)
        trsf.transform({
            'Price': ['Multiply_by Bedrooms', 'Divide_by Facades'],
        })
        df_actual = trsf.get_df()

        df_expected = pd.DataFrame(
            data=[
                [300, 3, 2, 2.59e+02, 900, 150],
                [200, 2, 2, 6.44e+02, 400, 100],
                [400, 2, 4, 6.44e+02, 800, 100],
                [50, 2, 2, 2.06e+02, 100, 25],
                [500, 2, 1, 2.29e+02, 1000, 500],
                [300, 3, 2, 4.40e+01, 900, 150],
                [1000, 1, 2, 0.00e+00, 1000, 500],
                [2000, 1, 4, 0.00e+00, 2000, 500],
            ],
            columns=[
                'Price', 'Bedrooms', 'Facades', 'EPC',
                'Price_mult_by_Bedrooms', 'Price_div_by_Facades',
            ]
        )

        assert_frame_equal(df_actual, df_expected, check_dtype=False)


class TestLogTransform(object):
    def test_log_transform(self, processed_inputs):
        df_actual = log_transform(df=processed_inputs, col='Price')

        df_expected = pd.DataFrame(
            data=[
                [300, 3, 2, 2.59e+02, np.log(300)],
                [200, 2, 2, 6.44e+02, np.log(200)],
                [400, 2, 4, 6.44e+02, np.log(400)],
                [50, 2, 2, 2.06e+02, np.log(50)],
                [500, 2, 1, 2.29e+02, np.log(500)],
                [300, 3, 2, 4.40e+01, np.log(300)],
                [1000, 1, 2, 0.00e+00, np.log(1000)],
                [2000, 1, 4, 0.00e+00, np.log(2000)],
            ],
            columns=['Price', 'Bedrooms', 'Facades', 'EPC', 'log_Price']
        )

        assert_frame_equal(df_actual['df'], df_expected, check_dtype=False)


class TestNaNDummies(object):
    def test_dummies(self, processed_inputs):
        df_actual = nan_dummies(df=processed_inputs, col='Bedrooms', na_flag=True)
        df_expected = pd.DataFrame(
            data=[
                [300, 3, 2, 2.59e+02, 0, 0, 1],
                [200, 2, 2, 6.44e+02, 0, 1, 0],
                [400, 2, 4, 6.44e+02, 0, 1, 0],
                [50, 2, 2, 2.06e+02, 0, 1, 0],
                [500, 2, 1, 2.29e+02, 0, 1, 0],
                [300, 3, 2, 4.40e+01, 0, 0, 1],
                [1000, 1, 2, 0.00e+00, 1, 0, 0],
                [2000, 1, 4, 0.00e+00, 1, 0, 0],
            ],
            columns=['Price', 'Bedrooms', 'Facades', 'EPC', 'missing_1', 'missing_2', 'missing_3']
        )
        assert_frame_equal(df_actual, df_expected, check_dtype=False)
