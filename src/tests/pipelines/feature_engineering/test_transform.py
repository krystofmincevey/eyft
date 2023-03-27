import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from eyft.pipelines.feature_engineering.transform import (
    Transform
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
