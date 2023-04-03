import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from eyft.pipelines.data_processing.processor import (
    mean_impute, mode_impute, z_normalise, min_max_scale,
    cap_3std
)


class TestCap3std(object):

    def test_values(self, epc_input_2):

        df_actual = cap_3std(epc_input_2, col='Bedrooms', mean=1, stdev=1)['df']

        df_expected = pd.DataFrame(
            data=[
                [3.25e+05, 1, None, 'Hendrik De Braekeleerlaan 68', 2.59e+02, 0, 1],
                [1.74e+05, 4, 2, 'Antwerpsesteenweg 50', 6.44e+02, 0, 1],
                [1.74e+05, 4, 4, 'Antwerpsesteenweg 5', 6.44e+02, 0, 1],
                [1.99e+05, 4, 2, 'Hoevelei 194', 2.06e+02, 0, 1],
                [2.39e+05, 0, 1, 'Leon Gilliotlaan 40', 2.29e+02, 0, -1],
                [2.85e+05, 1, None, 'De Cranelei 9', 4.40e+01, 0, -1],
                [3.69e+05, 1, 2, 'Kapellestraat 159', 0.00e+00, 0, -1],
                [3.69e+05, 0, 3, 'Kapellestraat 159 B', 0.00e+00, 0, -1],
            ]
            ,
            columns=['Price', 'Bedrooms', 'Facades', 'Street', 'EPC', 'Zeros', 'Ones']
        )

        assert_frame_equal(df_actual, df_expected, check_dtype=False)


class TestZNormalise(object):

    def test_z_normalise(self, epc_input):
        actual = z_normalise(
            epc_input, col='Bedrooms', mean=2, stdev=1
        )['df']['Bedrooms']
        expected = pd.Series([1, 0, 0, 0, 0, 1, -1, -1], name='Bedrooms')
        assert_series_equal(actual, expected, check_dtype=False)

    def test_z_normalise_zeros(self, epc_input):
        actual = z_normalise(
            epc_input, col='Zeros'
        )['df']['Zeros']
        expected = pd.Series([0, 0, 0, 0, 0, 0, 0, 0], name='Zeros')
        assert_series_equal(actual, expected, check_dtype=False)

    def test_z_normalise_ones(self, epc_input):
        actual = z_normalise(
            epc_input, col='Ones'
        )['df']['Ones']
        expected = pd.Series([1, 1, 1, 1, -1, -1, -1, -1], name='Ones')
        assert_series_equal(actual, expected, check_dtype=False)


class TestMinMaxScale(object):
    def test_min_max_scale(self, epc_input):
        actual = min_max_scale(
            epc_input, col='Bedrooms')['df']['Bedrooms']
        expected = pd.Series([1, 1/2, 1/2, 1/2, 1/2, 1, 0, 0], name='Bedrooms')
        assert_series_equal(actual, expected, check_dtype=False)

    def test_min_max_scale_zeros(self, epc_input):
        actual = min_max_scale(
            epc_input, col='Zeros')['df']['Zeros']
        expected = pd.Series([0, 0, 0, 0, 0, 0, 0, 0], name='Zeros')
        assert_series_equal(actual, expected, check_dtype=False)


def test_mean_impute(epc_input):
    actual = mean_impute(epc_input, col='Facades', mean=2)['df']['Facades']
    expected = pd.Series([2, 2, 4, 2, 1, 2, 2, 3], name='Facades')
    assert_series_equal(actual, expected, check_dtype=False)


class TestModeImpute(object):

    def test_mode(self, epc_input):
        mode = mode_impute(df=epc_input, col='Facades', mode=None)['mode']
        assert mode == 2

    def test_impute(self, epc_input):
        df_actual = mode_impute(df=epc_input, col='Facades', mode=None)['df']

        df_expected = pd.DataFrame(
            data=[
                [3.25e+05, 3, 2, 'Hendrik De Braekeleerlaan 68', 2.59e+02, 0, 1],
                [1.74e+05, 2, 2, 'Antwerpsesteenweg 50', 6.44e+02, 0, 1],
                [1.74e+05, 2, 4, 'Antwerpsesteenweg 5', 6.44e+02, 0, 1],
                [1.99e+05, 2, 2, 'Hoevelei 194', 2.06e+02, 0, 1],
                [2.39e+05, 2, 1, 'Leon Gilliotlaan 40', 2.29e+02, 0, -1],
                [2.85e+05, 3, 2, 'De Cranelei 9', 4.40e+01, 0, -1],
                [3.69e+05, 1, 2, 'Kapellestraat 159', 0.00e+00, 0, -1],
                [3.69e+05, 1, 3, 'Kapellestraat 159 B', 0.00e+00, 0, -1],
            ],
            columns=['Price', 'Bedrooms', 'Facades', 'Street', 'EPC', 'Zeros', 'Ones']
        )

        assert_frame_equal(df_actual, df_expected, check_dtype=False)

    def test_impute_with_mode(self, epc_input):
        actual = mode_impute(df=epc_input, col='Facades', mode=10)['df']['Facades']
        expected = pd.Series([10, 2, 4, 2, 1, 10, 2, 3], name='Facades')
        assert_series_equal(actual, expected, check_dtype=False)
