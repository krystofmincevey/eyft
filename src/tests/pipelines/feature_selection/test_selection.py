import pandas as pd
from pandas.testing import assert_frame_equal
from unittest import TestCase

from eyft.pipelines.feature_selection.selector import (
    forward_select, backward_eliminate,
    step_wise_select, random_forest, lasso, pearson_xy, vif,
    pearson_xs, pearson
)


class TestForwardSelect(object):

    def test_values(self, fs_inputs):
        df_actual = set(forward_select(fs_inputs, y_col='Y'))

        df_expected = {'COL1', 'COL2', 'COL4'}
        assert df_actual == df_expected

    def test_values_with_exclusion(self, fs_inputs):
        df_actual = set(forward_select(fs_inputs, y_col='Y', exclude_cols=['COL5']))

        df_expected = {'COL1', 'COL2', 'COL4', 'COL5'}
        assert df_actual == df_expected


class TestBackwardEliminate(object):

    def test_values(self, fs_inputs):
        df_actual = set(backward_eliminate(fs_inputs, y_col='Y'))

        df_expected = {'COL1', 'COL2', 'COL4'}
        assert df_actual == df_expected

    def test_values_with_exclusion(self, fs_inputs):
        df_actual = set(backward_eliminate(fs_inputs, y_col='Y', exclude_cols=['COL5']))

        df_expected = {'COL1', 'COL2', 'COL4', 'COL5'}
        assert df_actual == df_expected


class TestStepWiseSelect(object):

    def test_values(self, fs_inputs):
        df_actual = set(step_wise_select(fs_inputs, y_col='Y'))

        df_expected = {'COL1', 'COL2', 'COL4'}
        assert df_actual == df_expected

    def test_values_with_exclusion(self, fs_inputs):
        df_actual = set(
            step_wise_select(
                fs_inputs, y_col='Y', exclude_cols=['COL5'],
            )
        )

        df_expected = {'COL1', 'COL2', 'COL4', 'COL5'}
        assert df_actual == df_expected


class TestRandomForest(object):

    def test_values(self, rf_inputs):
        df_actual = set(
            random_forest(
                rf_inputs, y_col='Y',
            )
        )

        df_expected = {'COL1', 'COL2'}
        assert df_actual == df_expected

    def test_values_with_exclusion(self, rf_inputs):
        df_actual = set(
            random_forest(
                rf_inputs, y_col='Y',
                exclude_cols=['COL3'],
                importance_type='total_gain',
                cutoff=10,
            )
        )

        df_expected = {'COL1', 'COL2', 'COL3'}
        assert df_actual == df_expected


class TestLasso(object):

    def test_values_l(self, feat_sel_inputs):
        df_actual = set(lasso(feat_sel_inputs, 'Y', cutoff=0.1))

        df_expected = {'COL1', 'COL3'}

        assert df_actual == df_expected


class TestPearsonxy(object):

    def test_values(self, corr_inputs):
        df_actual = set(pearson_xy(corr_inputs, y_col='Y'))

        df_expected = {'COL1', 'COL2'}

        assert df_actual == df_expected

    def test_values2(self, corr_inputs):
        df_actual = set(pearson_xy(corr_inputs, y_col='Y', exclude_cols=['COL3']))

        df_expected = {'COL1', 'COL2', 'COL3'}

        assert df_actual == df_expected


class TestMulticollie(object):

    def test_values(self, corr_inputs):
        df_actual = set(pearson_xs(corr_inputs, y_col='Y'))

        df_expected = {'COL1', 'COL3', 'COL4'}

        assert df_actual == df_expected

    def test_values2(self, corr_inputs):
        df_actual = set(pearson_xs(corr_inputs, y_col='Y', exclude_cols=['COL2']))

        df_expected = {'COL1', 'COL2', 'COL3', 'COL4'}

        assert df_actual == df_expected


class TestPearson(object):
    def test_values(self, corr_inputs):
        df_actual = set(pearson(corr_inputs, y_col='Y'))

        df_expected = {'COL1'}

        assert df_actual == df_expected
    def test_values2(self, corr_inputs):
        df_actual = set(pearson(corr_inputs, y_col='Y', exclude_cols=['COL3']))

        df_expected = {'COL1', 'COL3'}

        assert df_actual == df_expected


class TestVIF(object):

    def test_values(self, corr_inputs_2):
        df_actual = set(vif(corr_inputs_2, y_col='Y', exclude_cols=['COL1']))

        df_expected = {'COL1', 'COL3'}
        assert df_actual == df_expected

    def test_values2(self, high_vif_inputs):
        df_actual = set(vif(high_vif_inputs, y_col='Y'))

        df_expected = {'COL4'}

        assert df_actual == df_expected
