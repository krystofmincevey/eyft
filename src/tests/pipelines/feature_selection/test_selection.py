import pandas as pd
from pandas.testing import assert_frame_equal
from unittest import TestCase


from eyft.pipelines.feature_selection.selector import (
     _sm_model, forward_select, backward_eliminate,
     step_wise_select, random_forrest, lasso, pearson, vif, remove_multicollinearity
)
from tests.pipelines.feature_selection.conftest import fs_inputs, corr_inputs
from tests.pipelines.feature_selection.conftest import feat_sel_inputs


# class TestSelect(object):
    #def test_values_ts(self, fs_inputs):
     #   df_actual = select(fs_inputs, cols=['COL1', 'COL2', 'COL3', 'COL4', 'COL5', 'COL6', 'COL7'])

     #   df_expected =

#class TestSMModel(object):
   # def test_values_sm(self, fs_inputs):
       # df_actual = _sm_model(fs_inputs)

        #df_expected =
class TestForwardSelect(object):
    def test_values_fs(self, fs_inputs):
        df_actual = set(forward_select(fs_inputs, y_col='Y'))

        df_expected = {'COL1', 'COL2', 'COL3', 'COL4'}
        assert df_actual == df_expected

class TestForwardSelectEX(object):
    def test_values_fs(self, fs_inputs):
        df_actual = set(forward_select(fs_inputs, y_col='Y', exclude_cols=['COL5']))

        df_expected = {'COL1', 'COL2', 'COL3', 'COL4', 'COL5'}
        assert df_actual == df_expected

class TestBackwardEliminate(object):
    def test_values_be(self, fs_inputs):
        df_actual = set(backward_eliminate(fs_inputs, y_col='Y'))

        df_expected = {'COL1', 'COL2', 'COL3', 'COL4'}
        assert df_actual == df_expected

class TestBackwardEliminateEX(object):
    def test_values_be(self, fs_inputs):
        df_actual = set(backward_eliminate(fs_inputs, y_col='Y', exclude_cols=['COL5']))

        df_expected = {'COL1', 'COL2', 'COL3', 'COL4', 'COL5'}
        assert df_actual == df_expected

class TestStepWiseSelect(object):
    def test_values_sws(self, fs_inputs):
        df_actual = set(step_wise_select(fs_inputs, y_col='Y'))

        df_expected = {'COL1', 'COL2', 'COL3', 'COL4'}

        assert df_actual == df_expected

class TestStepWiseSelectEX(object):
    def test_values_sws(self, fs_inputs):
        df_actual = set(step_wise_select(fs_inputs, y_col='Y', exclude_cols=['COL5']))

        df_expected = {'COL1', 'COL2', 'COL3', 'COL4', 'COL5'}

        assert df_actual == df_expected

class TestRandomForest(object):
    def test_values_rf(self, fs_inputs):
        df_actual = set(random_forrest(fs_inputs, y_col='Y'))

        df_expected = {'COL1', 'COL2', 'COL3', 'COL4'}

        assert df_actual == df_expected

class TestLasso(object):
    def test_values_l(self, feat_sel_inputs):
        df_actual = set(lasso(feat_sel_inputs, 'Y', cutoff=0.1))

        df_expected = {'COL1', 'COL3'}

        assert df_actual == df_expected

class TestPearson(object):
    def test_values_p(self, corr_inputs):
        df_actual = set(pearson(corr_inputs, y_col='Y'))

        df_expected = {'COL1'}

        assert df_actual == df_expected

class TestMulticollie(object):
    def test_valuesmc(self, corr_inputs):
        df_actual = set(remove_multicollinearity(corr_inputs, y_col='Y'))

        df_expected = {'COL1', 'COL3', 'COL4'}

        assert df_actual == df_expected

class Testvif(object):
    def test_valuesvif(self, fs_inputs):
        df_actual = set(vif(fs_inputs, y_col='Y'))

        df_expected = {'COL1', 'COL2', 'COL3', 'COL4'}

        assert df_actual == df_expected




