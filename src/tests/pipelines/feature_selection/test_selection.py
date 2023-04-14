import pandas as pd
from pandas.testing import assert_frame_equal
from unittest import TestCase


# from eyft.pipelines.feature_selection.nodes import (
#     select, _sm_model, forward_select, backward_eliminate,
#     step_wise_select, random_forrest, lasso, pearson, vif
# )

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
        df_actual = forward_select(fs_inputs, y_col='Y')

        df_expected = set['COL1', 'COL2', 'COL3', 'COL4']
        TestCase.assertCountEqual(df_actual, df_expected)

#class TestBackwardEliminate(object):
    #def test_values_be(self, fs_inputs):
        df_actual = backward_eliminate(fs_inputs)

       # df_expected =
#class TestStepWiseSelect(object):
   #def test_values_sws(self, fs_inputs):
        df_actual = step_wise_select(fs_inputs)

       # df_expected =
#class TestRandomForest(object):
    #def test_values_rf(self, fs_inputs):
        df_actual = random_forrest(fs_inputs)

     #  df_expected =
#class TestLasso(object):
    #def test_values_l(self, fs_inputs):
        df_actual = lasso(fs_inputs)

      # df_expected =
#class TestPearson(object):
    #def test_values_p(self, fs_inputs):
        df_actual = pearson(fs_inputs)

       #df_expected =



