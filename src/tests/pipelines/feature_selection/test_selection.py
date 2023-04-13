import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from eyft.pipelines.feature_selection.nodes import (select, _sm_model, forward_select, backward_eliminate,
                                                    step_wise_select, random_forrest, lasso, pearson, vif)

class TestSelect(object):
    def test_values_ts(self, fs_inputs):
        df actual = select(fs_inputs, [])

class TestSMModel(object):
    def test_values_sm(self, fs_inputs):
        df_actual =
class TestForwardSelect(object):
    def test_values_fs(self, fs_inputs):
        df_actual =
class TestBackwardEliminate(object):
    def test_values_be(self, fs_inputs):
        df_actual =
class TestStepWiseSelect(object):
    def test_values_sws(self, fs_inputs):
        df_actual =
class TestRandomForest(object):
    def test_values_rf(self, fs_inputs):
        df_actual =
class TestLasso(object):
    def test_values_l(self, fs_inputs):
        df_actual =
class TestPearson(object):
    def test_values_p(self, fs_inputs):
        df_actual =
class TestVIF(object):
    def test_values_vif(self, fs_inputs):
        df_actual =