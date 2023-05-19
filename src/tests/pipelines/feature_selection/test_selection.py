import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.datasets import load_iris
from unittest import TestCase

from eyft.pipelines.feature_selection.selector import (
    forward_select, backward_eliminate,
    step_wise_select, random_forest, lasso, yx_correlation, vif,
    xs_correlation, complete_correlation, mutual_info,
    missing_ratio
)


def test_missing_ratio():
    data = {
        'A': [1, 2, 3, None, 5, 6, 7, 8, 9, 10],
        'B': [1, None, None, None, 5, None, 7, None, 9, None],
        'C': [1, 2, None, None, None, 6, 7, 8, None, 10],
        'D': [None, None, None, None, None, None, None, None, None, None],
        'y_col': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    y_col = 'y_col'
    cutoff = 0.5
    keep_cols = ['D']

    selected_features = missing_ratio(df, y_col, cutoff, keep_cols)

    # Check if the output is a list
    assert isinstance(selected_features, list), "Selected features should be a list"

    # Check if the specified keep_cols are included in the selected features
    for col in keep_cols:
        assert col in selected_features, f"Column {col} should be in the selected features"

    # Check if the features with missing value ratios below the cutoff are included
    assert 'A' in selected_features, "Feature A should be in the selected features"
    assert 'C' in selected_features, "Feature C should not be in the selected features"
    assert 'B' not in selected_features, "Feature B should not be in the selected features"

    # Check if the y_col variable is not in the selected features
    assert y_col not in selected_features, f"Target variable {y_col} should not be in the selected features"


def test_mutual_info_feature_selection():
    data = load_iris()
    iris_df = pd.DataFrame(data.data, columns=data.feature_names)
    iris_df["species"] = data.target

    mi_cutoff = 0.2
    keep_columns = ['sepal width (cm)']
    selected_features = mutual_info(iris_df, "species", mi_cutoff, keep_columns)

    assert isinstance(selected_features, list), "Selected features should be a list"
    assert len(selected_features) > 0, "At least one feature should be selected"
    assert 'sepal width (cm)' in selected_features, "Specified column to keep should be in the selected features"

    for feature in selected_features:
        assert feature in iris_df.columns, f"Selected feature {feature} should be in the original DataFrame columns"


class TestForwardSelect(object):

    def test_values(self, fs_inputs):
        df_actual = set(forward_select(fs_inputs, y_col='Y'))

        df_expected = {'COL1', 'COL2', 'COL4'}
        assert df_actual == df_expected

    def test_values_with_exclusion(self, fs_inputs):
        df_actual = set(forward_select(fs_inputs, y_col='Y', keep_cols=['COL5']))

        df_expected = {'COL1', 'COL2', 'COL4', 'COL5'}
        assert df_actual == df_expected


class TestBackwardEliminate(object):

    def test_values(self, fs_inputs):
        df_actual = set(backward_eliminate(fs_inputs, y_col='Y'))

        df_expected = {'COL1', 'COL2', 'COL4'}
        assert df_actual == df_expected

    def test_values_with_exclusion(self, fs_inputs):
        df_actual = set(backward_eliminate(fs_inputs, y_col='Y', keep_cols=['COL5']))

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
                fs_inputs, y_col='Y', keep_cols=['COL5'],
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
                keep_cols=['COL3'],
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
        df_actual = set(yx_correlation(corr_inputs, y_col='Y'))

        df_expected = {'COL1', 'COL2'}

        assert df_actual == df_expected

    def test_values2(self, corr_inputs):
        df_actual = set(yx_correlation(corr_inputs, y_col='Y', keep_cols=['COL3']))

        df_expected = {'COL1', 'COL2', 'COL3'}

        assert df_actual == df_expected


class TestXsCorrelation(object):

    def test_values(self, corr_inputs):
        df_actual = set(xs_correlation(corr_inputs, y_col='Y'))

        df_expected = {'COL1', 'COL3', 'COL4'}

        assert df_actual == df_expected

    def test_values2(self, corr_inputs):
        df_actual = set(xs_correlation(corr_inputs, y_col='Y', keep_cols=['COL2']))

        df_expected = {'COL1', 'COL2', 'COL3', 'COL4'}

        assert df_actual == df_expected


class TestPearson(object):
    def test_values(self, corr_inputs):
        df_actual = set(complete_correlation(corr_inputs, y_col='Y'))

        df_expected = {'COL1'}

        assert df_actual == df_expected
    def test_values2(self, corr_inputs):
        df_actual = set(complete_correlation(corr_inputs, y_col='Y', keep_cols=['COL3']))

        df_expected = {'COL1', 'COL3'}

        assert df_actual == df_expected


class TestVIF(object):

    def test_values(self, corr_inputs_2):
        df_actual = set(vif(corr_inputs_2, y_col='Y', keep_cols=['COL1']))

        df_expected = {'COL1', 'COL3'}
        assert df_actual == df_expected

    def test_values2(self, high_vif_inputs):
        df_actual = set(vif(high_vif_inputs, y_col='Y'))

        df_expected = {'COL4'}

        assert df_actual == df_expected
