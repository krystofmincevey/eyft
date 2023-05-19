import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from typing import Callable, Union
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


# PROCESS DATA FOR DECISION TREE:
# ____________________________________________________________________________________
def process_dt(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    :param df:
    :param kwargs:
        cat_impute_key: str = 'mode',
        num_impute_key: str = 'median',
        prefix='ms_dummy', is_create_dummy=False,
    :return: pd.Dataframe
    """

    df = handle_missing(df=df, **kwargs)
    df = handle_categorical(df=df)

    return df


# PROCESS DATA FOR MIXED GLM:
# ____________________________________________________________________________________
def process_glm(
        df: pd.DataFrame, ycol: str,
        md_kwargs: dict, outlier_kwargs: dict,
        collinear_kwargs: dict,
) -> pd.DataFrame:
    """
    :param df:
    :param ycol: y_col column
    :param md_kwargs:
        cat_impute_key: str = 'mode',
        num_impute_key: str = 'median'
        prefix='ms_dummy', is_create_dummy=False,
    :param outlier_kwargs:
        iqr_thrd: float = 1.5,
        z_thrd: float = 3, prc_thrd: float = 0.05,
        is_outlier_dummy: bool = False
    :param collinear_kwargs:
        vif_threshold: float = 5.0
    :return: pd.Dataframe
    """

    df = handle_missing(df=df, **md_kwargs)
    df = handle_categorical(df=df)

    df = handle_outliers(
        df=df, cols=get_numericals(df, ycol), **outlier_kwargs
    )
    df = transform_features(df=df, cols=get_numericals(df, ycol))

    cat_cols = find_categorical(df)
    df = remove_collinear_variables(
        df=df, exclude_cols=[TARGET_COL, *cat_cols],
        **collinear_kwargs,
    )
    df = scale(df, get_numericals(df, ycol))  # scale to ensure comparable feature scores
    return df


# General:
# ____________________________________________________________________________________
def find_categorical(df: pd.DataFrame, threshold: int = 5) -> list:
    # HACK TO AVOID HAVING TO WRITE OUT THE VARIABLES.
    cat_cols = []
    for col in df.columns:
        n_unique = len(np.unique(df[col].values))
        if n_unique < threshold:
            cat_cols.append(col)
    return cat_cols


def get_numericals(df: pd.DataFrame, ycol: str):
    cat_cols = find_categorical(df)
    num_cols = [col for col in df.columns if col not in [ycol, *cat_cols]]
    return num_cols


# STEP 1; Handle Missing:
# ___________________________________________________________________________________
def key2impute(key):
    # WAS INITIALLY CREATED TO ALLOW MORE ADVANCED IMPUTATION IMPLEMENTATIONS!

    def mode(data: pd.Series):
        return stats.mode(data, nan_policy='omit')[0]

    def zero(*args, **kwargs):
        return 0

    _key2imp = {
        'mean': lambda x, y, z: static_impute(
            df=x, col=y, impute_idx=z, summary_func=np.nanmean,
        ),
        'median': lambda x, y, z: static_impute(
            df=x, col=y, impute_idx=z, summary_func=np.nanmedian,
        ),
        'mode': lambda x, y, z: static_impute(
            df=x, col=y, impute_idx=z, summary_func=mode,
        ),
        'zero': lambda x, y, z: static_impute(
            df=x, col=y, impute_idx=z, summary_func=zero,
        ),
    }

    _key = key.lower()
    if _key not in _key2imp:
        raise KeyError(f'{key} not in supported list: {_key2imp.keys()}')
    return _key2imp[_key]


def static_impute(
        df: pd.DataFrame, col: str,
        impute_idx: Union[np.ndarray, list],
        summary_func: Callable
) -> pd.DataFrame:
    summary_value = summary_func(df[col].values)
    print(f'{col} fill NANs with {summary_value}')
    df.loc[impute_idx, col] = summary_value
    return df


def impute(
        df: pd.DataFrame, col: str, impute_key: str,
        impute_idx: Union[np.ndarray, list],
):
    impute_funct = key2impute(impute_key)
    df = impute_funct(df, col, impute_idx)
    return df


def handle_missing(
        df: pd.DataFrame,
        cat_impute_key: str = 'mode', num_impute_key: str = 'median',
        prefix='ms_dummy', is_create_dummy=False
):
    # ADD DUMMIES AND IMPUTE
    cat_cols = find_categorical(df)

    for col in df.columns:

        is_null = df[col].isnull().values
        if is_null.any():

            print(f'\n{col} has {is_null.sum()} missings.')

            if is_create_dummy:
                df[f'{prefix}_{col}'] = is_null.astype(int)

            idx = np.nonzero(is_null)[0]
            if col in cat_cols:
                df = impute(
                    df=df, col=col, impute_key=cat_impute_key,
                    impute_idx=idx,
                )
                cat_cols.remove(col)  # for quicker lookup
            else:
                df = impute(
                    df=df, col=col, impute_key=num_impute_key,
                    impute_idx=idx,
                )

    return df


# STEP 2; Handle Categorical
# ____________________________________________________________________________________
def handle_categorical(df: pd.DataFrame) -> pd.DataFrame:
    # CONVERT TO DUMMY
    cat_cols = find_categorical(df)
    for col in cat_cols:
        n_unique = len(np.unique(df[col].values))
        if n_unique > 2:
            print(f"\nReplacing categoricals in {col} with dummies\n")
            df = pd.get_dummies(
                data=df, columns=[col],
                drop_first=True
            )
    return df


# STEP 3; Cap Outliers
# ____________________________________________________________________________________
def handle_outliers(
        df: pd.DataFrame, cols: list, iqr_thrd: float = 1.5,
        z_thrd: float = 3, prc_thrd: float = 0.05,
        is_outlier_dummy=False, verbose=False,
) -> pd.DataFrame:
    """
    Uses 3std check, prc, and IQR checks (all must be satisfied)
    to identify and cap/floor outliers. If is_outlier_dummy is true,
    then a dummy column demarcating an outlier in col
    is created.
    """

    outlier_dummy = np.zeros(len(df))
    for col in cols:

        mean, std = np.mean(df[col].values), np.std(df[col].values)
        z_val_bottom = mean - (z_thrd * std)
        z_val_top = mean + (z_thrd * std)

        bottom_prc, top_prc = df[col].quantile([prc_thrd, 1 - prc_thrd]).values

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        bottom_cap = Q1 - iqr_thrd * IQR
        top_cap = Q3 + iqr_thrd * IQR

        bottom_outliers = (df[col].values < bottom_cap) * \
                          (df[col].values < bottom_prc) * (df[col].values < z_val_bottom)
        outlier_dummy += bottom_outliers.astype(int)

        top_outliers = (df[col].values > top_cap) * \
                       (df[col].values > top_prc) * (df[col].values > z_val_top)
        outlier_dummy += top_outliers.astype(int)

        bottom_replace_val = (1 / 3) * (bottom_cap + bottom_prc + z_val_bottom)
        top_replace_val = (1 / 3) * (top_cap + top_prc + z_val_top)

        n_top = top_outliers.sum()
        n_bottom = bottom_outliers.sum()

        for out_id, n_out, outlier_idx, replace_val in zip(
                ['Peak', 'Tail'],
                [n_bottom, n_top], [bottom_outliers, top_outliers],
                [bottom_replace_val, top_replace_val]
        ):
            if n_out > 0:
                print(
                    f'\nFound {n_out} {out_id} outliers in {col}:\n' +
                    f'{df[col][outlier_idx].values}\n' +
                    f'Replacing outliers with {replace_val}\n'
                )

        if verbose:
            print(df[col].describe())

        df.loc[bottom_outliers, col] = bottom_replace_val
        df.loc[top_outliers, col] = top_replace_val

    if is_outlier_dummy:
        df['outliers_dummy'] = np.clip(outlier_dummy, 0, 1)

    return df


# STEP 4; Transform continuous covariates
# ____________________________________________________________________________________
def transform_features(df: pd.DataFrame, cols: list, is_plot=False):
    for col in cols:
        stat, p = stats.shapiro(df[col].values)
        if p < 0.01:
            print(f'\n{col} may not be Gaussian.')
            print(f'Pre Transformation; {col}:')
            print(df[col].describe())

            if is_plot:
                df[col].plot.hist(title=col)
                plt.show()

            # BOX COX NEEDS + VALS
            min_val = df[col].min()
            if min_val <= 0:
                normaliser = np.abs(min_val) + 1
                print(normaliser)
            else:
                normaliser = 0

            tf_vals, _lambda = stats.boxcox(df[col].values + normaliser)
            df.drop(labels=[col], axis="columns", inplace=True)

            # Commented out for sake of visuals in document
            #             if normaliser:
            #                 new_col = f'{col}_plus_{normaliser}_with_box_cox_{_lambda}'
            #             else:
            #                 new_col = f'{col}_with_box_cox_{_lambda}

            new_col = f'{col}_plus_with_box_cox'

            df[new_col] = tf_vals
            print(f'Post transformation; {new_col} properties:')
            print(df[new_col].describe())

            if is_plot:
                df[new_col].plot.hist(title=new_col)
                plt.show()

    return df


# STEP 5; Remove Collinear Variables
# ____________________________________________________________________________________
def remove_collinear_variables(
        df: pd.DataFrame, exclude_cols: list, vif_threshold: float = 5.0
):
    print('\nRemoving collinear variables...')
    collinear_variables = []
    max_vif_feature = 'None'
    round = 1
    features = [col for col in df.columns if col not in exclude_cols]
    while max_vif_feature is not None:
        X_constant = sm.add_constant(df[features])
        vif = [
            variance_inflation_factor(X_constant.values, i)
            for i in range(X_constant.shape[1])
        ]
        vif_df = pd.DataFrame({'vif': vif[1:]}, index=features).T
        max_vif = vif_threshold
        max_vif_feature = None

        for column in vif_df.columns:
            # print(float(vif_df[column]['vif']))
            if float(vif_df[column]['vif']) > max_vif:
                max_vif = float(vif_df[column]['vif'])
                # print(f'{column}s vif ={max_vif}')
                max_vif_feature = column

        if max_vif_feature is not None:
            df = df.drop([max_vif_feature], axis=1)
            features.remove(max_vif_feature)
            print(f'Round {round}: {max_vif_feature} is dropped with vif {max_vif}')
            collinear_variables.append(max_vif_feature)
        round += 1

    return df


# STEP 6; Scale:
# ____________________________________________________________________________________
def scale(df: pd.DataFrame, cols: list):
    print('\nMin Max scaling cols:')
    for col in cols:
        print(col)

    _min = df[cols].min()
    df[cols] -= _min

    _max = df[cols].max()
    df.loc[:, cols] /= _max
    return df
