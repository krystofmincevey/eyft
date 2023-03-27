import pandas as pd
import numpy as np
import geopandas

from typing import (
    Union, Tuple, Any, Dict, List, Callable
)
from collections import defaultdict

from ..data_processing import logger


# -------------------------------------
# TODO: Always return a kwargs dict.
#  This will allow you to process both train and
#  test data.
# TODO: Always have df and col in kwargs of processor nodes.


def do_nothing(
    df: Any,
    *args,  # To handle additional inputs passed to funct.
    **kwargs,  # To handle additional inputs passed to funct.
) -> Dict[str, Any]:
    return {"df": df}


def mean_impute(
    df: pd.DataFrame,
    col: str,
    mean: Union[int, float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    if mean is None:
        mean = df[col].mean()
        logger.info(f'Mean of {col} is {mean}.')

    df[col] = df[col].fillna(mean)
    return {"df": df, "col": col, "mean": mean}


def mode_impute(
    df: pd.DataFrame,
    col: str,
    mode: Union[int, float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    if mode is None:
        mode = df[col].mode()[0]
        logger.info(f'Mode of {col} is {mode}.')

    df[col] = df[col].fillna(mode)
    return {"df": df, "col": col, "mode": mode}


def z_normalise(
    df: pd.DataFrame,
    col: str,
    mean: Union[int, float] = None,
    stdev: Union[int, float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    if mean is None:
        mean = df[col].mean()
        logger.info(f'Mean of {col} is {mean}.')
    if stdev is None:
        stdev = df[col].std()
        logger.info(f'StDev of {col} is {stdev}.')
    df[col] = (df[col] - mean) / stdev
    return {"df": df, "col": col, "mean": mean, "stdev": stdev}


def min_max_scale(
    df: pd.DataFrame,
    col: str,
    min_val: Union[int, float] = None,
    max_val: Union[int, float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    if min_val is None:
        min_val = df[col].min()
        logger.info(f'Min of {col} is {min_val}.')
    if max_val is None:
        max_val = df[col].max()
        logger.info(f'Max of {col} is {max_val}.')
    df[col] = (df[col] - min_val) / (max_val - min_val)
    return {"df": df, "col": col, "min_val": min_val, "max_val": max_val}


def cap_3std(
    df: pd.DataFrame,
    col: str,
    mean: Union[int, float] = None,
    stdev: Union[int, float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    if mean is None:
        mean = df[col].mean()
        logger.info(f'Mean of {col} is {mean}.')
    if stdev is None:
        stdev = df[col].std()
        logger.info(f'StDev of {col} is {stdev}.')
    # raise NotImplementedError
    # return {"df": df, "col": col, "stdev": stdev}
    df[col] = np.where(df[col] < mean - 3 * stdev, mean - 3 * stdev, df[col])
    df[col] = np.where(df[col] > mean + 3 * stdev, mean + 3 * stdev, df[col])
    return {"df": df, "col": col, "stdev": stdev}


def cap(
    df: pd.DataFrame,
    col: str,
    raw_cap: Union[float] = None,
    prc_cap: float = 0.99,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    """
    Cap values in df[col]. If raw_cap is specified,
    values above raw_cap are converted to raw_cap.
    If raw_cap is not specified, the prc_cap percentile
    is calculated and used to cap values.
    """

    # -------------------------------
    # Process inputs:
    if prc_cap >= 1 or prc_cap <= 0:
        raise ValueError(
            f'The prc_cap must be between 0-1 and '
            f'not {prc_cap}.'
        )
    # -------------------------------

    if raw_cap is None:
        raw_cap = df[col].quantile(prc_cap)

    logger.info(f'Values above {raw_cap} in {col} are capped.')

    df[col] = np.where(df[col] > raw_cap, raw_cap, df[col])
    return{"df": df, "col": col, "raw_cap": raw_cap, "prc_cap": prc_cap}


def floor(
        df: pd.DataFrame,
        col: str,
        raw_cap: Union[float] = None,
        prc_cap: float = 0.99,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    """
    Floor values in df[col]. If raw_cap is specified,
    values below raw_cap are converted to raw_cap.
    If raw_cap is not specified, the prc_cap percentile
    is calculated and used to floor values.
    """

    # -------------------------------
    # Process inputs:
    if prc_cap >= 1 or prc_cap <= 0:
        raise ValueError(
            f'The prc_cap must be between 0-1 and '
            f'not {prc_cap}.'
        )
    # -------------------------------

    if raw_cap is None:
        raw_cap = df[col].quantile(prc_cap)

    logger.info(f'Values above {raw_cap} in {col} are capped.')

    df[col] = np.where(df[col] < raw_cap, raw_cap, df[col])
    return {"df": df, "col": col, "raw_cap": raw_cap, "prc_cap": prc_cap}


def median_impute(
    df: pd.DataFrame,
    col: str,
    median: Union[int, float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    if median is None:
        median = df[col].median()
        logger.info(f'Median of {col} is {median}.')

    df[col] = df[col].fillna(median)
    return {"df": df, "col": col, "median": median}


# TODO: Please add docstring
# PS: If the function already exists in pandas you don't
# have to implement it. Just add to the MAPPER.
def dummy_var(
    df: pd.DataFrame,
    col: str,
    dummy_na: Union[bool] = True,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    df = pd.get_dummies(df[col], dummy_na=dummy_na)
    return{"df": df, "col": col, "na_flag": dummy_na}


def categorize(
    df: pd.DataFrame,
    col: str,
    bins: int = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    df['new'] = pd.cut(df[col], bins="blocks")
    raise NotImplementedError
# How to optimize choice of bins --> use astroPy?


def geolocate(
    df: geopandas.GeoDataFrame,
    col: Union[str, Tuple[str]],
) -> Dict[str, Union[geopandas.GeoDataFrame, str]]:
    """
    Convert address in col to (x, y) coordinates.
    """
    raise NotImplementedError
# -------------------------------------


class Processor(object):

    MAPPER = {
        "mean_impute": mean_impute,
        "mode_impute": mode_impute,
        "median_impute": median_impute,
        "min_max_scale": min_max_scale,
        "z_normalise": z_normalise,
        "categorize": categorize,
        "cap": cap,
        "cap_3std": cap_3std,
        "floor": floor,
        "dummy_var": dummy_var,
        "geolocate": geolocate,
        "pass": do_nothing,
    }

    def __init__(
            self,
            df_train: pd.DataFrame,
            df_test: pd.DataFrame = None,
    ):
        if Processor.is_dataframe(df_test):
            # TEST WILL NOT FAIL IF MORE TEST COLUMNS THAN TRAIN COLS
            if set(df_train.columns) - set(df_test.columns):
                raise KeyError(
                    f"Missmatch between train ({df_train.columns}) "
                    f"and test ({df_test.columns}) columns."
                )

        self.df_train = df_train
        self.df_test = df_test

    @property
    def df_train(self):
        return self._df_train

    @df_train.setter
    def df_train(self, df: pd.DataFrame):
        self._df_train = df

    @property
    def df_test(self):
        return self._df_test

    @df_test.setter
    def df_test(self, df: pd.DataFrame):
        self._df_test = df

    @staticmethod
    def is_dataframe(df: pd.DataFrame) -> bool:
        if type(df) is pd.DataFrame:
            return True
        else:
            return False

    def _map(
            self, params: Dict[str, List[str]]
    ) -> Dict[str, List[Callable]]:
        mapped_params = defaultdict(list)
        for col, proc_instructions in params.items():

            for proc_key in proc_instructions:
                _proc_key = proc_key.lower()

                if _proc_key not in self.MAPPER:
                    raise KeyError(
                        f"{proc_key} not among supported methods in "
                        f"{Processor.__name__}: {self.MAPPER.keys()}."
                    )

                mapped_params[col].append(self.MAPPER[_proc_key])

        return mapped_params

    def process_data(
            self,
            params: Dict[str, List[str]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select and process columns specified
        in params. Note that params is a dict
        with col_name as key, and list of
        processing steps as vals.
        """

        params = self._map(params)

        # Select
        logger.info(
            f"Reducing the input space to: {params.keys()}"
        )
        df_train = self.df_train[params.keys()]
        if Processor.is_dataframe(self.df_test):
            df_test = self.df_test[params.keys()]
        else:
            df_test = None

        # Process
        for col, procs in params.items():
            for _proc in procs:

                logger.info(
                    f"Processing {col} by applying: {_proc.__name__}"
                )

                outputs = _proc(df_train, col)
                df_train = outputs.pop('df')
                if Processor.is_dataframe(df_test):
                    df_test = _proc(df_test, **outputs)['df']

        return df_train, df_test
