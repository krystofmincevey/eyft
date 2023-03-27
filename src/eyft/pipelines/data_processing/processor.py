import pandas as pd
import numpy as np
import geopandas
import matplotlib.pyplot as plt

from typing import (
    Union, Tuple, Any, Dict, List, Callable
)
from collections import defaultdict

from src.eyft.pipelines.feature_selection import logger


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
    mean: Union[float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    if mean is None:
        mean = float(df[col].mean())
        logger.info(f'Mean of {col} is {mean}.')

    df[col] = df[col].fillna(mean)
    return {"df": df, "col": col, "mean": mean}



def mode_impute(
    df: pd.DataFrame,
    col: str,
    mode: Union[float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    if mode is None:
        mode = float(df[col].mode())
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
    # What to do with the imputed values?
    # df[col] = df[col].fillna()
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
        logger.info(f'min of {col} is {max_val}.')

    df[col] = (df[col] - min_val) / (max_val - min_val)
    # What to do with imputed values?
    # ....
    return {"df": df, "col": col, "min_val": min_val, "max_val": max_val}





def cap_3std(
    df: pd.DataFrame,
    col: str,
    mean: Union[int, float] = None,
    stdev: Union[int, float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    if mean is None:
        mean = df[col].mean()
        logger.info(f'mean of {col} is {mean}.')
    if stdev is None:
        stdev = df[col].std()
        logger.info(f'StDev of {col} is {stdev}.')
    # raise NotImplementedError
    # return {"df": df, "col": col, "stdev": stdev}
    df[col] = np.where(df[col] < mean - 3 * stdev, mean - 3 * stdev, df[col])
    df[col] = np.where(df[col] > mean + 3 * stdev, mean + 3 * stdev, df[col])
    return {"df": df, "col": col, "stdev": stdev}

def cap_perc(
    df: pd.DataFrame,
    col: str,
    cap_perc: Union[float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    if cap_perc is None:
        cap_perc_value = df[col].quantile(0.99)
        logger.info(f'cap of {col} is {cap_perc}.')
    elif cap_perc <= 0 or cap_perc >= 1:
        raise "You need to insert a number between 0 and 1"
    else:
        cap_perc_value = df[col].quantile(cap_perc)
    df[col] = np.where(df[col] > cap_perc_value, cap_perc_value, df[col])
    return{"df": df, "col": col, "cap_perc": cap_perc, "cap_perc_value": cap_perc_value}




def floor_perc(
    df: pd.DataFrame,
    col: str,
    floor_perc: Union[float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    if floor_perc is None:
        floor_perc_value = df[col].quantile(0.01)
        logger.info(f'floor of {col} is {floor_perc}.')
    elif floor_perc <= 0 or floor_perc >= 1:
        raise "You need to insert a number between 0 and 1"
    else:
        floor_perc_value = df[col].quantile(floor_perc)
    df[col] = np.where(df[col] < floor_perc_value, floor_perc_value, df[col])
    return{"df": df, "col": col, "floor_perc": floor_perc, "floor_perc_value": floor_perc_value}


def median_impute(
    df: pd.DataFrame,
    col: str,
    median: Union[float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    if median is None:
        median = float(df[col].median())
        logger.info(f'Median of {col} is {median}.')

    df[col] = df[col].fillna(median)
    return {"df": df, "col": col, "median": median}

def dummy_var(
    df: pd.DataFrame,
    col: str,
    na_flag: Union[bool] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    if na_flag is None:
        na_flag = True
    df[col] = pd.get_dummies(df[col], dummy_na= na_flag)
    return{"df": df, "col": col, "na_flag": na_flag}




# This one is not finished

def summ_statistics(
    df: pd.DataFrame,
    col: str,
    stat: Union[int, float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:

    df[col] = df[col].describe




















######################################################################
def categorize(
    df: pd.DataFrame,
    col: str,
    bins: int = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:

    df['new'] = pd.cut(df[col], bins="blocks")
    return NotImplementedError
# How to optimize choice bins --> use astroPy?


def geolocate(
    df: geopandas.GeoDataFrame,
    col: Union[str, Tuple[str]],
) -> Dict[str, Union[geopandas.GeoDataFrame, str, int, float]]:


    raise NotImplementedError
# -------------------------------------


class Processor(object):

    MAPPER = {
        "mean_impute": mean_impute,
        "mode_impute": mode_impute,
        "z_normalise": z_normalise,
        "categorize": categorize,
        "cap_3std": cap_3std,
        "pass": do_nothing,
    }

    def __init__(
            self,
            df_train: pd.DataFrame,
            df_test: pd.DataFrame = None,
    ):
        self.df_train = df_train
        self.df_test = df_test

        if df_test not in [None, "None"]:
            if len(df_train[0].columns.intersection(df_test)) != df_train[0].shape[1]:
                raise KeyError(
                    f"Missmatch between train ({df_train.columns}) "
                    f"and test ({df_test.columns}) columns."
                )

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
        if self.df_test not in [None, "None"]:
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
                if df_test is not None:
                    df_test = _proc(df_test, **outputs)['df']

        return df_train, df_test
