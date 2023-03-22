import pandas as pd
import geopandas

from typing import (
    Union, Tuple, Any, Dict, List, Callable
)
from collections import defaultdict

from src.eyft import logger


# -------------------------------------
# TODO: Always return a kwargs dict.
#  This will allow you to process both train and
#  test data.
# TODO: Always have df and col in kwargs of processor nodes.


def do_nothing(
    df: Any
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
        mode = df[col].mode()
        logger.info(f'Mode of {col} is {mode}.')

    df[col] = df[col].fillna(mode)
    return {"df": df, "col": col, "mode": mode}



def z_normalise(
    df: pd.DataFrame,
    col: str,
    mean: Union[int, float] = None,
    stdev: Union[int, float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    if mean is None or stdev is None:
        stdev, mean = df[col].std(), df[col].mean()
        logger.info(f'Mean of {col} is {mean} and StDev is {stdev}.')

    df[col] = df[col].fillna((df[col] - df[col].mean()) / df[col].stdev())
    return {"df": df, "col": col, "mean": mean, "stdev": stdev}

def min_max_scale(
    df: pd.DataFrame,
    col: str,
    min_val: Union[int, float] = None,
    max_val: Union[int, float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    if min_val is None or max_val is None:
        min_val, max_val = df[col].min(), df[col].max()
        logger.info(
            f'Min of {col} is {min_val} and Max is {max_val}.'
        )

    df[col] = df[col].fillna((df[col] - df[col].min()) / (df[col].max() - df[col].min()))
    return {"df": df, "col": col, "min_val": min_val, "max_val": max_val}





def cap_3std(
    df: pd.DataFrame,
    col: str,
    stdev: Union[int, float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    if stdev is None:
        stdev = df[col].std()
        logger.info(f'StDev of {col} is {stdev}.')
    raise NotImplementedError
    # return {"df": df, "col": col, "stdev": stdev}

    df[col] = df[col].mean() - 3 * df[col].std(), df[col].mean() + 3 * df[col].std()
    return {"df": df, "col": col, "stdev": stdev}

def categorize(
    df: pd.DataFrame,
    col: str,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    return NotImplementedError


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
