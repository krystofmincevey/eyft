import numpy as np
import pandas as pd
import geopandas

from typing import (
    Dict, List, Union, Tuple
)

from pandas import DataFrame

from ..feature_engineering import logger


# -------------------------------------
# TODO: Do not change kwargs of functions as the names
#  are used in transform().
def log_transform(
    df: pd.DataFrame,
    col: str,
    prefix: str = "log",
    **kwargs,  # added just to collect additional vars passed to funct
) -> pd.DataFrame:
    new_col = f"{prefix}_{col}"
    if new_col not in df.columns:
        logger.info(
            f"Adding new column: {new_col}, to df."
        )
        df[new_col] = np.log(df[col])
    return df


def multiply_by(
    df: pd.DataFrame,
    col: str,
    by_col: str,
    connector: str = "mult_by",
    **kwargs,  # added just to collect additional vars passed to funct
) -> pd.DataFrame:

    for _col in [col, by_col]:
        if _col not in df.columns:
            raise KeyError(
                f"{_col} not in {df.columns}"
            )

    new_col = f"{col}_{connector}_{by_col}"
    if new_col not in df.columns:
        logger.info(
            f"Adding new column: {new_col}, to df."
        )
        df[new_col] = df[col] * df[by_col]
    return df


def divide_by(
    df: pd.DataFrame,
    col: str,
    by_col: str,
    connector: str = "div_by",
    **kwargs,  # added just to collect additional vars passed to funct
) -> pd.DataFrame:
    """

    :param df: dataset we use
    :param col: column you want to divide
    :param by_col: column you want to use as deviation
    :param connector:
    :param kwargs:
    :return:
    """

    for _col in [col, by_col]:
        if _col not in df.columns:
            raise KeyError(
                f"{_col} not in {df.columns}"
            )

    new_col = f"{col}_{connector}_{by_col}"
    if new_col not in df.columns:
        logger.info(
            f"Adding new column: {new_col}, to df."
        )
        df[new_col] = df[col] / df[by_col]

    return df


def inverse(
        df: pd.DataFrame,
        col: str,
        prefix: str = 'inverse',
        **kwargs,
):
    """
    Performs 1/col and saves in new_col = {prefix}_{col}
    use log function as guide.
    """

    # TODO: Arthur

    return df


def multiply_all(
    df: pd.DataFrame,
    col: str,
    suffix: str = "mult_all",
    **kwargs,  # added just to collect additional vars passed to funct
):
    """
    Function to mutliply all columns
    that contain col in name.
    EG: if cols ['a', 'aa', 'bc', 'ab'] in df
    than if col = 'a' multiply 'a' x 'aa' x 'ab'.
    store in {col}_{suffix}
    """

    # TODO: Arthur

    return df


def sum_all(
    df: pd.DataFrame,
    col: str,
    suffix: str = "mult_all",
    **kwargs,  # added just to collect additional vars passed to funct
):
    """
    Function to sum all columns
    that contain col in name.
    EG: if cols ['a', 'aa', 'bc', 'ab'] in df
    than if col = 'a' sum 'a' + 'aa' + 'ab'.
    store in {col}_{suffix}
    """

    # TODO: Arthur

    return df


def geolocate(
    df: geopandas.GeoDataFrame,
    col: Union[str, Tuple[str]],
    **kwargs,  # added just to collect additional vars passed to funct
) -> pd.DataFrame:
    raise NotImplementedError
    # return df
# -------------------------------------


class Transform(object):

    MAPPER = {
        "log": log_transform,
        "multiply_by": multiply_by,
        "multiply_all": multiply_all,
        "divide_by": divide_by,
        "geolocate": geolocate,
    }

    def __init__(
            self,
            df: pd.DataFrame,
    ):
        self._df = df

    def get_df(self):
        return self._df

    def transform(
        self,
        params: Dict[str, List[str]]
    ):
        """
        Transform columns specified
        in params. Note that params is a dict
        with col_name as key, and list of
        transformation steps as vals.
        """

        for col, proc_instructions in params.items():

            for proc_key in proc_instructions:
                full_proc = proc_key.split(' ')
                if len(full_proc) > 1:
                    _proc_key, by_col = full_proc
                else:
                    _proc_key = full_proc[0]
                    by_col = None

                _proc_key = _proc_key.lower()
                if _proc_key not in self.MAPPER:
                    raise KeyError(
                        f"{proc_key} not among supported methods in "
                        f"{Transform.__name__}: {self.MAPPER.keys()}."
                    )

                _proc = self.MAPPER[_proc_key]
                logger.info(
                    f"Transforming {col} by applying: {_proc.__name__}"
                )
                self._df = _proc(self._df, col=col, by_col=by_col)

        return self
