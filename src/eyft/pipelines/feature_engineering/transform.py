import numpy as np
import pandas as pd
import geopandas

from typing import (
    Dict, List, Union, Tuple
)

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


def nan_categorize(
    df: pd.DataFrame,
    col: str,
    na_flag: Union[bool] = True,
    prefix: str = "missing",
) -> pd.DataFrame:
    """
    :param df: dataset we use
    :param col: you can choose the column you want to divide in dummy variables
    :param na_flag: you can choose if you want to include NaN as a dummy
    :param prefix: Alias prefix for new column -> {prefix}_{col}
    """
    df[f"{prefix}_{col}"] = pd.get_dummies(df[col], dummy_na=na_flag)
    return df


def geolocate(
    df: geopandas.GeoDataFrame,
    col: Union[str, Tuple[str]],
    **kwargs,  # added just to collect additional vars passed to funct
) -> pd.DataFrame:
    raise NotImplementedError
# -------------------------------------


class Transform(object):

    MAPPER = {
        "log": log_transform,
        "multiply_by": multiply_by,
        "divide_by": divide_by,
        "nan_categorize": nan_categorize,
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
