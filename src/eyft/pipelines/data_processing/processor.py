import pandas as pd
import numpy as np

from typing import (
    Union, Tuple, Any, Dict, List, Callable
)
from collections import defaultdict
from scipy import stats

from ..data_processing import logger


# -------------------------------------
# TODO: Always return a kwargs dict.
#  This will allow you to process both train and
#  test data.
# TODO: Always have df and col in kwargs of processor nodes.
# TODO: Check which things are done in place
#   and potentially remove return statements.


def _get_mos(
    min_val: Union[int, float],
    max_val: Union[int, float],
    adj: float = 0.005  # 0.5%
) -> Union[int, float]:
    """
    Return margin of safety (MOS):
    adj * range
    """
    if min_val < 0 and max_val < 0:
        _range = abs(min_val) - abs(max_val)
    elif min_val < 0:
        _range = max_val + abs(min_val)
    else:
        _range = max_val - min_val

    return _range * adj


def do_nothing(
    df: Any,
    *args,  # To handle additional inputs passed to funct.
    **kwargs,  # To handle additional inputs passed to funct.
) -> Dict[str, Any]:
    return {"df": df}


# Impute ------------------------------------------------------
def mean_impute(
    df: pd.DataFrame,
    col: str,
    mean: Union[float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    """
    :param df: dataset we use
    :param col: you can choose the column you want to impute
    :param mean: you can choose the imputation value, if None => we use the calculated mean
    """
    if mean is None:
        mean = float(df[col].mean())

    logger.info(f'Imputing NANs in {col} with mean: {mean}.')
    df[col] = df[col].fillna(mean)

    return {"df": df, "col": col, "mean": mean}


def mode_impute(
    df: pd.DataFrame,
    col: str,
    mode: Union[float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    """
    :param df: dataset we use
    :param col: you can choose the column you want to impute
    :param mode: you can choose the imputation value, if None => we use the calculated mode
    """
    if mode is None:
        mode = float(df[col].mode())

    logger.info(f'Imputing NANs in {col} with mode: mode.')
    df[col] = df[col].fillna(mode)
    return {"df": df, "col": col, "mode": mode}


def median_impute(
    df: pd.DataFrame,
    col: str,
    median: Union[float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    """
    :param df: dataset we use
    :param col: you can choose the column you want to impute
    :param median: you can choose the imputation value, if None => we use the calculated median
    """
    if median is None:
        median = float(df[col].median())

    logger.info(f'Imputing NANs in {col} with median: {median}.')
    df[col] = df[col].fillna(median)
    return {"df": df, "col": col, "median": median}
# -------------------------------------------------------


# Scale -------------------------------------------------
def z_normalise(
    df: pd.DataFrame,
    col: str,
    mean: Union[int, float] = None,
    stdev: Union[int, float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    """
    :param df: dataset we use
    :param col: you can choose the column you want to impute
    :param mean: you can choose the value used as mean in the
        standardization, if None => we use the calculated mean
    :param stdev: you can choose the value used as standard
        deviation in the standardization, if None => we use the
        calculated standard deviation
    """
    if mean is None:
        mean = df[col].mean()
    if stdev is None:
        stdev = df[col].std(ddof=0)

    if stdev != 0:
        logger.info(
            f'Z Normalising {col} with mean: {mean} and stdev: {stdev}.'
        )
        df[col] = (df[col] - mean) / stdev
    else:
        logger.warning(
            f"The STDEV of {col} is zero, hence cannot "
            f"perform {z_normalise.__name__}."
        )

    return {"df": df, "col": col, "mean": mean, "stdev": stdev}


def boxcox_normalise(
    df: pd.DataFrame,
    col: str,
    normaliser: Union[int, float] = None
) -> pd.DataFrame:
    """
    Return a dataset transformed by a Box-Cox power transformation.
    Normaliser: used to ensure that all vals are > 0 in df.col.
    """

    if normaliser is None:
        min_val = df[col].min()
        max_val = df[col].max()

        normaliser = max(
            _get_mos(min_val=min_val, max_val=max_val), 1
        )  # margin_of_safety
        if min_val <= 0:
            normaliser += abs(min_val)

    logger.info(f'Applying Box-Cox transformation to {col}.')
    tf_vals, _ = stats.boxcox(df[col].values + normaliser)
    df[col] = tf_vals
    return df


def min_max_scale(
    df: pd.DataFrame,
    col: str,
    min_val: Union[int, float] = None,
    max_val: Union[int, float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    """
    :param df: dataset we use
    :param col: you can choose the column you want to impute
    :param min_val: you can choose the value used as
        minimum in the scaling, if None => we use the calculated minimum
    :param max_val: you can choose the value used as
        maximum in the scaling, if None => we use the calculated maximum
    """
    if min_val is None:
        min_val = df[col].min()
    if max_val is None:
        max_val = df[col].max()

    if max_val - min_val != 0:
        logger.info(
            f'Min-Max scale {col} with min: {min_val} and max: {max_val}.'
        )
        df[col] = (df[col] - min_val) / (max_val - min_val)
    else:
        logger.warning(
            f"The Min and Max of {col} are equal, hence cannot "
            f"to {min_max_scale.__name__}."
        )

    return {"df": df, "col": col, "min_val": min_val, "max_val": max_val}
# ---------------------------------------------------------


# Handle Outliers -----------------------------------------
def cap_3std(
    df: pd.DataFrame,
    col: str,
    mean: Union[int, float] = None,
    stdev: Union[int, float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    """
    :param df: dataset we use
    :param col: you can choose the column you want to impute
    :param mean: you can choose the value used as mean, if None => we use the calculated mean
    :param stdev: you can choose the value used as standard deviation,
        if None => we use the calculated standard deviation
    """
    if mean is None:
        mean = df[col].mean()
    if stdev is None:
        stdev = df[col].std()

    logger.info(
        f'Cap values larger and smaller than 3stdev in {col} |'
        f' mean: {mean} and stdev: {stdev}.'
    )
    df[col] = np.where(df[col] < mean - 3 * stdev, mean - 3 * stdev, df[col])
    df[col] = np.where(df[col] > mean + 3 * stdev, mean + 3 * stdev, df[col])
    return {"df": df, "col": col, "stdev": stdev}


def cap(
    df: pd.DataFrame,
    col: str,
    prc_cap: float = 0.99,
    abs_cap: float = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    """
    :param df: dataset we use
    :param col: you can choose the column you want to impute
    :param prc_cap: you can choose the percentile used for the
        capping of the outliers,
    :param abs_cap: if specified prc_cap is not used, and the absolute
        cap value is used to cap values in col.
    """
    if abs_cap is None:

        if prc_cap <= 0 or prc_cap >= 1:
            raise ValueError(
                f"prc_cap must be between 0 and 1 and not {prc_cap}"
            )
        else:
            abs_cap = df[col].quantile(prc_cap)
            logger.info(f"Capping values above {prc_cap}prc.")

    logger.info(
        f'Cap values larger than {abs_cap} in {col}.'
    )
    df[col] = np.where(df[col] > abs_cap, abs_cap, df[col])
    return{"df": df, "col": col, "prc_cap": prc_cap, "abs_cap": abs_cap}


def floor(
    df: pd.DataFrame,
    col: str,
    prc_floor: float = 0.01,
    abs_floor: float = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    """
    :param df: dataset we use
    :param col: you can choose the column you want to impute
    :param prc_floor: you can choose the percentile used for the flooring of the outliers,
        if None => we use the 0.01 percentile
    :param abs_floor: if specified prc_floor is not used, and the absolute
        floor value is used to floor values in col.
    """
    if abs_floor is None:

        if prc_floor <= 0 or prc_floor >= 1:
            raise ValueError(
                f"prc_cap must be between 0 and 1 and not {prc_floor}"
            )
        else:
            abs_floor = df[col].quantile(prc_floor)
            logger.info(f"Flooring values below {prc_floor}prc.")

    logger.info(
        f'Floor values smaller than {abs_floor} in {col}.'
    )
    df[col] = np.where(df[col] < abs_floor, abs_floor, df[col])
    return{"df": df, "col": col, "prc_floor": prc_floor, "abs_floor": abs_floor}


def floor_and_cap(
    df: pd.DataFrame,
    col: str,
    prc_cap: float = 0.99,
    abs_cap: float = None,
    prc_floor: float = 0.01,
    abs_floor: float = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    """
    Performs both capping and flooring
    on a column at the same time.
    """

    if abs_floor is None:

        if prc_floor <= 0 or prc_floor >= 1:
            raise ValueError(
                f"prc_cap must be between 0 and 1 and not {prc_floor}"
            )
        else:
            abs_floor = df[col].quantile(prc_floor)
            logger.info(f"Flooring values below {prc_floor}prc.")

    if abs_cap is None:

        if prc_cap <= 0 or prc_cap >= 1:
            raise ValueError(
                f"prc_cap must be between 0 and 1 and not {prc_cap}"
            )
        else:
            abs_cap = df[col].quantile(prc_cap)
            logger.info(f"Capping values above {prc_cap}prc.")

    logger.info(
        f'Floor values smaller than {abs_floor} and '
        f'cap values larger than {abs_cap} in {col}.'
    )
    df[col] = np.where(df[col] < abs_floor, abs_floor, df[col])
    df[col] = np.where(df[col] > abs_cap, abs_cap, df[col])

    return {
        "df": df, "col": col,
        "prc_floor": prc_floor, "abs_floor": abs_floor,
        "prc_cap": prc_cap, "abs_cap": abs_cap,
    }
# -------------------------------------------------------


# Bucket/Categorize -------------------------------------
def segment(
    df: pd.DataFrame,
    col: str,
    bins: List[Union[float, int]] = None,
    labels: List[str] = None,
    prefix: str = ''
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    """
    :param df: dataset we use
    :param col: you can choose the column that you want to categorize
    :param bins: numerical limits of the bins for the categories
    :param labels: specifies the labels for the returned bins.
    :param prefix: Alias prefix for new column -> {prefix}_{col}
    """
    if bins is None:
        min_val = np.abs(df[col].min())
        max_val = np.abs(df[col].max())
        margin_of_safety = _get_mos(min_val=min_val, max_val=max_val)

        bins = np.histogram_bin_edges(
            df[col], bins="auto",
            range=(min_val - margin_of_safety, max_val + margin_of_safety)
        )

    if prefix:
        new_col = f"{prefix}_{col}"
    else:
        new_col = col

    df[new_col] = pd.cut(df[col], bins=bins, labels=labels)
    return{"df": df, "col": col, "bins": bins, "prefix": prefix}


# PLACED IN PROCESSING.PY AS THIS STEP
# WILL NEED TO BE IMPLEMENTED BEFORE IMPUTING.
def cat_dummies(
    df: pd.DataFrame,
    col: str,
    prefix: str = 'no'
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    """
    Function for tracking presence of nans (missing values).
    A new column in df (with name new_col) should be created.

    I.E. [10, 40, nan, 40] -> [0, 0, 1, 0]
        ['a', 'b', '', nan, 'f'] -> [0, 0, 1, 1, 0]
    """
    if prefix:
        new_col = f"{prefix}_{col}"
    else:
        new_col = col

    return {"df": df, "col": col, "prefix": prefix}


def categorize(
    df: pd.DataFrame,
    col: str,
    cats: List[str],
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:
    """
    Function that creates dummies for each category
    of df[col]. Note that if categories are specified
    (think test data) than the function should use these
    to categorize the column. If cats are not provided
    that the function should find all the unique values
    in df[col] and create dummies for each category.
    The dummies are to be stored in columns following
    the naming {col}_{category}.

    EG:
    vals: ['a', 'b', 'a', 'd', nan] ->
        should create 3 extra columns:
        vals_a: [1, 0, 1, 0, 0]
        vals_b: [0, 1, 0, 0, 0]
        vals_d: [0, 0, 0, 1, 0]

    if cats are provided (imagine if cats = ['a', 'd', 'e']
    than only two cols are created:
        vals_a: [1, 0, 1, 0, 0]
        vals_d: [0, 0, 0, 1, 0]
        vals_e: [0, 0, 0, 0, 0]
    """

    df[col] = pd.get_dummies(df[col], columns=col, dtype=int)

    return {"df": df, "col": col, "cats": cats}
# --------------------------------------------------


# -------------------------------------
class Processor(object):

    MAPPER = {
        'boxcox_normalise': boxcox_normalise,
        'cap': cap,
        'cap_3std': cap_3std,
        'cat_dummies': cat_dummies,
        'categorize': categorize,
        'pass': do_nothing,
        'floor': floor,
        'floor_and_cap': floor_and_cap,
        'mean_impute': mean_impute,
        'median_impute': median_impute,
        'min_max_scale': min_max_scale,
        'mode_impute': mode_impute,
        'segment': segment,
        'z_normalise': z_normalise,
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
