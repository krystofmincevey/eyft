import pandas as pd
import numpy as np
import geopandas
import matplotlib.pyplot as plt
import re as re

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

    """

    :param df: dataset we use
    :param col: you can choose the column you want to impute
    :param mean: you can choose the imputation value, if None => we use the calculated mean
    :return: dataframe with the imputed column
    """
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

    """

    :param df: dataset we use
    :param col: you can choose the column you want to impute
    :param mode: you can choose the imputation value, if None => we use the calculated mode
    :return: dataframe with the imputed column
    """
    if mode is None:
        mode = float(df[col].mode())
        logger.info(f'Mode of {col} is {mode}.')

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
    :return: dataframe with the imputed column
    """
    if median is None:
        median = float(df[col].median())
        logger.info(f'Median of {col} is {median}.')

    df[col] = df[col].fillna(median)
    return {"df": df, "col": col, "median": median}

def z_normalise(
    df: pd.DataFrame,
    col: str,
    mean: Union[int, float] = None,
    stdev: Union[int, float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:

    """

    :param df: dataset we use
    :param col: you can choose the column you want to impute
    :param mean: you can choose the value used as mean in the standardization, if None => we use the calculated mean
    :param stdev: you can choose the value used as standard deviation in the standardization, if None => we use the calculated standard deviation
    :return: dataframe with the imputed column
    """
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

    """

    :param df: dataset we use
    :param col: you can choose the column you want to impute
    :param min_val: you can choose the value used as minimum in the scaling, if None => we use the calculated minimum
    :param max_val: you can choose the value used as maximum in the scaling, if None => we use the calculated maximum
    :return: dataframe with the imputed column
    """
    if min_val is None:
        min_val = df[col].min()
        logger.info(f'Min of {col} is {min_val}.')
    if max_val is None:
        max_val = df[col].max()
        logger.info(f'min of {col} is {max_val}.')

    df[col] = (df[col] - min_val) / (max_val - min_val)
    return {"df": df, "col": col, "min_val": min_val, "max_val": max_val}


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
    :param stdev: you can choose the value used as standard deviation, if None => we use the calculated standard deviation
    :return: dataframe with the imputed column
    """
    if mean is None:
        mean = df[col].mean()
        logger.info(f'mean of {col} is {mean}.')
    if stdev is None:
        stdev = df[col].std()
        logger.info(f'StDev of {col} is {stdev}.')
    df[col] = np.where(df[col] < mean - 3 * stdev, mean - 3 * stdev, df[col])
    df[col] = np.where(df[col] > mean + 3 * stdev, mean + 3 * stdev, df[col])
    return {"df": df, "col": col, "stdev": stdev}

def cap_perc(
    df: pd.DataFrame,
    col: str,
    cap: Union[float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:

    """

    :param df: dataset we use
    :param col: you can choose the column you want to impute
    :param cap: you can choose the percentile used for the capping of the outliers,
                if None => we use the 0.99 percentile
    :return: dataframe with the imputed column
    """
    if cap is None:
        cap_perc_value = df[col].quantile(0.99)
        logger.info(f'cap of {col} is {cap}.')
    elif cap <= 0 or cap >= 1:
        raise "You need to insert a number between 0 and 1"
    else:
        cap_perc_value = df[col].quantile(cap)
    df[col] = np.where(df[col] > cap_perc_value, cap_perc_value, df[col])
    return{"df": df, "col": col, "cap_perc": cap, "cap_perc_value": cap_perc_value}




def floor_perc(
    df: pd.DataFrame,
    col: str,
    floor: Union[float] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:

    """

    :param df: dataset we use
    :param col: you can choose the column you want to impute
    :param floor: you can choose the percentile used for the flooring of the outliers,
                  if None => we use the 0.01 percentile
    :return: dataframe with the imputed column
    """
    if floor is None:
        floor_perc_value = df[col].quantile(0.01)
        logger.info(f'floor of {col} is {floor_perc}.')
    elif floor <= 0 or floor >= 1:
        raise "You need to insert a number between 0 and 1"
    else:
        floor_perc_value = df[col].quantile(floor)
    df[col] = np.where(df[col] < floor_perc_value, floor_perc_value, df[col])
    return{"df": df, "col": col, "floor": floor, "floor_perc_value": floor_perc_value}




def dummy_var(
    df: pd.DataFrame,
    col: str,
    na_flag: Union[bool] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:

    """

    :param df: dataset we use
    :param col: you can choose the column you want to divide in dummy variables
    :param na_flag: you can choose if you want to include NaN as a dummy
    :return: dataframe with the dummy variables
    """
    if na_flag is None:
        na_flag = True
    df[col] = pd.get_dummies(df[col], dummy_na= na_flag)
    return{"df": df, "col": col, "na_flag": na_flag}

def categorize(
    df: pd.DataFrame,
    col: str,
    edges_cat: Union[list] is None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:

    """

    :param df: dataset we use
    :param col: you can choose the column that you want to categorize
    :param bins: numerical limits of the bins for the categories
    :return: values within the chosen categories
    """
    if edges_cat is None:
        edges_cat = np.histogram_bin_edges(df[col], bins="auto")
    df['new'] = pd.cut(df[col], bins=edges_cat)
    return{"df": df, "edges": edges_cat}

def merge_data(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_key: str,
    right_key: str,
    columns: list,
    merge_type: Union[str] = "left",
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:

    """

    :param left: dataset we want to enrich
    :param right: dataset used to add information to the left dataset
    :param left_key: field used for the matching in the target table
    :param right_key: field used for the matching in the right table
    :param columns: columns from the right table we want to add in the left table
    :param merge_type: what type of join you want to do
    :return: merged dataset you wanted
    """
    columns = [right_key] + columns
    right = right[columns]
    left = left.merge(right, left_on = left_key, right_on = right_key, how = merge_type)
    return{"df": left, "cols": columns}

def summ_statistics(
    df: pd.DataFrame,
    col: str,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:

    """

    :param df: dataset we use
    :param col: you can choose the column where you want to know the summary statistics from
    :return: the summary statistics of the chosen column
    """

    df[col] = df[col].describe

def boxplot(
    df: pd.DataFrame,
    col: str,
    by_user = str,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:

    """

    :param df: dataset we use
    :param col: you can choose the column you want to use to calculate a boxplot
    :param by_user: optional choice to perform a grouping of the targeted column
    :return: boxplot of the chosen column
    """

    b_plot = plt.boxplot(df[col], by=by_user )
    return{"b_plot": b_plot, "grouped_by": by_user}

"""
def boxplot(
    df: pd.DataFrame,
    col: str,
    box: Union[str] = None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:

    plt.boxplot(col = str, by = str)
    #WHAT I TRY TO DO IS CREATING A BOXPLOT WHERE THEY CAN COMPARE F.I. RANGE IN PRICE FOR DIFFERENT LOCATIONS
"""

def histogram(
    df: pd.DataFrame,
    col: str,
    edges: Union[list] is None,
) -> Dict[str, Union[pd.DataFrame, str, int, float]]:

    """

    :param df: dataset we use
    :param col: you can choose the column you want to use to calculate the histogram
    :param edges: numerical limits of the bins in the histogram
    :return: histogram of the chosen column
    """
    if edges is None:
        edges = np.histogram_bin_edges(df[col], bins="fd")
    plot = plt.hist(df[col], bins=edges)
    return{"plot": plot}





# move to dummy


def remove_symbols(
        df: pd.DataFrame,
        col: str,
        list_symbols: list,
        cleaning_list: list,
) -> Dict[str, Union[pd.DataFrame, str, list]]:

    """

    :param df: dataset we use
    :param col: you can choose the column you want to clean
    :param list_symbols: symbols you want to remove
    :param cleaning_list: what value needs to be put instead of the removed symbol
    :return: dataset with removed symbols
    """
    if len(list_symbols) != len(cleaning_list):
        raise "List of symbols length do not match with the cleaning values length"
    for n in range(list_symbols):
        df[col] = df[col].str.replace(list_symbols[n], cleaning_list[n])
    return {"df": df, "col": col}

def remove_spaces(
        df: pd.DataFrame,
        col: str,
):
    """

    :param df: dataset we use
    :param col: you can choose which column you want to clean
    :return: dataset with removed spaces
    """

    df[col] = df[col].str.re.replace(" +", " ").str.trim()



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
