import numpy as np
import pandas as pd

from typing import Union
from functools import wraps


def is_binary(
        y: Union[pd.Series, np.ndarray]
) -> bool:
    if type(y) == pd.Series:
        y = y.values

    if len(np.unique(y)) <= 2:
        return True
    else:
        return False


def is_categorical(
    y: Union[pd.Series, np.ndarray]
) -> bool:
    if type(y) == pd.Series:
        y = y.values

    if y.dtype == 'object' or determine_problem_type(y) == 'classification':
        return True
    else:
        return False


def determine_problem_type(
        target: Union[pd.Series, np.ndarray],
        classification_threshold: int = 30
) -> str:
    """
    Determine if a problem is a regression problem or a classification problem
    based on the unique values in the target series.

    Args:
        target (pd.Series): The target variable.
        classification_threshold (int): The maximum number of unique values a
        target variable can have to be considered a classification problem.

    Returns:
        str: "regression" or "classification" based on the nature of the target.
    """

    if type(target) == pd.Series:
        target = target.values

    # Adjust threshold in case of small datasets
    classification_threshold = min(
        max(int(len(target) / 10), 2),
        classification_threshold
    )

    # Calculate the number of unique values in the target
    unique_values = np.unique(target)

    # If number of unique values is less than or equal to the classification threshold,
    # we consider it a classification problem. Otherwise, it's a regression problem.
    if len(unique_values) <= classification_threshold:
        return "classification"
    else:
        return "regression"


def seed(func):
    @wraps(func)
    def function_wrapper(*args, **kwargs):
        orig_seed = np.random.get_state()
        results = func(*args, **kwargs)
        np.random.set_state(orig_seed)
        return results
    return function_wrapper
