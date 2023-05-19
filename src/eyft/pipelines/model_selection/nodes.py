import re
import pandas as pd

from typing import Tuple, Dict, Any, Union

from .modeler import SklearnModeler
from ..model_selection import logger


def process_bool(
    val: Any
) -> Any:
    if type(val) == str:
        if val in ['True', 'true', 'TRUE']:
            val = True
        elif val in ['False', 'false', 'FALSE']:
            val = False
        else:
            pass
    return val


def process_digit(
    val: Any
) -> Any:
    if type(val) == str:
        try:
            val = float(val)
        except TypeError or ValueError:
            pass
    return val


def process_yml(
    params: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Loading in a dictionary from a yml file
    can sometimes corrupt the data.
    Nones can be outputted as 'None', bools as strings  and
    numbers as strings. process_model2params
    converts digits to floats, True/False to bool,
    and 'None' to None.
    """
    # Replace string None with None and process numerics:
    updated_params = {}
    for model_name, search_space in params.items():
        updated_search_space = {}
        for hyper_param, param_search_space in search_space.items():

            if type(param_search_space) == list:
                updated_param_search_space = []
                if 'None' in param_search_space:
                    updated_param_search_space = param_search_space.copy()
                    updated_param_search_space.remove('None')
                    updated_param_search_space.append(None)
                else:
                    for val in param_search_space:
                        updated_param_search_space.append(
                            process_bool((process_digit(val)))  # only processed if numeric
                        )
            elif param_search_space == 'None':
                updated_param_search_space = None
            else:  # str, int, float
                updated_param_search_space = process_bool(process_digit(
                    param_search_space
                )) # only processed if actually numeric or bool else keep

            updated_search_space[hyper_param] = updated_param_search_space.copy()

        updated_params[model_name] = updated_search_space.copy()

    return updated_params


def model_search(
        df_train: pd.DataFrame,
        target_col: str,
        is_feature_select: bool,
        model2params: Dict[str, Dict[str, Any]],
        scoring: str,
        order_col: str = None,
        n_iter: int = 10,
        cv: int = 5,
) -> Dict[str, Tuple[Any, float, Dict[str, Any]]]:

    model2params = process_yml(model2params)
    n_iter = process_digit(n_iter)
    cv = process_digit(cv)

    msk = SklearnModeler(
        df=df_train,
        target_col=target_col,
        is_feature_select=is_feature_select,
        order_col=order_col
    )
    msk.search(
        model2params=model2params,
        scoring=scoring,
        n_iter=n_iter,
        cv=cv,
    )

    results = msk.get_search_results()

    return results
