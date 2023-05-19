import re
import pandas as pd

from typing import Tuple, Dict, Any

from .selector import Selector
from ..feature_selection import logger


def select_best_features(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_col: str,
        params: Dict[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    steps = list(params.keys())
    steps.sort(key=lambda x: int(re.findall(r'\d+$', x)[0]))

    for key in steps:

        logger.info(
            f'Performing feature selection with params: {params[key]}'
        )

        fs = Selector(
            df=df_train,
            target_col=target_col
        )
        fs.feature_select(**params[key])
        features = fs.get_best_features()

        cols_to_keep = [target_col, *features]
        df_train, df_test = df_train[cols_to_keep], df_test[cols_to_keep]

    return df_train, df_test
