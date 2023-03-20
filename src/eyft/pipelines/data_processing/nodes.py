import pandas as pd

from typing import Tuple, Dict, List

from .processor import Processor


def process(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        params: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dp = Processor(
        df_train=df_train,
        df_test=df_test
    )

    df_train_prc, df_test_prc = dp.process_data(params=params)
    return df_train_prc, df_test_prc
