import pandas as pd

from typing import Dict, List

from transform import Transform


def transform(
    df: pd.DataFrame,
    params: Dict[str, List[str]],
) -> pd.DataFrame:
    trans = Transform(df)
    trans.transform(params=params)
    return trans.get_df()
