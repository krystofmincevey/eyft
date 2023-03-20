import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.eyft import logger


def explore(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    max_cats: int = 15,
):
    """
    Analyses column x_col in df, generating both descriptive
    statistics, and a seaborn plot (x_col vs y_col).
    Note that if the number of unique values in x_col exceeds
    max_cats, the column is treated as numerical.
    Otherwise, the column is treated as a categorical
    col.
    """

    print('---'*10)
    print(f'Analysing X: {x_col} | Y: {y_col}')
    print(df[x_col].describe())

    if df[x_col].nunique() > max_cats:
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        sns.scatterplot(x=df[x_col], y=df[y_col])
    else:
        try:
            sns.boxplot(x=x_col, y=y_col, data=df)
        except ValueError:
            logger.info(
                f'Unable to plot X: {x_col} vs Y: {y_col}.'
            )
            pass

    plt.show()
