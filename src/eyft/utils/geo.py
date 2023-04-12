import os
import openpyxl
import pandas as pd

from ..utils import logger
from ..pipelines.feature_engineering.transform import geolocate as get_location


def geolocate(
    dir_name: str,
    file_name: str,
    save_dir_name: str = "./data/geolocation",
    save_file_name: str = "location.pkl",
    chunk_size: int = 1000,
):
    file_path = os.path.join(dir_name, file_name)

    wb = openpyxl.load_workbook(
        file_path, enumerate
    )
    sheet = wb.worksheets[0]
    total = sheet.max_row

    for skip in range(0, total, chunk_size):
        df = pd.read_excel(file_path, skiprows=skip, nrows=chunk_size)

        df = get_location(df)

        save_path = os.path.join(
            save_dir_name,
            f"row_{skip+1}_{skip+chunk_size}_{save_file_name}"
        )

        logger.info(f'Saving dataframe: {df.shape} to: {save_path}')
        df.to_pickle(save_path)
        return df
