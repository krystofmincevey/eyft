from typing import Dict
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import process


def create_pipeline(
    inputs: Dict[str, str] = {
        "train": "preprocessed_train",
        "test": "preprocessed_test",
    }, outputs: Dict[str, str] = {
        "train": "processed_train",
        "test": "processed_test",
    },
) -> Pipeline:
    return pipeline(
        [
            node(
                func=process,
                inputs={
                    "df_train": inputs["train"],
                    "df_test": inputs["train"],  # TODO: change to inputs['test'] when available
                    "params": "params:data_processing.cols",
                }, outputs=[outputs["train"], outputs["test"]],
                name="preprocess_data_node",
            ),
        ]
    )
