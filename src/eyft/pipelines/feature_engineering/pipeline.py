from typing import Dict
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import transform


def create_pipeline(
    inputs: Dict[str, str] = {
        "train": "processed_train",
        "test": "processed_test",
    }, outputs: Dict[str, str] = {
        "train": "complete_features_train",
        "test": "complete_features_test",
    },
) -> Pipeline:
    return pipeline(
        [
            node(
                func=transform,
                inputs={
                    "df": inputs["train"],
                    "params": "params:feature_engineering.cols",
                }, outputs="temp_train_fe",
                name="transform_train_data_node_a",
            ),
            node(
                func=transform,
                inputs={
                    "df": "temp_train_fe",
                    "params": "params:feature_engineering.transformed_cols",
                }, outputs=outputs["train"],
                name="transform_train_data_node_b",
            ),
            node(
                func=transform,
                inputs={
                    "df": inputs["test"],
                    "params": "params:feature_engineering.cols",
                }, outputs="temp_test_fe",
                name="transform_test_data_node_a",
            ),
            node(
                func=transform,
                inputs={
                    "df": "temp_test_fe",
                    "params": "params:feature_engineering.transformed_cols",
                }, outputs=outputs["test"],
                name="transform_test_data_node_b",
            ),
        ]
    )
