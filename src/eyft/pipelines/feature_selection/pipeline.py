from typing import Dict, Any
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import select_best_features


def create_pipeline(
    inputs: Dict[str, Any] = {
        "train": "complete_features_train",
        "test": "complete_features_test",
    }, outputs: Dict[str, Any] = {
        "train": "final_train",
        "test": "final_test",
    },
) -> Pipeline:
    return pipeline(
        [
            node(
                func=select_best_features,
                inputs={
                    "df_train": inputs["train"],
                    "df_test": inputs["test"],
                    "target_col": "params:target_col",
                    "params": "params:fs_params",
                }, outputs=[
                    outputs["train"],
                    outputs["test"],
                ], name="feature_select",
            )
        ]
    )
