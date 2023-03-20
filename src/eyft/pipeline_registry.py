"""Project pipelines."""
from typing import Dict

from .pipelines.data_processing.pipeline import (
    create_pipeline as process_inputs
)
from .pipelines.feature_engineering.pipeline import (
    create_pipeline as transform_inputs
)
from kedro.pipeline import Pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    dp_pipeline = process_inputs()
    fe_pipeline = transform_inputs()

    return {
        "__default__": dp_pipeline + fe_pipeline,
        "data_processing": dp_pipeline,
        "feature_engineering": fe_pipeline,
    }
