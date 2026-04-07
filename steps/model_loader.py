"""
model_loader.py — ZenML Step: Load a Trained Model from the ZenML Model Registry
==================================================================================

PURPOSE:
    Retrieves a previously trained sklearn Pipeline from ZenML's Model Control Plane.
    The model is looked up by name and the "production" version tag.

USED BY:
    Can be called manually or by custom pipelines that need to load the latest model.

OUTPUT:
    sklearn.pipeline.Pipeline — the full preprocessing + model pipeline, ready for prediction.
"""

from sklearn.pipeline import Pipeline
from zenml import Model, step


@step
def model_loader(model_name: str) -> Pipeline:
    """
    Loads the current production model pipeline.

    Args:
        model_name: Name of the Model to load.

    Returns:
        Pipeline: The loaded scikit-learn pipeline.
    """
    # Load the model by name and version
    model = Model(name=model_name, version="production")

    # Load the pipeline artifact (assuming it was saved as "sklearn_pipeline")
    model_pipeline: Pipeline = model.load_artifact("sklearn_pipeline")

    return model_pipeline
