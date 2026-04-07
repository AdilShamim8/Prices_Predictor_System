"""
deployment_pipeline.py — ZenML Pipelines: Model Deployment & Batch Inference
=============================================================================

PURPOSE:
    Defines two pipelines:

    1. continuous_deployment_pipeline:
       - Runs the full training pipeline (ml_pipeline)
       - Deploys the trained model as an MLflow REST prediction server
       - Server listens at http://127.0.0.1:8000/invocations

    2. inference_pipeline:
       - Loads sample batch data (dynamic_importer)
       - Finds the running MLflow prediction server (prediction_service_loader)
       - Sends the batch data to the server for prediction (predictor)

HOW TO RUN:
    python run_deployment.py          (or: make deploy)
    python run_deployment.py --stop-service   (or: make stop-deploy)
"""

from pipelines.training_pipeline import ml_pipeline
from steps.dynamic_importer import dynamic_importer
from steps.model_loader import model_loader
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step


@pipeline
def continuous_deployment_pipeline():
    """Run a training job and deploy an MLflow model deployment."""
    # Run the training pipeline
    trained_model = ml_pipeline()  # No need for is_promoted return value anymore

    # (Re)deploy the trained model
    mlflow_model_deployer_step(workers=3, deploy_decision=True, model=trained_model)


@pipeline(enable_cache=False)
def inference_pipeline():
    """Run a batch inference job with data loaded from an API."""
    # Load batch data for inference
    batch_data = dynamic_importer()

    # Load the deployed model service
    model_deployment_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step",
    )

    # Run predictions on the batch data
    predictor(service=model_deployment_service, input_data=batch_data)
