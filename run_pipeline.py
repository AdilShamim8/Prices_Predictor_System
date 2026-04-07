"""
run_pipeline.py — Entry Point: Train the Model
================================================

PURPOSE:
    Command-line entry point that triggers the full ZenML training pipeline.
    After the pipeline finishes, it prints the MLflow tracking URI so you can
    inspect experiment runs in the MLflow UI.

HOW TO RUN:
    python run_pipeline.py      (or: make train)

WHAT HAPPENS:
    1. Runs ml_pipeline() — the 7-step training pipeline
    2. Prints the MLflow UI command so you can view metrics and artifacts
"""

import click
from pipelines.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


@click.command()
def main():
    """
    Run the ML pipeline and start the MLflow UI for experiment tracking.
    """
    # Run the pipeline
    run = ml_pipeline()

    # You can uncomment and customize the following lines if you want to retrieve and inspect the trained model:
    # trained_model = run["model_building_step"]  # Replace with actual step name if different
    # print(f"Trained Model Type: {type(trained_model)}")

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the experiment."
    )


if __name__ == "__main__":
    main()
