"""
training_pipeline.py — ZenML Pipeline: End-to-End Model Training
=================================================================

PURPOSE:
    Defines the complete training pipeline that takes raw data from a ZIP archive
    and produces a trained, evaluated scikit-learn model. This is the core pipeline
    of the project.

PIPELINE STEPS (executed in order):
    1. data_ingestion_step          → Load CSV from ZIP archive
    2. handle_missing_values_step   → Fill NaNs (default: mean)
    3. feature_engineering_step     → Log-transform skewed features
    4. outlier_detection_step       → Remove Z-score outliers
    5. data_splitter_step           → 80/20 train/test split
    6. model_building_step          → Train LinearRegression + log to MLflow
    7. model_evaluator_step         → Compute MSE and R² on test set

HOW TO RUN:
    python run_pipeline.py          (or: make train)
"""

from pathlib import Path

from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.outlier_detection_step import outlier_detection_step
from zenml import Model, pipeline, step

# Resolve data path relative to the project root (the parent of this file's directory)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_ARCHIVE = str(_PROJECT_ROOT / "data" / "archive.zip")


@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="prices_predictor"
    ),
)
def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path=_DATA_ARCHIVE
    )

    # Handling Missing Values Step
    filled_data = handle_missing_values_step(raw_data)

    # Feature Engineering Step
    engineered_data = feature_engineering_step(
        filled_data, strategy="log", features=["Gr Liv Area", "SalePrice"]
    )

    # Outlier Detection Step
    clean_data = outlier_detection_step(engineered_data, column_name="SalePrice")

    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column="SalePrice")

    # Model Building Step
    model = model_building_step(X_train=X_train, y_train=y_train)

    # Model Evaluation Step
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model


if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()
