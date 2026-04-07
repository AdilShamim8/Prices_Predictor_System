"""
feature_engineering_step.py — ZenML Step: Transform Features
=============================================================

PURPOSE:
    Applies a feature transformation (log, scaling, or encoding) to specified columns.
    Delegates to src/feature_engineering.py (Strategy pattern).

PIPELINE POSITION:  Step 3 of 7 in the training pipeline.
INPUT:              pd.DataFrame + strategy name + list of feature columns
OUTPUT:             pd.DataFrame with transformed features
DEFAULT:            Log transformation on ["Gr Liv Area", "SalePrice"]
"""

import pandas as pd
from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling,
)
from zenml import step


@step
def feature_engineering_step(
    df: pd.DataFrame, strategy: str = "log", features: list = None
) -> pd.DataFrame:
    """Performs feature engineering using FeatureEngineer and selected strategy."""

    # Ensure features is a list, even if not provided
    if features is None:
        features = []  # or raise an error if features are required

    if strategy == "log":
        engineer = FeatureEngineer(LogTransformation(features))
    elif strategy == "standard_scaling":
        engineer = FeatureEngineer(StandardScaling(features))
    elif strategy == "minmax_scaling":
        engineer = FeatureEngineer(MinMaxScaling(features))
    elif strategy == "onehot_encoding":
        engineer = FeatureEngineer(OneHotEncoding(features))
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")

    transformed_df = engineer.apply_feature_engineering(df)
    return transformed_df
