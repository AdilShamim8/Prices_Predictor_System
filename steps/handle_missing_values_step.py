"""
handle_missing_values_step.py — ZenML Step: Clean Missing Values
=================================================================

PURPOSE:
    Fills or drops missing (NaN) values in the DataFrame.
    Delegates to src/handle_missing_values.py (Strategy pattern).

PIPELINE POSITION:  Step 2 of 7 in the training pipeline.
INPUT:              pd.DataFrame with potential NaN values
OUTPUT:             pd.DataFrame with NaN values handled
DEFAULT STRATEGY:   "mean" — fills numeric NaNs with column means
"""

import pandas as pd
from src.handle_missing_values import (
    DropMissingValuesStrategy,
    FillMissingValuesStrategy,
    MissingValueHandler,
)
from zenml import step


@step
def handle_missing_values_step(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Handles missing values using MissingValueHandler and the specified strategy."""

    if strategy == "drop":
        handler = MissingValueHandler(DropMissingValuesStrategy(axis=0))
    elif strategy in ["mean", "median", "mode", "constant"]:
        handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy))
    else:
        raise ValueError(f"Unsupported missing value handling strategy: {strategy}")

    cleaned_df = handler.handle_missing_values(df)
    return cleaned_df
