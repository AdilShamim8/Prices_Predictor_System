"""
data_splitter_step.py — ZenML Step: Split Data into Train/Test Sets
====================================================================

PURPOSE:
    Splits the cleaned DataFrame into training features (X_train), testing features (X_test),
    training target (y_train), and testing target (y_test) using an 80/20 random split.
    Delegates to src/data_splitter.py (Strategy pattern).

PIPELINE POSITION:  Step 5 of 7 in the training pipeline.
INPUT:              pd.DataFrame + target_column name ("SalePrice")
OUTPUT:             (X_train, X_test, y_train, y_test) — four separate DataFrames/Series
"""

from typing import Tuple

import pandas as pd
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy
from zenml import step


@step
def data_splitter_step(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the data into training and testing sets using DataSplitter and a chosen strategy."""
    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
    X_train, X_test, y_train, y_test = splitter.split(df, target_column)
    return X_train, X_test, y_train, y_test
