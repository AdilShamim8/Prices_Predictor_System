"""
data_ingestion_step.py — ZenML Step: Load Raw Data from ZIP Archive
====================================================================

PURPOSE:
    This is a ZenML pipeline step that loads the Ames Housing dataset from a ZIP file.
    It delegates the actual extraction work to src/ingest_data.py (Factory pattern).

PIPELINE POSITION:  Step 1 of 7 in the training pipeline.
INPUT:              file_path (str) — path to data/archive.zip
OUTPUT:             pd.DataFrame    — raw dataset (~2,930 rows × 82 columns)
"""

import logging

import pandas as pd
from src.ingest_data import DataIngestorFactory
from zenml import step

logger = logging.getLogger(__name__)


@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """Ingest data from a ZIP file using the appropriate DataIngestor.

    Args:
        file_path: Absolute path to the ZIP archive containing the dataset.

    Returns:
        Raw DataFrame loaded from the archive.

    Raises:
        FileNotFoundError: If the zip archive does not exist at the given path.
        ValueError: If no suitable ingestor is found for the file extension.
    """
    logger.info("Starting data ingestion from: %s", file_path)

    # Determine the file extension
    file_extension = ".zip"  # Since we're dealing with ZIP files, this is hardcoded

    # Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.ingest(file_path)

    logger.info(
        "Data ingestion complete. Loaded %d rows and %d columns.",
        len(df),
        len(df.columns),
    )
    return df
