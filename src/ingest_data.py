"""
ingest_data.py — Data Ingestion using the Factory Design Pattern
================================================================

PURPOSE:
    This file is responsible for loading raw data from disk into a pandas DataFrame.
    It uses the **Factory pattern** so that different file formats (ZIP, CSV, etc.)
    can be handled by simply adding a new DataIngestor subclass — without changing
    any calling code.

HOW IT WORKS:
    1. DataIngestor (ABC)       — Defines the interface: every ingestor must have an `ingest()` method.
    2. ZipDataIngestor          — Concrete implementation: extracts a ZIP archive, finds the CSV inside,
                                  and loads it into a DataFrame.
    3. DataIngestorFactory      — Factory class: given a file extension (e.g. ".zip"), it returns
                                  the correct DataIngestor subclass instance.

USED BY:
    steps/data_ingestion_step.py  →  calls DataIngestorFactory.get_data_ingestor(".zip")

DATA FLOW:
    data/archive.zip  →  ZipDataIngestor.ingest()  →  extracted_data/AmesHousing.csv  →  pd.DataFrame
"""

import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd


# Define an abstract class for Data Ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Abstract method to ingest data from a given file."""
        pass


# Implement a concrete class for ZIP Ingestion
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Extracts a .zip file and returns the content as a pandas DataFrame."""
        # Ensure the file is a .zip
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")

        # Extract the zip file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")

        # Find the extracted CSV file (assuming there is one CSV file inside the zip)
        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found. Please specify which one to use.")

        # Read the CSV into a DataFrame
        csv_file_path = os.path.join("extracted_data", csv_files[0])
        df = pd.read_csv(csv_file_path)

        # Return the DataFrame
        return df


# Implement a Factory to create DataIngestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """Returns the appropriate DataIngestor based on file extension."""
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")


# Example usage:
if __name__ == "__main__":
    # # Specify the file path (relative to the project root)
    # file_path = "data/archive.zip"

    # # Determine the file extension
    # file_extension = os.path.splitext(file_path)[1]

    # # Get the appropriate DataIngestor
    # data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # # Ingest the data and load it into a DataFrame
    # df = data_ingestor.ingest(file_path)

    # # Now df contains the DataFrame from the extracted CSV
    # print(df.head())  # Display the first few rows of the DataFrame
    pass
