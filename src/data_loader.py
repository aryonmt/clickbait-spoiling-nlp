import os
from typing import Optional

import pandas as pd


class JSONLLoader:
    """A professional loader for handling JSON Lines (.jsonl) datasets,

    ensuring structural integrity and providing baseline data validation.

    Attributes:
        file_path (str): Path to the target .jsonl dataset file.
        df (Optional[pd.DataFrame]): Cached DataFrame containing loaded data.
    """

    def __init__(self, file_path: str):
        """Initializes the loader with a specific dataset path.

        Args:
            file_path (str): Path to the target .jsonl file.
        """
        self.file_path: str = file_path
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Loads the JSONL file into a Pandas DataFrame.

        Returns:
            pd.DataFrame: Loaded dataset.

        Raises:
            FileNotFoundError: If the file does not exist at the specified path.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Dataset not found at target path: {self.file_path}"
            )

        print(f"Reading data from {self.file_path}...")
        self.df = pd.read_json(self.file_path, lines=True)
        return self.df

    def filter_by_tag(self, tag: str, reset_index: bool = True) -> "JSONLLoader":
        """Filters the loaded dataset to keep only samples matching a specific spoiler tag.

        Args:
            tag (str): The spoiler type to filter by (e.g. 'phrase', 'passage', 'multi').
            reset_index (bool): If True, resets row index labels to a contiguous 0-based range.
                Defaults to True (highly recommended to prevent list index errors during training).

        Returns:
            JSONLLoader: A new JSONLLoader instance containing the filtered DataFrame.
        """
        if self.df is None:
            self.load_data()

        filtered_loader = JSONLLoader(self.file_path)
        from src.utils import extract_primary_tag

        # Filter and optionally reset pandas row indices safely
        filtered_df = self.df[self.df["tags"].apply(extract_primary_tag) == tag].copy()
        if reset_index:
            filtered_df = filtered_df.reset_index(drop=True)

        filtered_loader.df = filtered_df

        print(
            f"Filtered dataset from {len(self.df)} to {len(filtered_loader.df)} samples of type '{tag}'."
        )
        return filtered_loader

    def run_sanity_checks(self) -> None:
        """Performs structural integrity, missing column, and null value validation checks

        on the loaded dataset.

        Raises:
            ValueError: If the dataset has not been loaded prior to calling this method.
        """
        if self.df is None:
            raise ValueError(
                "Data is not loaded. Call load_data() before running sanity checks."
            )

        print("\n=== Running Dataset Sanity Checks ===")
        print(f"Total Rows: {len(self.df)}")
        print(f"Total Columns: {len(self.df.columns)}")

        # Check for mandatory fields in Clickbait Spoiling challenge
        mandatory_fields = ["uuid", "postText", "targetParagraphs", "tags"]
        missing_fields = [
            field for field in mandatory_fields if field not in self.df.columns
        ]

        if missing_fields:
            print(f"[WARNING] Missing essential columns: {missing_fields}")
        else:
            print("[SUCCESS] All essential columns are present.")

        # Check for null values in critical columns
        null_counts = self.df[mandatory_fields].isnull().sum()
        print("\nMissing Values Count per Critical Column:")
        print(null_counts)

        # Detect entirely empty/null rows in critical columns
        empty_rows = self.df[mandatory_fields].isnull().all(axis=1).sum()
        if empty_rows > 0:
            print(
                f"[WARNING] Detected {empty_rows} entirely empty/null row(s) in critical columns."
            )
