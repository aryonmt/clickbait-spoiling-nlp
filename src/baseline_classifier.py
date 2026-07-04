from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.utils import extract_primary_tag, join_text_list


class BaselineSpoilerClassifier:
    """A baseline classifier utilizing TF-IDF features and a Logistic Regression model

    for Task 1 (Spoiler Type Classification).

    Attributes:
        vectorizer (TfidfVectorizer): Feature extractor for converting text to token count matrices.
        model (LogisticRegression): Linear classifier configured with balanced class weights.
    """

    def __init__(self):
        """Initializes the BaselineSpoilerClassifier with standard vectorizer parameters

        and a balanced logistic regression model.
        """
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=10000
        )
        self.model: LogisticRegression = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )

    def _prepare_features(self, df: pd.DataFrame) -> List[str]:
        """Combines the clickbait post text with the top paragraphs of the article

        to form a unified textual feature representation.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'postText' and 'targetParagraphs'.

        Returns:
            List[str]: A list of combined text features formatted with separator tokens.
        """
        combined_texts: List[str] = []
        for _, row in df.iterrows():
            post = join_text_list(row.get("postText", ""))
            # Safely join up to 5 paragraphs using the centralized utility function
            top_paragraphs = join_text_list(row.get("targetParagraphs", ""), limit=5)
            combined_texts.append(f"{post} [SEP] {top_paragraphs}")
        return combined_texts

    @staticmethod
    def _extract_labels(df: pd.DataFrame) -> List[str]:
        """Extracts and flattens target labels from the tags column using central utilities.

        Args:
            df (pd.DataFrame): Input DataFrame containing the target 'tags' column.

        Returns:
            List[str]: Cleaned list of primary target labels.
        """
        if "tags" not in df.columns:
            return []
        return df["tags"].apply(extract_primary_tag).tolist()

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fits the TF-IDF vectorizer and trains the Logistic Regression classifier.

        Args:
            train_df (pd.DataFrame): Training dataset with features and target labels.

        Raises:
            ValueError: If the training dataset is empty or missing essential columns.
        """
        if train_df.empty:
            raise ValueError(
                "[ERROR] Cannot fit baseline classifier on an empty DataFrame."
            )

        print("Extracting features and training baseline classifier...")
        X_train_text = self._prepare_features(train_df)
        y_train = self._extract_labels(train_df)

        if not y_train:
            raise ValueError(
                "[ERROR] No valid labels extracted. Verify the 'tags' column exists."
            )

        X_train_vec = self.vectorizer.fit_transform(X_train_text)
        self.model.fit(X_train_vec, y_train)
        print("[SUCCESS] Classifier training completed.")

    def predict(self, df: pd.DataFrame) -> List[str]:
        """Predicts the spoiler types ('phrase', 'passage', 'multi') for the target dataset.

        Args:
            df (pd.DataFrame): Target evaluation dataset.

        Returns:
            List[str]: List of predicted scalar string labels.
        """
        if df.empty:
            return []

        X_text = self._prepare_features(df)
        X_vec = self.vectorizer.transform(X_text)
        return self.model.predict(X_vec).tolist()
