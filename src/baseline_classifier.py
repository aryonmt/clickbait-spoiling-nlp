from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.utils import extract_primary_tag, join_text_list


class BaselineSpoilerClassifier:
    """A baseline classifier using TF-IDF and Logistic Regression for Task 1

    (Spoiler Type Classification).
    """

    def __init__(self):
        """Initializes the vectorizer and the linear model with balanced class

        weights.
        """
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=10000
        )
        self.model = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )

    def _prepare_features(self, df: pd.DataFrame) -> List[str]:
        """Combines the clickbait post text with the first 5 paragraphs of the

        article.

        Args:
            df (pd.DataFrame): Input DataFrame containing challenge data.

        Returns:
            List[str]: Combined textual features for each row.
        """
        combined_texts = []
        for _, row in df.iterrows():
            post = join_text_list(row["postText"])
            top_paragraphs = (
                join_text_list(row["targetParagraphs"], limit=5)
                if isinstance(row["targetParagraphs"], list)
                else ""
            )
            combined_texts.append(f"{post} [SEP] {top_paragraphs}")
        return combined_texts

    @staticmethod
    def _extract_labels(df: pd.DataFrame) -> List[str]:
        """Extracts and flattens the tags column into string labels using central

        utilities.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            List[str]: Cleaned list of target labels.
        """
        return df["tags"].apply(extract_primary_tag).tolist()

    def fit(self, train_df: pd.DataFrame) -> None:
        """Trains the TF-IDF vectorizer and Logistic Regression model.

        Args:
            train_df (pd.DataFrame): Training dataset.
        """
        print("Extracting features and training baseline classifier...")
        X_train_text = self._prepare_features(train_df)
        y_train = self._extract_labels(train_df)

        X_train_vec = self.vectorizer.fit_transform(X_train_text)
        self.model.fit(X_train_vec, y_train)
        print("[SUCCESS] Classifier training completed.")

    def predict(self, df: pd.DataFrame) -> List[str]:
        """Predicts spoiler types for the given dataset.

        Args:
            df (pd.DataFrame): Target dataset for inference.

        Returns:
            List[str]: Predicted labels ('phrase', 'passage', 'multi').
        """
        X_text = self._prepare_features(df)
        X_vec = self.vectorizer.transform(X_text)
        return self.model.predict(X_vec).tolist()
