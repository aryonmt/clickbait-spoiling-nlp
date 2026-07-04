from typing import List

import nltk
import pandas as pd

# Ensure both punkt and punkt_tab are safely downloaded for newer NLTK versions
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


class HeuristicSpoilerGenerator:
    """Generates spoilers using rule-based heuristics based on predicted spoiler

    types and empirical length constraints.
    """

    @staticmethod
    def _extract_heuristic(paragraphs: List[str], tag: str) -> List[str]:
        """Applies heuristic rules on target paragraphs to extract a baseline spoiler.

        Args:
            paragraphs (List[str]): List of paragraphs from the source article.
            tag (str): Predicted spoiler tag ('phrase', 'passage', or 'multi').

        Returns:
            List[str]: Extracted baseline spoiler tokens or phrases.
        """
        if not paragraphs or not isinstance(paragraphs, list):
            return [""]

        first_para = str(paragraphs[0])

        if tag == "phrase":
            # Target phrase length: ~3-5 words from the very beginning of the article
            words = first_para.split()
            return [" ".join(words[:5])]

        elif tag == "passage":
            # Target passage length: return the complete first paragraph
            return [first_para]

        elif tag == "multi":
            # Target multi length: combine the first sentences of the first 3 paragraphs
            extracted_sentences: List[str] = []
            for para in paragraphs[:3]:
                sentences = nltk.sent_tokenize(str(para))
                if sentences:
                    extracted_sentences.append(sentences[0])
            return extracted_sentences

        return [first_para]

    def generate(self, df: pd.DataFrame, predicted_tags: List[str]) -> List[List[str]]:
        """Orchestrates rule-based heuristic generation across the entire dataset.

        Args:
            df (pd.DataFrame): Input dataset containing 'targetParagraphs'.
            predicted_tags (List[str]): Predicted spoiler type classes from Task 1 classifier.

        Returns:
            List[List[str]]: List of generated spoilers per instance.
        """
        if df.empty or not predicted_tags:
            return []

        print("Generating spoilers using heuristic rules...")
        generated_spoilers: List[List[str]] = []

        # Use enumerate to guarantee safe 0-based sequential lookup in predicted_tags list,
        # protecting against non-sequential or custom indexes in the input DataFrame.
        for i, (_, row) in enumerate(df.iterrows()):
            paragraphs = row.get("targetParagraphs", [])
            pred_tag = predicted_tags[i]
            spoiler = self._extract_heuristic(paragraphs, pred_tag)
            generated_spoilers.append(spoiler)

        return generated_spoilers
