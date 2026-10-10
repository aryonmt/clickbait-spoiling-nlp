from typing import List

import nltk

from src.logging_setup import setup_logger

logger = setup_logger("retrieval_generator")

# Ensure both punkt and punkt_tab are safely downloaded for newer NLTK versions
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


class RetrievalSpoilerGenerator:
    """Generates multi spoilers by ranking article sentences with token Jaccard overlap against the post.
    """

    @staticmethod
    def _compute_jaccard(tokens_a: set, tokens_b: set) -> float:
        """Calculates Jaccard Similarity between two sets of tokens.

        Args:
            tokens_a (set): First token set.
            tokens_b (set): Second token set.

        Returns:
            float: Calculated Jaccard coefficient (intersection over union).
        """
        intersection = tokens_a.intersection(tokens_b)
        union = tokens_a.union(tokens_b)
        return len(intersection) / len(union) if union else 0.0

    @classmethod
    def generate_multi_spoiler_jaccard(
        cls, post_text: str, paragraphs: List[str], top_k: int = 3
    ) -> List[str]:
        """Selects top-k sentences across all paragraphs ranked by Jaccard similarity to the clickbait post.

        Args:
            post_text (str): The clickbait headline string.
            paragraphs (List[str]): All paragraphs from the source article.
            top_k (int): Number of highest ranking sentences to extract. Defaults to 3.

        Returns:
            List[str]: Extracted top-k spoiler sentences.
        """
        if not paragraphs:
            return []

        post_words = set(post_text.lower().split())
        if not post_words:
            return [paragraphs[0]] if paragraphs else []

        scored_sentences = []
        sentence_idx = 0

        for para_idx, para in enumerate(paragraphs):
            sentences = nltk.sent_tokenize(para)
            for sent in sentences:
                sent_words = set(sent.lower().split())
                score = cls._compute_jaccard(post_words, sent_words)

                scored_sentences.append(
                    {"text": sent.strip(), "idx": sentence_idx, "score": score}
                )
                sentence_idx += 1

        # Sort descending by score, ascending by original document index
        scored_sentences.sort(key=lambda x: (-x["score"], x["idx"]))
        top_sentences = scored_sentences[:top_k]

        # Sort selected sentences back to original chronological document order
        top_sentences.sort(key=lambda x: x["idx"])
        return [item["text"] for item in top_sentences]

    @classmethod
    def generate_multi_spoiler(
        cls,
        post_text: str,
        paragraphs: List[str],
        top_k: int = 3,
        method: str = "jaccard",
    ) -> List[str]:
        """Rank article sentences against the post and return the top matches.

        Args:
            post_text: Clickbait headline.
            paragraphs: Article paragraphs.
            top_k: Number of sentences to return. Defaults to 3.
            method: Only ``jaccard`` is implemented. Any other value, including
                the former ``tfidf`` option, is logged and treated as Jaccard
                so callers do not hit a missing ranker.

        Returns:
            Spoiler sentences in their original document order.
        """
        if method != "jaccard":
            logger.warning(
                "Retrieval method %r is not implemented; using Jaccard overlap.",
                method,
            )
        return cls.generate_multi_spoiler_jaccard(post_text, paragraphs, top_k)
