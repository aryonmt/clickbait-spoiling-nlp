from typing import List

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    """Generates multi-type spoilers by dynamically retrieving and ranking sentences

    from the article based on lexical and vector similarity to the clickbait post.
    """

    @staticmethod
    def _compute_jaccard(tokens_a: set, tokens_b: set) -> float:
        """Calculates Jaccard Similarity between two sets of tokens."""
        intersection = tokens_a.intersection(tokens_b)
        union = tokens_a.union(tokens_b)
        return len(intersection) / len(union) if union else 0.0

    @classmethod
    def generate_multi_spoiler_jaccard(
        cls, post_text: str, paragraphs: List[str], top_k: int = 3
    ) -> List[str]:
        """Selects top-k sentences across all paragraphs ranked by Jaccard similarity.

        Args:
            post_text: The clickbait headline string.
            paragraphs: All paragraphs from the source article.
            top_k: Number of highest ranking sentences to extract.

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
    def generate_multi_spoiler_tfidf(
        cls, post_text: str, paragraphs: List[str], top_k: int = 3
    ) -> List[str]:
        """Selects top-k sentences across all paragraphs ranked by TF-IDF Cosine Similarity.

        Args:
            post_text: The clickbait headline string.
            paragraphs: All paragraphs from the source article.
            top_k: Number of highest ranking sentences to extract.

        Returns:
            List[str]: Extracted top-k spoiler sentences.
        """
        if not paragraphs:
            return []

        # Flatten paragraphs into a flat list of sentences
        all_sentences = []
        sentence_idx = 0
        sentence_metadata = []

        for para_idx, para in enumerate(paragraphs):
            sentences = nltk.sent_tokenize(para)
            for sent in sentences:
                clean_sent = sent.strip()
                if clean_sent:
                    all_sentences.append(clean_sent)
                    sentence_metadata.append({"text": clean_sent, "idx": sentence_idx})
                    sentence_idx += 1

        if not all_sentences:
            return []

        # Include the clickbait post at the end of the corpus for vocabulary fitting
        corpus = all_sentences + [post_text]

        try:
            # Fit and transform the corpus with English stop words
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
            tfidf_matrix = vectorizer.fit_transform(corpus)

            # The last vector is the clickbait post query
            post_vector = tfidf_matrix[-1]
            sentence_vectors = tfidf_matrix[:-1]

            # Compute cosine similarities between sentences and the query post
            similarities = cosine_similarity(sentence_vectors, post_vector).flatten()

            # Populate scores into metadata
            for i, score in enumerate(similarities):
                sentence_metadata[i]["score"] = float(score)

        except Exception as exc:
            # Safe fallback if TF-IDF vectorizer collapses (e.g. empty vocabulary)
            logger.warning(
                "TF-IDF similarity computation failed (%s); falling back to "
                "zero scores for all sentences in this sample.",
                exc,
            )
            for item in sentence_metadata:
                item["score"] = 0.0

        # Sort descending by score, ascending by document index
        sentence_metadata.sort(key=lambda x: (-x["score"], x["idx"]))
        top_items = sentence_metadata[:top_k]

        # Sort back to chronological document order
        top_items.sort(key=lambda x: x["idx"])

        return [item["text"] for item in top_items]

    @classmethod
    def generate_multi_spoiler(
        cls,
        post_text: str,
        paragraphs: List[str],
        top_k: int = 3,
        method: str = "tfidf",
    ) -> List[str]:
        """Orchestrator method for multi-type spoiler generation.

        Args:
            post_text: The clickbait headline string.
            paragraphs: All paragraphs from the source article.
            top_k: Number of highest ranking sentences to extract.
            method: Similarity method ('jaccard' or 'tfidf').

        Returns:
            List[str]: Extracted top-k spoiler sentences.
        """
        if method == "jaccard":
            return cls.generate_multi_spoiler_jaccard(post_text, paragraphs, top_k)
        return cls.generate_multi_spoiler_tfidf(post_text, paragraphs, top_k)
