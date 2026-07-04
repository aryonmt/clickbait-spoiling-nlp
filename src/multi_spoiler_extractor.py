from typing import List, Optional

import torch

from src.qa_inference import predict_best_span

try:
    from thefuzz import fuzz

    def compute_similarity(s1: str, s2: str) -> float:
        """Computes lexical similarity using thefuzz ratio."""
        return fuzz.ratio(s1, s2) / 100.0
except ImportError:
    from difflib import SequenceMatcher

    def compute_similarity(s1: str, s2: str) -> float:
        """Fallback lexical similarity using Python's SequenceMatcher."""
        return SequenceMatcher(None, s1, s2).ratio()


def generate_iterative_multi_spoiler(
    model,
    tokenizer,
    question: str,
    context: str,
    max_length: int,
    doc_stride: int,
    max_answer_length: int,
    max_iterations: int = 7,
    max_spoilers: int = 5,
    similarity_threshold: float = 0.6,
    device: Optional[torch.device] = None,
) -> List[str]:
    """Iteratively extracts non-redundant spoiler spans using a QA model and token-level logit suppression.

    Features a defensive guard against Slow Tokenizers to prevent runtime type errors.

    Args:
        model: Question answering transformer model.
        tokenizer: Fast tokenizer matching the model.
        question (str): The clickbait headline string.
        context (str): Source article target paragraphs.
        max_length (int): Maximum sequence token constraint.
        doc_stride (int): Sliding window stride steps.
        max_answer_length (int): Maximum length boundary for sub-segment predictions.
        max_iterations (int): Maximum extraction steps to run. Defaults to 7.
        max_spoilers (int): Maximum unique spoilers to extract. Defaults to 5.
        similarity_threshold (float): Jaccard/Fuzzy threshold to discard redundant answers. Defaults to 0.6.
        device (torch.device, optional): Computing target (cuda or cpu).

    Returns:
        List[str]: Extracted list of unique, non-overlapping multi-spoilers.
    """
    spoilers: List[str] = []

    # Format tokenize first to extract document window lengths
    encodings = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )
    num_windows = len(encodings["input_ids"])

    # List of sets tracking suppressed token indices per window
    suppressed_tokens = [set() for _ in range(num_windows)]

    for _ in range(max_iterations):
        if len(spoilers) >= max_spoilers:
            break

        # Predict the best non-suppressed span in context
        best_span = predict_best_span(
            model=model,
            tokenizer=tokenizer,
            question=question,
            context=context,
            max_length=max_length,
            doc_stride=doc_stride,
            max_answer_length=max_answer_length,
            device=device,
            suppressed_tokens=suppressed_tokens,
        )

        pred_text = best_span["text"].strip()
        # Exit iteration early if confidence scores drop too low
        if not pred_text or best_span["score"] < -5.0:
            break

        # Perform lexical redundancy verification
        is_redundant = False
        for existing in spoilers:
            if (
                compute_similarity(pred_text.lower(), existing.lower())
                > similarity_threshold
            ):
                is_redundant = True
                break

        if not is_redundant:
            spoilers.append(pred_text)

        # Force suppress all token indices covering the extracted character limits
        start_char = best_span["start_char"]
        end_char = best_span["end_char"]

        for w_idx in range(num_windows):
            offset_mapping = encodings["offset_mapping"][w_idx]
            sequence_ids = encodings.sequence_ids(w_idx)

            # Defensive Check: Protect against python crashes if the tokenizer
            # falls back to a Slow Tokenizer (which doesn't support sequence mapping).
            if sequence_ids is None:
                continue

            for idx, offsets in enumerate(offset_mapping):
                if sequence_ids[idx] == 1:  # Context tokens only
                    t_start, t_end = int(offsets[0]), int(offsets[1])
                    if max(t_start, start_char) < min(t_end, end_char):
                        suppressed_tokens[w_idx].add(idx)

    return spoilers
