from typing import Any, Dict, List

import numpy as np
import torch


def predict_best_span(
    model,
    tokenizer,
    question: str,
    context: str,
    max_length: int,
    doc_stride: int,
    max_answer_length: int,
    n_best_size: int = 20,
    device: torch.device = None,
    suppressed_tokens: List[
        set
    ] = None,  # List of sets containing token indexes to mask per window
) -> Dict[str, Any]:
    """Extracts the global best answer span across all sliding windows with dynamic logit masking.

    Args:
        model: Question answering transformer model.
        tokenizer: Formatter tokenizer.
        question: The clickbait headline string.
        context: Source article target paragraphs.
        max_length: Maximum sequence window constraints.
        doc_stride: Sliding window stride steps.
        max_answer_length: Boundary constraint for predicted spans.
        n_best_size: Number of top logits candidates to evaluate per window.
        device: Active computing target (cuda or cpu).
        suppressed_tokens: List of token index sets to suppress per sliding window.

    Returns:
        Dict[str, Any]: Best predicted span information containing:
            'text', 'score', 'start_char', 'end_char'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize with sliding window settings matching preprocess configuration
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

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    start_logits = outputs.start_logits.cpu().numpy()
    end_logits = outputs.end_logits.cpu().numpy()
    offset_mappings = encodings["offset_mapping"]

    candidates = []

    # Loop through each overflowing text window (sliding feature slice)
    for w_idx in range(len(input_ids)):
        sequence_ids = encodings.sequence_ids(w_idx)
        offset_mapping = offset_mappings[w_idx]

        # Extract limits of valid context tokens (sequence_id == 1)
        context_token_indices = [
            idx for idx, s_id in enumerate(sequence_ids) if s_id == 1
        ]
        if not context_token_indices:
            continue
        context_start = context_token_indices[0]
        context_end = context_token_indices[-1]

        # Apply logit suppression masks for previously extracted tokens
        w_start_logits = start_logits[w_idx].copy()
        w_end_logits = end_logits[w_idx].copy()

        if (
            suppressed_tokens
            and w_idx < len(suppressed_tokens)
            and suppressed_tokens[w_idx]
        ):
            for token_idx in suppressed_tokens[w_idx]:
                w_start_logits[token_idx] = -10000.0
                w_end_logits[token_idx] = -10000.0

        # Rank top logits indexes within current context window
        start_indexes = np.argsort(w_start_logits)[::-1][:n_best_size]
        end_indexes = np.argsort(w_end_logits)[::-1][:n_best_size]

        for s_idx in start_indexes:
            for e_idx in end_indexes:
                # Enforce context, ordering and length constraints
                if s_idx < context_start or s_idx > context_end:
                    continue
                if e_idx < context_start or e_idx > context_end:
                    continue
                if e_idx < s_idx:
                    continue

                span_len = e_idx - s_idx + 1
                if span_len > max_answer_length:
                    continue

                # Map token boundaries back to raw context characters
                start_char = int(offset_mapping[s_idx][0])
                end_char = int(offset_mapping[e_idx][1])
                text_slice = context[start_char:end_char].strip()

                score = float(w_start_logits[s_idx] + w_end_logits[e_idx])
                candidates.append(
                    {
                        "text": text_slice,
                        "score": score,
                        "start_char": start_char,
                        "end_char": end_char,
                    }
                )

    # Global sort descending by score across all windows combined
    if not candidates:
        return {"text": "", "score": -10000.0, "start_char": 0, "end_char": 0}

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]


from typing import Any, Dict, List

import torch


def predict_best_span(
    model,
    tokenizer,
    question: str,
    context: str,
    max_length: int,
    doc_stride: int,
    max_answer_length: int,
    n_best_size: int = 20,
    device: torch.device = None,
    suppressed_tokens: List[
        set
    ] = None,  # List of sets containing token indexes to mask per window
) -> Dict[str, Any]:
    """Extracts the global best answer span across all sliding windows with dynamic logit masking.

    Args:
        model: Question answering transformer model.
        tokenizer: Formatter tokenizer.
        question: The clickbait headline string.
        context: Source article target paragraphs.
        max_length: Maximum sequence window constraints.
        doc_stride: Sliding window stride steps.
        max_answer_length: Boundary constraint for predicted spans.
        n_best_size: Number of top logits candidates to evaluate per window.
        device: Active computing target (cuda or cpu).
        suppressed_tokens: List of token index sets to suppress per sliding window.

    Returns:
        Dict[str, Any]: Best predicted span information containing:
            'text', 'score', 'start_char', 'end_char'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize with sliding window settings matching preprocess configuration
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

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    start_logits = outputs.start_logits.cpu().numpy()
    end_logits = outputs.end_logits.cpu().numpy()
    offset_mappings = encodings["offset_mapping"]

    candidates = []

    # Loop through each overflowing text window (sliding feature slice)
    for w_idx in range(len(input_ids)):
        sequence_ids = encodings.sequence_ids(w_idx)
        offset_mapping = offset_mappings[w_idx]

        # Extract limits of valid context tokens (sequence_id == 1)
        context_token_indices = [
            idx for idx, s_id in enumerate(sequence_ids) if s_id == 1
        ]
        if not context_token_indices:
            continue
        context_start = context_token_indices[0]
        context_end = context_token_indices[-1]

        # Apply logit suppression masks for previously extracted tokens
        w_start_logits = start_logits[w_idx].copy()
        w_end_logits = end_logits[w_idx].copy()

        if (
            suppressed_tokens
            and w_idx < len(suppressed_tokens)
            and suppressed_tokens[w_idx]
        ):
            for token_idx in suppressed_tokens[w_idx]:
                w_start_logits[token_idx] = -10000.0
                w_end_logits[token_idx] = -10000.0

        # Rank top logits indexes within current context window
        start_indexes = np.argsort(w_start_logits)[::-1][:n_best_size]
        end_indexes = np.argsort(w_end_logits)[::-1][:n_best_size]

        for s_idx in start_indexes:
            for e_idx in end_indexes:
                # Enforce context, ordering and length constraints
                if s_idx < context_start or s_idx > context_end:
                    continue
                if e_idx < context_start or e_idx > context_end:
                    continue
                if e_idx < s_idx:
                    continue

                span_len = e_idx - s_idx + 1
                if span_len > max_answer_length:
                    continue

                # Strict overlap/suppression check to prevent overlapping multi-spoilers.
                # Stops candidate spans from bridging over or fully enclosing previously suppressed tokens.
                if (
                    suppressed_tokens
                    and w_idx < len(suppressed_tokens)
                    and suppressed_tokens[w_idx]
                ):
                    if any(
                        t_idx in suppressed_tokens[w_idx]
                        for t_idx in range(s_idx, e_idx + 1)
                    ):
                        continue

                # Map token boundaries back to raw context characters
                start_char = int(offset_mapping[s_idx][0])
                end_char = int(offset_mapping[e_idx][1])
                text_slice = context[start_char:end_char].strip()

                score = float(w_start_logits[s_idx] + w_end_logits[e_idx])
                candidates.append(
                    {
                        "text": text_slice,
                        "score": score,
                        "start_char": start_char,
                        "end_char": end_char,
                    }
                )

    # Global sort descending by score across all windows combined
    if not candidates:
        return {"text": "", "score": -10000.0, "start_char": 0, "end_char": 0}

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]
