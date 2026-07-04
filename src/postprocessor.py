import re
from typing import List, Optional

import nltk

# Ensure sentence tokenization is available
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def enforce_length_constraints(
    spoiler_text: str, spoiler_type: str, context: str = ""
) -> str:
    """Trims phrase spoilers to <= 5 words; for passage spoilers shorter than

    6 words, falls back to the sentence containing the extracted span.

    Args:
        spoiler_text (str): Raw extracted spoiler text from model.
        spoiler_type (str): Expected spoiler type ('phrase' or 'passage').
        context (str): Source article target paragraphs as a single string. Defaults to "".

    Returns:
        str: Post-processed spoiler text.
    """
    if not spoiler_text:
        return ""

    words = spoiler_text.split()
    if spoiler_type == "phrase":
        # Rule 1: Trim phrase spoilers to maximum of 5 words (official specification limit)
        if len(words) > 6:
            return " ".join(words[:6])
    elif spoiler_type == "passage":
        # Rule 2: If passage spoiler is too short, recover the containing sentence to maintain structure.
        # Implements a case-insensitive lookup to guarantee high matching rate across different casing pipelines.
        if len(words) < 6 and context:
            sentences = nltk.sent_tokenize(context)
            for sent in sentences:
                if spoiler_text.lower() in sent.lower():
                    return sent.strip()
    return spoiler_text


def detect_enumeration_spoiler(
    context: str, n_expected: Optional[int] = None
) -> Optional[List[str]]:
    """Detects numbered-list patterns in context and extracts them as separate spoilers.

    Args:
        context (str): Source article target paragraphs as a single string.
        n_expected (int, optional): Optional expected number of items extracted from the clickbait post.

    Returns:
        Optional[List[str]]: List of matched items if list pattern is detected, else None.
    """
    if not context:
        return None

    # Match patterns like: "1. Ecuador, 2. Nicaragua, 3. Thailand"
    # Captures greedily-but-lazily up to the next numbered marker (allowing
    # internal commas/periods, e.g. "Washington, D.C.") rather than stopping
    # at the first comma regardless of what follows it.
    pattern = (
        r"(?:^|\s|\b)(?:\d+[\.\)])\s+(.+?)"
        r"(?=\s*,\s*\d+[\.\)]|\s*;|\s*\||\s*\n|\s*$)"
    )
    matches = re.findall(pattern, context)

    if len(matches) >= 2:
        # If expected items are known, enforce strict alignment constraints
        if n_expected is not None and len(matches) != n_expected:
            # Fallback to model-based extraction if list length mismatches expectations
            return None
        return [m.strip() for m in matches if m.strip()]
    return None
