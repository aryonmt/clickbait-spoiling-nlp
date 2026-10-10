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
    """Trims phrase spoilers to at most 6 words. Passage spoilers shorter than

    6 words are replaced by the sentence that contains the extracted span.

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
        # Keep at most 6 words. The check is len(words) > 6, so a 6-word phrase stays.
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
