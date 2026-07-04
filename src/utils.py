from typing import List, Optional, Union

# Global constant for strict label ordering across evaluation and visualization
SPOILER_LABELS = ["phrase", "passage", "multi"]


def join_text_list(value: Union[str, List[str]], limit: Optional[int] = None) -> str:
    """Joins a list of strings into one space-separated string, optionally

    truncated to the first `limit` items; stringifies non-list input as-is.
    Centralizes a pattern ("post"/"targetParagraphs" flattening) that was
    previously duplicated across baseline_classifier, transformer_classifier,
    inference_transformer, and main_integrated_pipeline.

    Args:
        value (Union[str, List[str]]): A list of text segments (e.g. postText
          words or article paragraphs), or an already-scalar string.
        limit (Optional[int]): If given, only the first `limit` list items
          are joined (e.g. the first N paragraphs of an article).

    Returns:
        str: The joined (and optionally truncated) string.
    """
    if isinstance(value, list):
        return " ".join(value[:limit] if limit else value)
    return str(value)


def extract_primary_tag(tag_input: Union[str, List[str]]) -> str:
    """Extracts the first string tag if the input is a list, otherwise returns
    the raw string safely. Returns an empty string if the input is empty or None.

    Args:
        tag_input (Union[str, List[str]]): The tag field extracted from the
          dataset.

    Returns:
        str: The cleaned primary scalar tag.
    """
    if not tag_input:
        return ""
    if isinstance(tag_input, list):
        return str(tag_input[0]) if len(tag_input) > 0 else ""
    return str(tag_input)
