from typing import List, Optional, Union

# Global constant for strict label ordering across evaluation and visualization
SPOILER_LABELS: List[str] = ["phrase", "passage", "multi"]


def join_text_list(value: Union[str, List[str]], limit: Optional[int] = None) -> str:
    """Joins a list of strings into one space-separated string, optionally truncated.

    Stringifies non-list inputs as-is. Centralizes sequence flattening patterns
    previously duplicated across multiple classifiers and pipelines.

    Args:
        value (Union[str, List[str]]): A list of text segments, or an already-scalar string.
        limit (int, optional): If given, only the first `limit` list items are joined.

    Returns:
        str: The joined (and optionally truncated) space-separated string.
    """
    if isinstance(value, list):
        return " ".join(value[:limit] if limit else value)
    return str(value)


def extract_primary_tag(tag_input: Union[str, List[str]]) -> str:
    """Extracts the first string tag if the input is a list, otherwise returns the raw string safely.

    Args:
        tag_input (Union[str, List[str]]): The raw tag field extracted from the dataset.

    Returns:
        str: The cleaned primary scalar string tag, or an empty string if input is invalid.
    """
    if not tag_input:
        return ""
    if isinstance(tag_input, list):
        return str(tag_input[0]) if len(tag_input) > 0 else ""
    return str(tag_input)
