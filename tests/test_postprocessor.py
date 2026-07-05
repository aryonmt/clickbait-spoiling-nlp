from src.postprocessor import detect_enumeration_spoiler, enforce_length_constraints


def test_enforce_length_constraints_phrase():
    """Trims long phrase predictions to maximum 6 words (optimized threshold)."""
    long_phrase = "Paris is the capital city of France"
    trimmed = enforce_length_constraints(long_phrase, "phrase")
    assert len(trimmed.split()) == 6
    assert trimmed == "Paris is the capital city of"


def test_enforce_length_constraints_passage():
    """Recovers full containing sentence if passage prediction is too short."""
    context = "Paris is the capital of France. It is a beautiful city."
    short_passage = "beautiful city"
    recovered = enforce_length_constraints(short_passage, "passage", context)
    assert recovered == "It is a beautiful city."


def test_detect_enumeration_spoiler():
    """Detects standard numbered-list patterns from context successfully."""
    context = "Here are the top countries: 1. Ecuador, 2. Nicaragua, 3. Thailand"
    spoilers = detect_enumeration_spoiler(context, n_expected=3)
    assert spoilers == ["Ecuador", "Nicaragua", "Thailand"]

    # Mismatch expected count returns None
    assert detect_enumeration_spoiler(context, n_expected=4) is None
