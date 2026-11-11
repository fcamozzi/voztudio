import pytest

from voztudio.text import chunk_text, validate_language


def test_validate_language_accepts_exact_target_set():
    assert [validate_language(item) for item in ("en", "pt", "es", "hi", "zh", "fr")] == [
        "en", "pt", "es", "hi", "zh", "fr"
    ]


def test_validate_language_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported language"):
        validate_language("de")


def test_chunk_text_preserves_order_and_paragraph_boundaries():
    text = "First sentence.\n\nSecond sentence is longer.\nThird sentence."
    assert chunk_text(text, max_chars=25) == [
        "First sentence.",
        "Second sentence is", 
        "longer.",
        "Third sentence.",
    ]
