import pytest

from voztudio.text import chunk_text, validate_language


def test_validate_language_accepts_exact_target_set():
    assert [validate_language(item) for item in ("en", "pt", "es", "hi", "zh", "fr")] == [
        "en",
        "pt",
        "es",
        "hi",
        "zh",
        "fr",
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


def test_chunk_text_splits_cjk_sentences_and_long_runs():
    assert chunk_text("你好。世界！", max_chars=3) == ["你好。", "世界！"]
    assert chunk_text("甲" * 5, max_chars=2) == ["甲甲", "甲甲", "甲"]
    assert " " not in "".join(chunk_text("甲" * 5, max_chars=2))


def test_default_chunk_limit_is_one_s2_pass():
    assert len(chunk_text("word " * 30)) == 1
    assert len(chunk_text("word " * 80)) > 1
