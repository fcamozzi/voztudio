from __future__ import annotations

import re

from voztudio.config import TARGET_LANGUAGES


def validate_language(language: str) -> str:
    normalized = language.strip().lower()
    if normalized not in TARGET_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    return normalized


def chunk_text(text: str, max_chars: int = 500) -> list[str]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    text = text.strip()
    if not text:
        raise ValueError("Text must not be empty")
    chunks: list[str] = []
    for paragraph in re.split(r"\n+", text):
        words = paragraph.split()
        current: list[str] = []
        for word in words:
            candidate = " ".join(current + [word])
            if current and len(candidate) > max_chars:
                chunks.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            chunks.append(" ".join(current))
    return chunks
