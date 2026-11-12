from __future__ import annotations

import re

from voztudio.config import TARGET_LANGUAGES

_SENTENCE_BREAK = re.compile(r"[.!?。！？；;]+")


def validate_language(language: str) -> str:
    normalized = language.strip().lower()
    if normalized not in TARGET_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    return normalized


def chunk_text(text: str, max_chars: int = 200) -> list[str]:
    """Pack sentences up to max_chars.

    S2-Pro does not slice an untagged script on chunk_length. One request is
    one generation pass capped at 1024 frames, so the client keeps chunks near
    200 characters and lets each request finish.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    text = text.strip()
    if not text:
        raise ValueError("Text must not be empty")
    chunks: list[str] = []
    for paragraph in re.split(r"\n+", text):
        chunks.extend(_pack(_sentences(paragraph), max_chars))
    return chunks


def _sentences(paragraph: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_BREAK.finditer(paragraph):
        piece = paragraph[start : match.end()].strip()
        if piece:
            parts.append(piece)
        start = match.end()
    tail = paragraph[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _pack(sentences: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        pieces = [sentence] if len(sentence) <= max_chars else _split_long(sentence, max_chars)
        for piece in pieces:
            if not current:
                current = piece
                continue
            if len(current) + 1 + len(piece) <= max_chars:
                current = f"{current} {piece}"
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


def _split_long(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if len(words) <= 1:
        raw = words[0] if words else text
        return [raw[index : index + max_chars] for index in range(0, len(raw), max_chars)]
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        if len(word) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.extend(word[index : index + max_chars] for index in range(0, len(word), max_chars))
            continue
        candidate = " ".join(current + [word])
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks
