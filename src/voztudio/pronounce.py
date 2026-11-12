"""Rewrite brand names and loanwords before synthesis."""

from __future__ import annotations

import json
import re
from pathlib import Path


def parse_pronunciation_lines(text: str | None) -> dict[str, str]:
    """Parse ``FastHunting=Fést Rânting`` lines. Blank lines and ``#`` comments are ignored."""
    mapping: dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            mapping[key] = value
    return mapping


def load_pronunciation_file(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return parse_pronunciation_lines(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Pronunciation file must be a JSON object: {path}")
    mapping: dict[str, str] = {}
    for key, value in data.items():
        spoken = str(key).strip()
        replacement = str(value).strip()
        if spoken and replacement:
            mapping[spoken] = replacement
    return mapping


def apply_pronunciations(text: str, mapping: dict[str, str]) -> str:
    """Replace dictionary keys outside ``[style tags]``. Longer keys win."""
    if not mapping or not text:
        return text
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(
        "|".join(rf"(?<![\w]){re.escape(key)}(?![\w])" for key in keys),
        re.IGNORECASE,
    )
    lookup = {key.casefold(): mapping[key] for key in keys}

    def replace_plain(segment: str) -> str:
        return pattern.sub(lambda match: lookup.get(match.group(0).casefold(), match.group(0)), segment)

    pieces = re.split(r"(\[[^\[\]]*\])", text)
    return "".join(piece if piece.startswith("[") else replace_plain(piece) for piece in pieces)
