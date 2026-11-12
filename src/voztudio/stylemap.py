"""Style notes for backends that do not speak Fish Speech [tags] directly."""

from __future__ import annotations

import re

_TAG = re.compile(r"\[([^\[\]\r\n]+)\]")
_LEADING_PAREN = re.compile(r"^\(([^()\n]*)\)\s*")

# Fish-style tags -> the parenthetical phrase VoxCPM2 reads.
# Upstream controllable cloning uses "(instruction)text", for example
# "(slightly faster, cheerful tone)This is a cloned voice".
# Phrases are Brazilian Portuguese because that is the content language.
STYLE_TO_VOX: dict[str, str] = {
    "whisper": "sussurrando",
    "whispering": "sussurrando",
    "sussurro": "sussurrando",
    "sussurrando": "sussurrando",
    "soft": "voz suave",
    "suave": "voz suave",
    "calm": "tom calmo",
    "calmo": "tom calmo",
    "slow": "mais devagar",
    "devagar": "mais devagar",
    "fast": "mais rápido",
    "quickly": "mais rápido",
    "rápido": "mais rápido",
    "rapido": "mais rápido",
    "excited": "animado",
    "excitedly": "animado",
    "animado": "animado",
    "happy": "tom alegre",
    "alegre": "tom alegre",
    "sad": "tom triste",
    "triste": "tom triste",
    "angry": "tom irritado",
    "irritado": "tom irritado",
    "shout": "em voz alta",
    "shouting": "em voz alta",
    "loud": "em voz alta",
    "alto": "em voz alta",
    "serious": "tom sério",
    "serio": "tom sério",
    "sério": "tom sério",
    "laugh": "rindo",
    "laughing": "rindo",
    "rindo": "rindo",
    "surprised": "surpreso",
    "surpreso": "surpreso",
    "breathy": "voz soprosa",
    "gentle": "tom gentil",
    "gentil": "tom gentil",
}

QWEN_STYLE_WARNING = (
    "Qwen3-TTS Base voice clone does not support style instructions. "
    "Tags such as [whisper] are removed and are not spoken."
)

HINDI_WARNING = (
    "Qwen3-TTS does not list Hindi. The language label is stored with the file, "
    "and synthesis uses Auto."
)


def _tidy(text: str) -> str:
    cleaned = _TAG.sub("", text)
    cleaned = re.sub(r"\s+([.!?。！？；;])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def strip_qwen_style(text: str, style: str | None) -> tuple[str, tuple[str, ...]]:
    """Drop style tags. Base voice clone has no instruct channel."""
    cleaned = _tidy(text)
    leading = _LEADING_PAREN.match(cleaned)
    if leading:
        cleaned = cleaned[leading.end() :].strip()
    note = (style or "").strip()
    if note or _TAG.search(text) or leading:
        return cleaned, (QWEN_STYLE_WARNING,)
    return cleaned, ()


def vox_instruction(text: str, style: str | None) -> tuple[str, str | None, tuple[str, ...]]:
    """Map tags onto one parenthetical instruction and remove them from the script.

    The returned instruction has no parentheses. MLX VoxCPM2 wraps ``instruct``
    itself. The PyTorch package expects the parentheses inside ``text``.
    """
    warnings: list[str] = []
    phrases: list[str] = []

    def add(raw: str, bracketed: bool) -> None:
        key = " ".join(raw.strip().lower().split())
        if not key:
            return
        mapped = STYLE_TO_VOX.get(key)
        if mapped is None and not bracketed and "," in raw:
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            if len(parts) > 1 and all(" ".join(part.lower().split()) in STYLE_TO_VOX for part in parts):
                for part in parts:
                    add(part, False)
                return
        if mapped is None:
            mapped = " ".join(raw.strip().split())
            if bracketed:
                warnings.append(f"No VoxCPM2 mapping for [{raw.strip()}]; sent as ({mapped}).")
        if mapped not in phrases:
            phrases.append(mapped)

    note = (style or "").strip()
    if note.startswith("(") and note.endswith(")") and note.count("(") == 1:
        add(note[1:-1], bracketed=False)
    elif note:
        found = False
        for match in _TAG.finditer(note):
            found = True
            add(match.group(1), bracketed=True)
        leftover = _TAG.sub(" ", note).strip()
        if leftover:
            add(leftover, bracketed=False)
        elif not found:
            add(note, bracketed=False)

    spoken = text
    leading = _LEADING_PAREN.match(spoken)
    if leading:
        add(leading.group(1), bracketed=False)
        spoken = spoken[leading.end() :]

    def replace(match: re.Match[str]) -> str:
        add(match.group(1), bracketed=True)
        return ""

    spoken = _tidy(_TAG.sub(replace, spoken))
    instruct = ", ".join(phrases) if phrases else None
    return spoken, instruct, tuple(warnings)
