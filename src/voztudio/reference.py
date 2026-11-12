"""Trim a stored reference to the clip synthesis actually sends."""

from __future__ import annotations

import wave
from pathlib import Path


def prepare_reference(
    source: Path,
    cache_dir: Path | None,
    max_seconds: float,
) -> tuple[Path, bool]:
    """Return a WAV no longer than ``max_seconds``.

    The original upload is kept. A clipped copy is reused until the source
    file changes. Non-WAV input is returned unchanged so the caller still
    has a path; voice storage writes PCM WAV before this runs.
    """
    if max_seconds <= 0:
        raise ValueError("Reference clip length must be positive")
    source = Path(source)
    try:
        with wave.open(str(source), "rb") as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
            if rate <= 0 or frames <= 0:
                return source, False
            limit = int(max_seconds * rate)
            if frames <= limit:
                return source, False
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            raw = handle.readframes(limit)
    except (EOFError, OSError, wave.Error):
        return source, False

    root = Path(cache_dir) if cache_dir is not None else source.parent / ".reference-clips"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{source.stem}-{limit}f.wav"
    if not destination.is_file() or destination.stat().st_mtime < source.stat().st_mtime:
        with wave.open(str(destination), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(width)
            handle.setframerate(rate)
            handle.writeframes(raw)
    return destination, True


def clip_warning(backend: str, seconds: float) -> str:
    return (
        f"Reference audio is longer than {seconds:g} seconds. "
        f"The {backend} backend receives the first {seconds:g} seconds. "
        "The stored transcript still describes the full upload."
    )
