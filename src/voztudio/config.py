from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

TARGET_LANGUAGES = ("en", "pt", "es", "hi", "zh", "fr")


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    cache_dir: Path
    fish_speech_root: Path
    model_id: str
    device: str
    host: str
    port: int
    languages: tuple[str, ...] = TARGET_LANGUAGES


def _path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


def get_settings() -> Settings:
    return Settings(
        data_dir=_path("VOZTUDIO_DATA_DIR", "data"),
        cache_dir=_path("VOZTUDIO_CACHE_DIR", ".cache"),
        fish_speech_root=_path("FISH_SPEECH_ROOT", ".cache/fish-speech"),
        model_id=os.getenv("FISH_MODEL_ID", "fishaudio/s2-pro"),
        device=os.getenv("VOZTUDIO_DEVICE", "auto"),
        host=os.getenv("VOZTUDIO_HOST", "127.0.0.1"),
        port=int(os.getenv("VOZTUDIO_PORT", "7860")),
    )
