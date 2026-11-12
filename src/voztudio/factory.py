"""Pick the speech backend named by settings."""

from __future__ import annotations

from voztudio.config import Settings
from voztudio.engine import FishSpeechEngine


def build_engine(settings: Settings):
    if settings.backend == "fish":
        return FishSpeechEngine.from_settings(settings)
    if settings.backend in {"qwen3", "qwen3-fast"}:
        from voztudio.qwen import Qwen3Backend

        return Qwen3Backend.from_settings(settings)
    if settings.backend == "voxcpm2":
        from voztudio.voxcpm import VoxCPM2Backend

        return VoxCPM2Backend.from_settings(settings)
    raise SystemExit(
        f"Unknown VOZTUDIO_BACKEND={settings.backend}. "
        "Expected one of: qwen3, qwen3-fast, voxcpm2, fish."
    )
