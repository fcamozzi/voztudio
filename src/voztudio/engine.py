from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from voztudio.config import Settings
from voztudio.domain import VoiceRecord


@dataclass(frozen=True)
class AudioBuffer:
    samples: Sequence[float]
    sample_rate: int


class SpeechEngine(Protocol):
    def generate(
        self,
        text: str,
        language: str,
        reference: VoiceRecord,
        instructions: str | None = None,
    ) -> AudioBuffer: ...


class FishSpeechEngine:
    """HTTP adapter for the official Fish Speech API server."""

    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    @classmethod
    def from_settings(cls, settings: Settings) -> "FishSpeechEngine":
        import os

        return cls(settings.fish_speech_url, os.getenv("FISH_SPEECH_API_KEY"))

    def generate(
        self,
        text: str,
        language: str,
        reference: VoiceRecord,
        instructions: str | None = None,
    ) -> AudioBuffer:
        if not reference.transcript:
            raise ValueError(
                "Fish Speech requires a transcript for the reference audio; "
                "register the voice with its exact transcript"
            )
        try:
            import httpx
            import msgpack
        except ImportError as exc:
            raise RuntimeError(
                "Fish Speech adapter requires httpx and msgpack; run scripts/install.sh"
            ) from exc

        prompt = text if not instructions else f"{instructions.strip()}\n{text}"
        payload = {
            "text": prompt,
            "references": [
                {
                    "audio": Path(reference.audio_path).read_bytes(),
                    "text": reference.transcript,
                }
            ],
            "format": "wav",
        }
        headers = {"content-type": "application/msgpack"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        try:
            response = httpx.post(
                f"{self.base_url}/v1/tts?format=msgpack",
                content=msgpack.packb(payload, use_bin_type=True),
                headers=headers,
                timeout=None,
            )
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Fish Speech request failed: {exc}") from exc
        return self._decode_wav(response.content)

    @staticmethod
    def _decode_wav(content: bytes) -> AudioBuffer:
        try:
            with wave.open(io.BytesIO(content), "rb") as handle:
                if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                    raise ValueError("Fish Speech must return mono 16-bit WAV")
                raw = handle.readframes(handle.getnframes())
                samples = tuple(
                    int.from_bytes(raw[index : index + 2], "little", signed=True) / 32768.0
                    for index in range(0, len(raw), 2)
                )
                return AudioBuffer(samples=samples, sample_rate=handle.getframerate())
        except (EOFError, OSError, wave.Error, ValueError) as exc:
            raise RuntimeError("Fish Speech returned invalid WAV audio") from exc
