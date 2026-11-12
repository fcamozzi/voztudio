from __future__ import annotations

import io
import re
import wave
from dataclasses import dataclass
from typing import Protocol, Sequence

import httpx
import msgpack

from voztudio.config import Settings
from voztudio.domain import VoiceRecord

_COMPLETE_TAGS = re.compile(r"^(?:\[[^\[\]\r\n]+\]\s*)+$")
_ERROR_BODY_LIMIT = 400


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


def apply_style(text: str, instructions: str | None) -> str:
    """Turn a style note into one Fish Speech inline tag.

    A value that is already one or more complete [tags] is kept. Any other
    note is wrapped as a single [tag]. The language argument is not part of
    this string: this Fish Speech build has no language field.
    """
    if instructions is None:
        return text
    stripped = instructions.strip()
    if not stripped:
        return text
    if _COMPLETE_TAGS.fullmatch(stripped):
        return f"{stripped} {text}"
    return f"[{stripped}] {text}"


class FishSpeechEngine:
    """HTTP adapter for the official Fish Speech API server."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        timeout: httpx.Timeout | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = client
        self.timeout = timeout or httpx.Timeout(900.0, connect=10.0)

    @classmethod
    def from_settings(cls, settings: Settings) -> "FishSpeechEngine":
        return cls(
            settings.fish_speech_url,
            settings.fish_api_key,
            timeout=httpx.Timeout(settings.fish_timeout_s, connect=10.0),
        )

    def generate(
        self,
        text: str,
        language: str,
        reference: VoiceRecord,
        instructions: str | None = None,
    ) -> AudioBuffer:
        del language  # Local label only. ServeTTSRequest has no language field.
        if not reference.transcript:
            raise ValueError(
                "Fish Speech requires a transcript for the reference audio; "
                "register the voice with its exact transcript"
            )
        payload = {
            "text": apply_style(text, instructions),
            "references": [
                {
                    "audio": reference.audio_path.read_bytes(),
                    "text": reference.transcript,
                }
            ],
            "format": "wav",
            "use_memory_cache": "on",
        }
        headers = {"content-type": "application/msgpack"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        try:
            response = self._http().post(
                f"{self.base_url}/v1/tts?format=msgpack",
                content=msgpack.packb(payload, use_bin_type=True),
                headers=headers,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = exc.response.text[:_ERROR_BODY_LIMIT]
            if 400 <= status < 500:
                raise ValueError(f"Fish Speech rejected the request ({status}): {body}") from exc
            raise RuntimeError(f"Fish Speech request failed ({status}): {body}") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Fish Speech request failed: {exc}") from exc
        return self._decode_wav(response.content)

    def health(self) -> tuple[bool, str]:
        try:
            response = self._http().get(f"{self.base_url}/v1/health", timeout=5.0)
            if response.is_success and response.json().get("status") == "ok":
                return True, "Fish Speech server ready"
            return False, f"Fish Speech server returned HTTP {response.status_code}"
        except Exception as exc:
            return False, f"Fish Speech server unavailable: {exc}"

    def _http(self) -> httpx.Client:
        if self.client is None:
            self.client = httpx.Client(timeout=self.timeout)
        return self.client

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
