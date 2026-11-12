from __future__ import annotations

import json
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from voztudio.domain import GenerationRequest, VoiceRecord
from voztudio.engine import AudioBuffer, SpeechEngine
from voztudio.storage import VoiceStore
from voztudio.text import chunk_text, validate_language


@dataclass(frozen=True)
class GeneratedAudio:
    id: str
    audio_path: Path
    sample_rate: int
    duration_s: float
    language: str
    voice_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "audio_path": str(self.audio_path),
            "sample_rate": self.sample_rate,
            "duration_s": self.duration_s,
            "language": self.language,
            "voice_id": self.voice_id,
        }


class GenerationService:
    def __init__(self, engine: SpeechEngine, store: VoiceStore, output_root: Path, max_chars: int = 500):
        self.engine = engine
        self.store = store
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.max_chars = max_chars

    def generate(self, request: GenerationRequest) -> GeneratedAudio:
        language = validate_language(request.language)
        if request.output_format != "wav":
            raise ValueError("Only WAV output is supported")
        reference = self.store.get(request.voice_id)
        buffers = [
            self.engine.generate(chunk, language, reference, request.instructions)
            for chunk in chunk_text(request.text, self.max_chars)
        ]
        if not buffers:
            raise ValueError("Text must not be empty")
        sample_rate = buffers[0].sample_rate
        if any(buffer.sample_rate != sample_rate for buffer in buffers):
            raise RuntimeError("Fish Speech returned inconsistent sample rates")
        samples = tuple(sample for buffer in buffers for sample in buffer.samples)
        audio_id = uuid.uuid4().hex
        audio_path = self.output_root / f"{audio_id}.wav"
        self._write_wav(audio_path, samples, sample_rate)
        result = GeneratedAudio(
            id=audio_id,
            audio_path=audio_path,
            sample_rate=sample_rate,
            duration_s=len(samples) / sample_rate,
            language=language,
            voice_id=reference.id,
        )
        metadata_path = audio_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {
                    **result.to_dict(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result

    @staticmethod
    def _write_wav(path: Path, samples: tuple[float, ...], sample_rate: int) -> None:
        pcm = bytearray()
        for sample in samples:
            bounded = max(-1.0, min(1.0, float(sample)))
            pcm.extend(int(bounded * 32767).to_bytes(2, "little", signed=True))
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(bytes(pcm))
