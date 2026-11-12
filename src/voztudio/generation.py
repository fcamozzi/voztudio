from __future__ import annotations

import json
import math
import uuid
import wave
from dataclasses import dataclass, replace as replace_record
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from voztudio.asr import choose_reference_transcript
from voztudio.domain import GenerationRequest
from voztudio.duration import cap_seconds, trim_to_duration
from voztudio.engine import AudioBuffer, SpeechEngine
from voztudio.pronounce import apply_pronunciations
from voztudio.reference import prepare_reference
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
    warnings: tuple[str, ...] = ()
    reference_transcript: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "audio_path": str(self.audio_path),
            "sample_rate": self.sample_rate,
            "duration_s": self.duration_s,
            "language": self.language,
            "voice_id": self.voice_id,
            "warnings": list(self.warnings),
            "reference_transcript": self.reference_transcript,
        }


class GenerationService:
    def __init__(
        self,
        engine: SpeechEngine,
        store: VoiceStore,
        output_root: Path,
        max_chars: int = 200,
        fade_s: float = 0.02,
        transcriber=None,
        pronunciations: dict[str, str] | None = None,
    ):
        self.engine = engine
        self.store = store
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.max_chars = max_chars
        self.fade_s = fade_s
        self.transcriber = transcriber
        self.pronunciations = dict(pronunciations or {})

    def generate(self, request: GenerationRequest) -> GeneratedAudio:
        language = validate_language(request.language)
        if request.output_format != "wav":
            raise ValueError("Only WAV output is supported")
        script = apply_pronunciations(request.text, self._pronunciations(request.pronunciations))
        reference = self.store.get(request.voice_id)
        reference, transcript_warnings = self._prepare_reference(reference, language)
        warnings: list[str] = list(transcript_warnings)
        buffers: list[AudioBuffer] = []
        for chunk in chunk_text(script, self.max_chars):
            buffer, chunk_warnings = self._speak(chunk, language, reference, request.instructions)
            capped, removed = trim_to_duration(buffer.samples, buffer.sample_rate, cap_seconds(chunk))
            if removed:
                buffer = AudioBuffer(capped, buffer.sample_rate)
                chunk_warnings = tuple(chunk_warnings) + (
                    "The clip ran past the length of this sentence, so the extra tail was removed.",
                )
            buffers.append(buffer)
            for warning in chunk_warnings:
                if warning not in warnings:
                    warnings.append(warning)
        if not buffers:
            raise ValueError("Text must not be empty")
        sample_rate = buffers[0].sample_rate
        if any(buffer.sample_rate != sample_rate for buffer in buffers):
            raise RuntimeError("The speech backend returned inconsistent sample rates")
        samples = crossfade_audio([buffer.samples for buffer in buffers], sample_rate, self.fade_s)
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
            warnings=tuple(warnings),
            reference_transcript=reference.transcript,
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

    def _pronunciations(self, extra: str | None) -> dict[str, str]:
        mapping = dict(self.pronunciations)
        from voztudio.pronounce import parse_pronunciation_lines

        mapping.update(parse_pronunciation_lines(extra))
        return mapping

    def _prepare_reference(self, reference, language: str):
        max_seconds = getattr(self.engine, "reference_max_seconds", None)
        if not max_seconds:
            if not reference.transcript:
                raise ValueError(
                    "A transcript of the reference audio is required; "
                    "register the voice with its exact transcript"
                )
            return reference, ()
        cache_dir = getattr(self.engine, "cache_dir", None)
        window, _trimmed = prepare_reference(reference.audio_path, cache_dir, float(max_seconds))

        def transcribe() -> str:
            if self.transcriber is None:
                from voztudio.asr import TranscriptUnavailable

                raise TranscriptUnavailable("No transcriber is configured.")
            return self.transcriber.transcribe(window, language)

        text, _source, warnings = choose_reference_transcript(
            reference.transcript, float(max_seconds), transcribe
        )
        return replace_record(reference, audio_path=window, transcript=text), warnings

    def _speak(self, chunk: str, language: str, reference, instructions: str | None):
        synthesize = getattr(self.engine, "synthesize", None)
        if callable(synthesize):
            result = synthesize(
                text=chunk,
                reference_audio=reference.audio_path,
                reference_transcript=reference.transcript,
                style=instructions,
                language=language,
            )
            return result.audio, result.warnings
        return (
            self.engine.generate(chunk, language, reference, instructions),
            tuple(getattr(self.engine, "last_warnings", ())),
        )

    @staticmethod
    def _write_wav(path: Path, samples: Sequence[float], sample_rate: int) -> None:
        pcm = bytearray()
        for sample in samples:
            bounded = max(-1.0, min(1.0, float(sample)))
            pcm.extend(int(bounded * 32767).to_bytes(2, "little", signed=True))
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(bytes(pcm))


def crossfade_audio(
    chunks: Sequence[Sequence[float]],
    sample_rate: int,
    fade_s: float = 0.02,
) -> tuple[float, ...]:
    """Equal-power join. Buffers shorter than the fade are concatenated."""
    if not chunks:
        return ()
    merged = tuple(float(sample) for sample in chunks[0])
    fade = int(round(fade_s * sample_rate))
    for raw in chunks[1:]:
        nxt = tuple(float(sample) for sample in raw)
        if fade <= 1 or len(merged) < fade or len(nxt) < fade:
            merged = merged + nxt
            continue
        last = fade - 1
        overlap = tuple(
            merged[-fade + index] * math.cos((math.pi / 2) * (index / last))
            + nxt[index] * math.sin((math.pi / 2) * (index / last))
            for index in range(fade)
        )
        merged = merged[:-fade] + overlap + nxt[fade:]
    return merged
