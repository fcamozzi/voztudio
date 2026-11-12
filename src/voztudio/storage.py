from __future__ import annotations

import array
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import wave
from pathlib import Path

from voztudio.domain import VoiceRecord

_SILENCE_RMS = 0.005


class VoiceStore:
    def __init__(self, root: Path, min_seconds: float = 10.0, max_seconds: float = 90.0):
        if min_seconds <= 0 or max_seconds < min_seconds:
            raise ValueError("Voice duration window is invalid")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

    def register(self, source: Path, transcript: str | None = None, consent: bool = False) -> VoiceRecord:
        if not consent:
            raise ValueError("Confirm that you have the rights to use this reference voice")
        cleaned = transcript.strip() if transcript else ""
        if not cleaned:
            raise ValueError("A transcript of the reference audio is required")
        samples, sample_rate, duration_s, rms = self._decode(Path(source))
        slack = 1.0 / sample_rate
        if duration_s < self.min_seconds - slack or duration_s > self.max_seconds + slack:
            raise ValueError(
                f"Reference audio must be between {self.min_seconds:g} and {self.max_seconds:g} seconds "
                f"(this file is {duration_s:.2f}s). 10–30 seconds is the recommended range."
            )
        if rms < _SILENCE_RMS:
            raise ValueError("Reference audio is silent or too quiet")

        voice_id = uuid.uuid4().hex
        destination = self.root / f"{voice_id}.wav"
        metadata_path = self.root / f"{voice_id}.json"
        self._write_pcm(destination, samples, sample_rate)
        from datetime import datetime, timezone

        record = VoiceRecord(
            id=voice_id,
            audio_path=destination,
            transcript=cleaned,
            sample_rate=sample_rate,
            created_at=datetime.now(timezone.utc),
            consent=True,
            duration_s=duration_s,
        )
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(metadata_path)
        return record

    def get(self, voice_id: str) -> VoiceRecord:
        if not self._safe_id(voice_id):
            raise KeyError(voice_id)
        metadata_path = self.root / f"{voice_id}.json"
        if not metadata_path.is_file():
            raise KeyError(voice_id)
        record = VoiceRecord.from_dict(json.loads(metadata_path.read_text(encoding="utf-8")))
        if not record.audio_path.is_file():
            raise KeyError(voice_id)
        return record

    def delete(self, voice_id: str) -> None:
        if not self._safe_id(voice_id):
            raise KeyError(voice_id)
        metadata_path = self.root / f"{voice_id}.json"
        if not metadata_path.is_file():
            self._unlink_if_inside(self.root / f"{voice_id}.wav")
            raise KeyError(voice_id)
        record = VoiceRecord.from_dict(json.loads(metadata_path.read_text(encoding="utf-8")))
        self._unlink_if_inside(record.audio_path)
        self._unlink_if_inside(self.root / f"{voice_id}.wav")
        metadata_path.unlink(missing_ok=True)

    def list(self) -> list[VoiceRecord]:
        records = []
        for metadata_path in sorted(self.root.glob("*.json")):
            try:
                records.append(self.get(metadata_path.stem))
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
        return records

    def _decode(self, source: Path) -> tuple[array.array, int, float, float]:
        decoded = self._decode_wave(source)
        if decoded is not None:
            return decoded
        converted = self._ffmpeg(source)
        try:
            decoded = self._decode_wave(converted)
        finally:
            converted.unlink(missing_ok=True)
        if decoded is None:
            raise ValueError("Reference audio must be a readable audio file")
        return decoded

    def _decode_wave(self, source: Path) -> tuple[array.array, int, float, float] | None:
        try:
            with wave.open(str(source), "rb") as audio:
                if audio.getsampwidth() != 2 or audio.getcomptype() != "NONE" or audio.getnchannels() < 1:
                    return None
                channels = audio.getnchannels()
                sample_rate = audio.getframerate()
                frames = audio.getnframes()
                if frames <= 0 or sample_rate <= 0:
                    return None
                raw = audio.readframes(frames)
        except (EOFError, FileNotFoundError, OSError, wave.Error):
            return None
        samples = array.array("h")
        samples.frombytes(raw)
        if sys.byteorder != "little":
            samples.byteswap()
        expected = frames * channels
        if len(samples) < expected:
            return None
        samples = samples[:expected]
        duration_s = frames / sample_rate
        rms = _dc_removed_rms(samples)
        mono = samples if channels == 1 else _downmix(samples, channels)
        return mono, sample_rate, duration_s, rms

    def _ffmpeg(self, source: Path) -> Path:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise ValueError("FFmpeg is required to read this audio format")
        descriptor, name = tempfile.mkstemp(prefix="voztudio-", suffix=".wav")
        os.close(descriptor)
        temporary = Path(name)
        completed = subprocess.run(
            [ffmpeg, "-y", "-i", str(source), "-ac", "1", "-c:a", "pcm_s16le", str(temporary)],
            check=False,
            capture_output=True,
            timeout=60,
        )
        if completed.returncode != 0 or not temporary.is_file():
            temporary.unlink(missing_ok=True)
            raise ValueError("Reference audio must be a readable audio file")
        return temporary

    def _write_pcm(self, path: Path, samples: array.array, sample_rate: int) -> None:
        payload = array.array("h", samples)
        if sys.byteorder != "little":
            payload.byteswap()
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(payload.tobytes())

    def _unlink_if_inside(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            root = self.root.resolve()
        except OSError:
            return
        if resolved == root or root not in resolved.parents:
            return
        resolved.unlink(missing_ok=True)

    @staticmethod
    def _safe_id(voice_id: str) -> bool:
        return bool(voice_id) and Path(voice_id).name == voice_id


def _downmix(samples: array.array, channels: int) -> array.array:
    mono = array.array("h")
    for index in range(0, len(samples), channels):
        total = 0
        for channel in range(channels):
            total += samples[index + channel]
        mono.append(int(round(total / channels)))
    return mono


def _dc_removed_rms(samples: array.array) -> float:
    count = len(samples)
    if count == 0:
        return 0.0
    mean = sum(samples) / count / 32768.0
    total = 0.0
    for sample in samples:
        delta = sample / 32768.0 - mean
        total += delta * delta
    return math.sqrt(total / count)
