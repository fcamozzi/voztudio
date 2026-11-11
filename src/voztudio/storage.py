from __future__ import annotations

import json
import uuid
import wave
from pathlib import Path

from voztudio.domain import VoiceRecord


class VoiceStore:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, source: Path, transcript: str | None = None) -> VoiceRecord:
        source = Path(source)
        try:
            with wave.open(str(source), "rb") as audio:
                sample_rate = audio.getframerate()
                frames = audio.getnframes()
                channels = audio.getnchannels()
        except (EOFError, FileNotFoundError, OSError, wave.Error) as exc:
            raise ValueError("Reference audio must be a readable WAV file") from exc
        if frames <= 0 or sample_rate <= 0 or channels <= 0:
            raise ValueError("Reference audio must be a non-empty WAV file")

        voice_id = uuid.uuid4().hex
        destination = self.root / f"{voice_id}.wav"
        metadata_path = self.root / f"{voice_id}.json"
        destination.write_bytes(source.read_bytes())
        from datetime import datetime, timezone

        record = VoiceRecord(
            id=voice_id,
            audio_path=destination,
            transcript=transcript.strip() if transcript and transcript.strip() else None,
            sample_rate=sample_rate,
            created_at=datetime.now(timezone.utc),
        )
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(metadata_path)
        return record

    def get(self, voice_id: str) -> VoiceRecord:
        if not voice_id or Path(voice_id).name != voice_id:
            raise KeyError(voice_id)
        metadata_path = self.root / f"{voice_id}.json"
        if not metadata_path.is_file():
            raise KeyError(voice_id)
        record = VoiceRecord.from_dict(json.loads(metadata_path.read_text(encoding="utf-8")))
        if not record.audio_path.is_file():
            raise KeyError(voice_id)
        return record

    def list(self) -> list[VoiceRecord]:
        records = []
        for metadata_path in sorted(self.root.glob("*.json")):
            try:
                records.append(self.get(metadata_path.stem))
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
        return records
