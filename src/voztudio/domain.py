from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VoiceRecord:
    id: str
    audio_path: Path
    transcript: str | None
    sample_rate: int
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["audio_path"] = str(self.audio_path)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceRecord":
        return cls(
            id=str(data["id"]),
            audio_path=Path(data["audio_path"]),
            transcript=data.get("transcript"),
            sample_rate=int(data["sample_rate"]),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass(frozen=True)
class GenerationRequest:
    text: str
    language: str
    voice_id: str
    instructions: str | None = None
    output_format: str = "wav"
