from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from voztudio.config import Settings, get_settings
from voztudio.domain import GenerationRequest
from voztudio.engine import FishSpeechEngine, SpeechEngine
from voztudio.generation import GenerationService
from voztudio.storage import VoiceStore
from voztudio.text import validate_language


class GeneratePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    language: str
    voice_id: str
    instructions: str | None = None
    output_format: str = "wav"


def _voice_json(record) -> dict[str, Any]:
    return {
        "id": record.id,
        "transcript": record.transcript,
        "sample_rate": record.sample_rate,
        "created_at": record.created_at.isoformat(),
    }


def create_app(
    settings: Settings | None = None,
    engine: SpeechEngine | None = None,
    data_dir: Path | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    if data_dir is not None:
        settings = replace(settings, data_dir=Path(data_dir).expanduser().resolve())
    voices = VoiceStore(settings.data_dir / "voices")
    output_root = settings.data_dir / "outputs"
    service = GenerationService(
        engine or FishSpeechEngine.from_settings(settings), voices, output_root
    )
    app = FastAPI(title="Voztudio", version="0.1.0")
    app.state.service = service
    app.state.engine = engine or service.engine
    app.state.settings = settings

    @app.get("/health")
    def health():
        ready = True
        detail = "engine injected"
        check = getattr(app.state.engine, "health", None)
        if check is not None:
            ready, detail = check()
        return {
            "status": "ok" if ready else "degraded",
            "engine_ready": ready,
            "detail": detail,
            "device": settings.device,
            "model_id": settings.model_id,
        }

    @app.get("/voices")
    def list_voices():
        return [_voice_json(record) for record in voices.list()]

    @app.post("/voices", status_code=201)
    async def register_voice(
        audio: UploadFile = File(...), transcript: str | None = Form(default=None)
    ):
        suffix = Path(audio.filename or "reference.wav").suffix.lower() or ".wav"
        temporary_path = settings.data_dir / f".upload-{next(tempfile._get_candidate_names())}{suffix}"
        try:
            temporary_path.write_bytes(await audio.read())
            record = voices.register(temporary_path, transcript)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            temporary_path.unlink(missing_ok=True)
        return _voice_json(record)

    @app.post("/generate")
    def generate(payload: GeneratePayload):
        try:
            request = GenerationRequest(**payload.model_dump())
            result = service.generate(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Voice not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return result.to_dict()

    @app.get("/outputs/{output_id}")
    def output(output_id: str):
        if not output_id.isalnum():
            raise HTTPException(status_code=404, detail="Output not found")
        path = output_root / f"{output_id}.wav"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Output not found")
        return FileResponse(path, media_type="audio/wav", filename=path.name)

    return app
