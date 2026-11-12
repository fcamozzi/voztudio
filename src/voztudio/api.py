from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from voztudio.config import Settings, get_settings
from voztudio.domain import GenerationRequest
from voztudio.engine import FishSpeechEngine, SpeechEngine
from voztudio.generation import GenerationService
from voztudio.storage import VoiceStore

_UPLOAD_SUFFIXES = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
    ".webm",
    ".mp4",
    ".aiff",
    ".aif",
}
_CONSENT_VALUES = {"true", "1", "yes", "on"}


class GeneratePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    language: str
    voice_id: str
    instructions: str | None = None
    output_format: str = "wav"


def consent_granted(value: str | None) -> bool:
    return (value or "").strip().lower() in _CONSENT_VALUES


def _voice_json(record) -> dict[str, Any]:
    return {
        "id": record.id,
        "transcript": record.transcript,
        "sample_rate": record.sample_rate,
        "duration_s": record.duration_s,
        "consent": record.consent,
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
    voices = VoiceStore(
        settings.data_dir / "voices",
        min_seconds=settings.voice_min_seconds,
        max_seconds=settings.voice_max_seconds,
    )
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
            "resolved_device": settings.resolved_device,
            "model_id": settings.model_id,
        }

    @app.get("/voices")
    def list_voices():
        return [_voice_json(record) for record in voices.list()]

    @app.post("/voices", status_code=201)
    async def register_voice(
        audio: UploadFile = File(...),
        transcript: str | None = Form(default=None),
        consent: str | None = Form(default=None),
    ):
        if not consent_granted(consent):
            raise HTTPException(
                status_code=422,
                detail="Confirm that you have the rights to use this reference voice",
            )
        suffix = Path(audio.filename or "reference.wav").suffix.lower()
        if suffix not in _UPLOAD_SUFFIXES:
            suffix = ".wav"
        temporary = tempfile.NamedTemporaryFile(
            prefix="voztudio-upload-", suffix=suffix, dir=settings.data_dir, delete=False
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        try:
            await _copy_upload(audio, temporary_path, settings.max_upload_bytes)
            record = voices.register(temporary_path, transcript, consent=True)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            temporary_path.unlink(missing_ok=True)
        return _voice_json(record)

    @app.delete("/voices/{voice_id}", status_code=204)
    def delete_voice(voice_id: str):
        try:
            voices.delete(voice_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Voice not found") from exc
        return Response(status_code=204)

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
        return {
            "id": result.id,
            "sample_rate": result.sample_rate,
            "duration_s": result.duration_s,
            "language": result.language,
            "voice_id": result.voice_id,
            "download_url": f"/outputs/{result.id}",
        }

    @app.get("/outputs/{output_id}")
    def output(output_id: str):
        path = _output_path(output_root, output_id)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="Output not found")
        return FileResponse(path, media_type="audio/wav", filename=path.name)

    @app.delete("/outputs/{output_id}", status_code=204)
    def delete_output(output_id: str):
        path = _output_path(output_root, output_id)
        if path is None or not path.is_file():
            if path is not None:
                path.with_suffix(".json").unlink(missing_ok=True)
            raise HTTPException(status_code=404, detail="Output not found")
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)
        return Response(status_code=204)

    return app


def _output_path(output_root: Path, output_id: str) -> Path | None:
    if not output_id.isalnum():
        return None
    return output_root / f"{output_id}.wav"


async def _copy_upload(upload: UploadFile, destination: Path, limit: int) -> None:
    written = 0
    with destination.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                raise HTTPException(status_code=413, detail="Upload exceeds the size limit")
            handle.write(chunk)
