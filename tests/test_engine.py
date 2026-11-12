import io
import wave

import httpx
import msgpack

from tests.support import write_reference
from voztudio.api import create_app
from voztudio.engine import FishSpeechEngine, apply_style
from voztudio.storage import VoiceStore
from fastapi.testclient import TestClient


def tiny_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x10" * 16)
    return output.getvalue()


def test_apply_style_wraps_plain_text_and_keeps_complete_tags():
    assert apply_style("Olá.", None) == "Olá."
    assert apply_style("Olá.", "  ") == "Olá."
    assert apply_style("Olá.", "calm") == "[calm] Olá."
    assert apply_style("Olá.", " professional broadcast tone ") == "[professional broadcast tone] Olá."
    assert apply_style("Olá.", "[whisper]") == "[whisper] Olá."
    assert apply_style("Olá.", "[whisper][excited]") == "[whisper][excited] Olá."
    assert apply_style("Olá.", "[whisper] [excited]") == "[whisper] [excited] Olá."


def _engine(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return FishSpeechEngine("http://fish.local", client=client)


def _reference(tmp_path):
    store = VoiceStore(tmp_path / "voices")
    return store.register(write_reference(tmp_path / "ref.wav"), transcript="exact words", consent=True)


def test_generate_sends_msgpack_without_language(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["type"] = request.headers["content-type"]
        seen["body"] = msgpack.unpackb(request.content, raw=False)
        seen["query"] = request.url.params.get("format")
        return httpx.Response(200, content=tiny_wav())

    reference = _reference(tmp_path)
    buffer = _engine(handler).generate("Bom dia.", "pt", reference, instructions="calm")
    body = seen["body"]
    assert seen["type"] == "application/msgpack"
    assert seen["query"] == "msgpack"
    assert body["use_memory_cache"] == "on"
    assert body["format"] == "wav"
    assert body["text"] == "[calm] Bom dia."
    assert "language" not in body
    assert body["references"] == [{"audio": reference.audio_path.read_bytes(), "text": "exact words"}]
    assert buffer.sample_rate == 16000
    assert len(buffer.samples) == 16


def test_model_4xx_is_a_client_error_and_5xx_is_runtime(tmp_path):
    reference = _reference(tmp_path)

    def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="x" * 1000)

    try:
        _engine(reject).generate("Bom dia.", "pt", reference)
    except ValueError as exc:
        assert "rejected the request (400)" in str(exc)
        assert len(str(exc)) < 600
    else:
        raise AssertionError("4xx was not a ValueError")

    def crash(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="busy")

    try:
        _engine(crash).generate("Bom dia.", "pt", reference)
    except RuntimeError as exc:
        assert "503" in str(exc)
    else:
        raise AssertionError("5xx was not a RuntimeError")


def test_api_maps_model_4xx_to_422(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="prompt is too long")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    api = TestClient(
        create_app(
            data_dir=tmp_path,
            engine=FishSpeechEngine("http://fish.local", client=client),
        )
    )
    voice = api.post(
        "/voices",
        files={"audio": ("reference.wav", write_reference(tmp_path / "in.wav").read_bytes(), "audio/wav")},
        data={"transcript": "exact words", "consent": "true"},
    )
    assert voice.status_code == 201
    response = api.post(
        "/generate",
        json={"text": "hello", "language": "en", "voice_id": voice.json()["id"]},
    )
    assert response.status_code == 422
    assert "prompt is too long" in response.json()["detail"]
