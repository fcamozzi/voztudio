from dataclasses import replace

from fastapi.testclient import TestClient

from tests.support import write_reference
from voztudio.api import create_app
from voztudio.config import get_settings
from voztudio.engine import AudioBuffer


def wav_bytes(path) -> bytes:
    return write_reference(path).read_bytes()


class FakeEngine:
    def generate(self, text, language, reference, instructions=None):
        return AudioBuffer(samples=(0.1, 0.2), sample_rate=16000)


def client(tmp_path):
    return TestClient(create_app(data_dir=tmp_path, engine=FakeEngine()))


def _register(api, tmp_path, name="reference.wav", **data):
    payload = {"transcript": "hello there", "consent": "true"}
    payload.update(data)
    return api.post(
        "/voices",
        files={"audio": (name, wav_bytes(tmp_path / name), "audio/wav")},
        data=payload,
    )


def test_health_and_voice_registration(tmp_path):
    api = client(tmp_path)
    health = api.get("/health").json()
    assert health["status"] == "ok"
    assert health["resolved_device"] == "auto"
    response = _register(api, tmp_path)
    assert response.status_code == 201
    body = response.json()
    assert body["sample_rate"] == 16000
    assert body["consent"] is True
    assert body["duration_s"] == 12.0
    assert len(api.get("/voices").json()) == 1


def test_generate_rejects_unknown_language_and_returns_audio(tmp_path):
    api = client(tmp_path)
    voice = _register(api, tmp_path).json()
    bad = api.post(
        "/generate",
        json={"text": "hello", "language": "de", "voice_id": voice["id"]},
    )
    assert bad.status_code == 422
    good = api.post(
        "/generate",
        json={"text": "hello", "language": "en", "voice_id": voice["id"]},
    )
    assert good.status_code == 200
    output_id = good.json()["id"]
    downloaded = api.get(f"/outputs/{output_id}")
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"].startswith("audio/wav")
    deleted = api.delete(f"/outputs/{output_id}")
    assert deleted.status_code == 204
    assert api.get(f"/outputs/{output_id}").status_code == 404


def test_health_reports_engine_failure_and_generation_hides_local_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VOZTUDIO_BACKEND", "qwen3")
    monkeypatch.setenv("VOZTUDIO_RUNTIME", "pytorch")

    class OfflineEngine(FakeEngine):
        def health(self):
            return False, "Fish Speech server unavailable"

    api = TestClient(create_app(data_dir=tmp_path, engine=OfflineEngine()))
    health = api.get("/health")
    assert health.json() == {
        "status": "degraded",
        "engine_ready": False,
        "detail": "Fish Speech server unavailable",
        "device": "auto",
        "resolved_device": "auto",
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "backend": "qwen3",
        "runtime": "pytorch",
        "style_supported": True,
    }
    voice = _register(api, tmp_path).json()
    result = api.post(
        "/generate",
        json={"text": "hello", "language": "en", "voice_id": voice["id"]},
    ).json()
    assert "audio_path" not in result
    assert result["download_url"].startswith("/outputs/")


def test_registration_requires_consent_and_transcript(tmp_path):
    api = client(tmp_path)
    missing = _register(api, tmp_path, name="missing.wav", consent="no")
    assert missing.status_code == 422
    blank = _register(api, tmp_path / "blank", name="blank.wav", transcript="")
    assert blank.status_code == 422


def test_upload_limit_returns_413(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = replace(get_settings(), max_upload_bytes=8)
    api = TestClient(create_app(settings=settings, data_dir=tmp_path / "data", engine=FakeEngine()))
    response = api.post(
        "/voices",
        files={"audio": ("reference.wav", b"0123456789abcdef", "audio/wav")},
        data={"transcript": "hello there", "consent": "true"},
    )
    assert response.status_code == 413


def test_delete_voice(tmp_path):
    api = client(tmp_path)
    voice_id = _register(api, tmp_path).json()["id"]
    assert api.delete(f"/voices/{voice_id}").status_code == 204
    assert api.delete(f"/voices/{voice_id}").status_code == 404
    assert api.get("/voices").json() == []


def test_output_id_rejects_path_traversal(tmp_path):
    api = client(tmp_path)
    assert api.get("/outputs/../secret").status_code == 404
