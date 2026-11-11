import io
import wave

from fastapi.testclient import TestClient

from voztudio.api import create_app
from voztudio.engine import AudioBuffer


def wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)
    return output.getvalue()


class FakeEngine:
    def generate(self, text, language, reference, instructions=None):
        return AudioBuffer(samples=(0.1, 0.2), sample_rate=16000)


def client(tmp_path):
    return TestClient(create_app(data_dir=tmp_path, engine=FakeEngine()))


def test_health_and_voice_registration(tmp_path):
    api = client(tmp_path)
    assert api.get("/health").json()["status"] == "ok"
    response = api.post(
        "/voices",
        files={"audio": ("reference.wav", wav_bytes(), "audio/wav")},
        data={"transcript": "hello"},
    )
    assert response.status_code == 201
    assert response.json()["sample_rate"] == 16000
    assert len(api.get("/voices").json()) == 1


def test_generate_rejects_unknown_language_and_returns_audio(tmp_path):
    api = client(tmp_path)
    voice = api.post(
        "/voices",
        files={"audio": ("reference.wav", wav_bytes(), "audio/wav")},
        data={"transcript": "hello"},
    ).json()
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


def test_health_reports_engine_failure_and_generation_hides_local_path(tmp_path):
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
        "model_id": "fishaudio/s2-pro",
    }
    voice = api.post(
        "/voices",
        files={"audio": ("reference.wav", wav_bytes(), "audio/wav")},
        data={"transcript": "hello"},
    ).json()
    result = api.post(
        "/generate",
        json={"text": "hello", "language": "en", "voice_id": voice["id"]},
    ).json()
    assert "audio_path" not in result
    assert result["download_url"].startswith("/outputs/")
