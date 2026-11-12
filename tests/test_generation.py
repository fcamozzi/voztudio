import wave
from pathlib import Path

from voztudio.domain import GenerationRequest
from voztudio.engine import AudioBuffer
from voztudio.generation import GenerationService
from voztudio.storage import VoiceStore


def write_wav(path: Path) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)
    return path


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate(self, text, language, reference, instructions=None):
        self.calls.append((text, language, reference.id, instructions))
        return AudioBuffer(samples=(0.1, 0.2), sample_rate=16000)


def test_generation_chunks_in_order_and_writes_metadata(tmp_path):
    voices = VoiceStore(tmp_path / "voices")
    reference = voices.register(write_wav(tmp_path / "reference.wav"))
    engine = FakeEngine()
    service = GenerationService(engine, voices, tmp_path / "outputs", max_chars=12)

    result = service.generate(
        GenerationRequest(
            text="First line\nSecond line",
            language="en",
            voice_id=reference.id,
            instructions="calm",
        )
    )

    assert [call[:2] for call in engine.calls] == [
        ("First line", "en"),
        ("Second line", "en"),
    ]
    assert all(call[2:] == (reference.id, "calm") for call in engine.calls)
    assert result.audio_path.exists()
    assert result.sample_rate == 16000
    assert result.duration_s == 4 / 16000
    assert result.audio_path.with_suffix(".json").exists()
    with wave.open(str(result.audio_path), "rb") as handle:
        assert handle.getnframes() == 4


def test_generation_rejects_unknown_language_before_engine_call(tmp_path):
    voices = VoiceStore(tmp_path / "voices")
    reference = voices.register(write_wav(tmp_path / "reference.wav"))
    engine = FakeEngine()
    service = GenerationService(engine, voices, tmp_path / "outputs")

    try:
        service.generate(GenerationRequest("hello", "de", reference.id))
    except ValueError as exc:
        assert "Unsupported language" in str(exc)
    else:
        raise AssertionError("unsupported language was accepted")
    assert engine.calls == []
