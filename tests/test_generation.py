import wave
from pathlib import Path

import pytest

from tests.support import write_reference
from voztudio.domain import GenerationRequest
from voztudio.engine import AudioBuffer
from voztudio.generation import GenerationService, crossfade_audio
from voztudio.storage import VoiceStore


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate(self, text, language, reference, instructions=None):
        self.calls.append((text, language, reference.id, instructions))
        return AudioBuffer(samples=(0.1, 0.2), sample_rate=16000)


def _service(tmp_path: Path, max_chars: int = 200):
    voices = VoiceStore(tmp_path / "voices")
    reference = voices.register(
        write_reference(tmp_path / "reference.wav"),
        transcript="hello there",
        consent=True,
    )
    engine = FakeEngine()
    return GenerationService(engine, voices, tmp_path / "outputs", max_chars=max_chars), engine, reference


def test_generation_chunks_in_order_and_writes_metadata(tmp_path):
    service, engine, reference = _service(tmp_path, max_chars=12)

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
    service, engine, reference = _service(tmp_path)

    try:
        service.generate(GenerationRequest("hello", "de", reference.id))
    except ValueError as exc:
        assert "Unsupported language" in str(exc)
    else:
        raise AssertionError("unsupported language was accepted")
    assert engine.calls == []


def test_default_max_chars_matches_one_s2_pass():
    assert GenerationService.__init__.__defaults__[0] == 200


def test_crossfade_mixes_overlap_and_leaves_tiny_buffers_alone():
    mixed = crossfade_audio([(1, 1, 1, 1), (0.5, 0.5, 0.5, 0.5)], sample_rate=100)
    assert mixed == pytest.approx((1, 1, 1.0, 0.5, 0.5, 0.5))
    assert len(mixed) == 6
    tiny = crossfade_audio([(0.1, 0.2), (0.1, 0.2)], sample_rate=16000)
    assert tiny == (0.1, 0.2, 0.1, 0.2)
