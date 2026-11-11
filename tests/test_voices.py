import wave
from pathlib import Path

import pytest

from voztudio.storage import VoiceStore


def write_wav(path: Path, frames: int = 1600) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * frames)
    return path


def test_register_persists_voice_metadata(tmp_path):
    source = write_wav(tmp_path / "reference.wav")
    root = tmp_path / "data" / "voices"
    record = VoiceStore(root).register(source, transcript="hello")

    assert record.id
    assert record.audio_path.exists()
    assert record.sample_rate == 16000
    assert record.transcript == "hello"
    restored = VoiceStore(root).get(record.id)
    assert restored.audio_path == record.audio_path
    assert VoiceStore(root).list()[0].id == record.id


def test_register_rejects_non_audio_and_empty_file(tmp_path):
    store = VoiceStore(tmp_path / "voices")
    bad = tmp_path / "bad.txt"
    bad.write_text("not audio")
    empty = tmp_path / "empty.wav"
    empty.touch()

    with pytest.raises(ValueError, match="WAV"):
        store.register(bad)
    with pytest.raises(ValueError, match="WAV"):
        store.register(empty)
