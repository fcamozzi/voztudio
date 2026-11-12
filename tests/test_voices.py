import json
import wave
from pathlib import Path

import pytest

from tests.support import write_reference
from voztudio.storage import VoiceStore


def test_register_persists_voice_metadata(tmp_path):
    source = write_reference(tmp_path / "reference.wav")
    root = tmp_path / "data" / "voices"
    record = VoiceStore(root).register(source, transcript="hello there", consent=True)

    assert record.id
    assert record.audio_path.exists()
    assert record.sample_rate == 16000
    assert record.transcript == "hello there"
    assert record.consent is True
    assert record.duration_s == pytest.approx(12.0)
    with wave.open(str(record.audio_path), "rb") as audio:
        assert audio.getnchannels() == 1
    restored = VoiceStore(root).get(record.id)
    assert restored.audio_path == record.audio_path
    assert VoiceStore(root).list()[0].id == record.id


def test_register_rejects_non_audio_and_empty_file(tmp_path):
    store = VoiceStore(tmp_path / "voices")
    bad = tmp_path / "bad.txt"
    bad.write_text("not audio")
    empty = tmp_path / "empty.wav"
    empty.touch()

    with pytest.raises(ValueError, match="readable audio"):
        store.register(bad, transcript="hello there", consent=True)
    with pytest.raises(ValueError, match="readable audio"):
        store.register(empty, transcript="hello there", consent=True)


def test_register_requires_transcript_and_consent(tmp_path):
    store = VoiceStore(tmp_path / "voices")
    source = write_reference(tmp_path / "reference.wav")
    with pytest.raises(ValueError, match="transcript"):
        store.register(source, transcript="  ", consent=True)
    with pytest.raises(ValueError, match="rights"):
        store.register(source, transcript="hello there", consent=False)


def test_register_rejects_silence_short_and_long_audio(tmp_path):
    store = VoiceStore(tmp_path / "voices")
    silent = write_reference(tmp_path / "silent.wav", amplitude=0)
    short = write_reference(tmp_path / "short.wav", seconds=5)
    long = write_reference(tmp_path / "long.wav", seconds=91)
    with pytest.raises(ValueError, match="quiet"):
        store.register(silent, transcript="hello there", consent=True)
    with pytest.raises(ValueError, match="between 10 and 90"):
        store.register(short, transcript="hello there", consent=True)
    with pytest.raises(ValueError, match="between 10 and 90"):
        store.register(long, transcript="hello there", consent=True)


def test_register_accepts_duration_boundaries(tmp_path):
    store = VoiceStore(tmp_path / "voices")
    low = store.register(write_reference(tmp_path / "low.wav", seconds=10), transcript="hello", consent=True)
    high = store.register(write_reference(tmp_path / "high.wav", seconds=90), transcript="hello", consent=True)
    assert low.duration_s == pytest.approx(10.0)
    assert high.duration_s == pytest.approx(90.0)


def test_register_rejects_dc_offset_and_a_single_click(tmp_path):
    store = VoiceStore(tmp_path / "voices")
    dc = write_reference(tmp_path / "dc.wav", constant=0.5)
    click = write_reference(tmp_path / "click.wav", amplitude=0)
    with wave.open(str(click), "rb") as audio:
        frames = audio.readframes(audio.getnframes())
    frames = bytearray(frames)
    frames[0:2] = (32767).to_bytes(2, "little", signed=True)
    click.write_bytes(b"")
    with wave.open(str(click), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(frames)
    with pytest.raises(ValueError, match="quiet"):
        store.register(dc, transcript="hello there", consent=True)
    with pytest.raises(ValueError, match="quiet"):
        store.register(click, transcript="hello there", consent=True)


def test_register_downmixes_stereo_pcm(tmp_path):
    store = VoiceStore(tmp_path / "voices")
    source = write_reference(tmp_path / "stereo.wav", channels=2)
    record = store.register(source, transcript="hello there", consent=True)
    with wave.open(str(record.audio_path), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 16000


def test_old_metadata_without_consent_still_loads(tmp_path):
    root = tmp_path / "voices"
    root.mkdir()
    audio = write_reference(root / "abc.wav", seconds=10)
    payload = {
        "id": "abc",
        "audio_path": str(audio),
        "transcript": "hello there",
        "sample_rate": 16000,
        "created_at": "2026-11-12T00:00:00+00:00",
    }
    (root / "abc.json").write_text(json.dumps(payload), encoding="utf-8")
    record = VoiceStore(root).get("abc")
    assert record.consent is None
    assert record.duration_s is None
    assert VoiceStore(root).list()[0].id == "abc"


def test_delete_removes_voice_files(tmp_path):
    store = VoiceStore(tmp_path / "voices")
    record = store.register(write_reference(tmp_path / "reference.wav"), transcript="hello", consent=True)
    store.delete(record.id)
    with pytest.raises(KeyError):
        store.get(record.id)
    assert list(store.root.glob("*")) == []


def test_ffmpeg_accepts_mp3_when_available(tmp_path):
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    wav = write_reference(tmp_path / "tone.wav")
    mp3 = tmp_path / "tone.mp3"
    completed = subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav), str(mp3)],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        pytest.skip("ffmpeg could not encode mp3")
    record = VoiceStore(tmp_path / "voices").register(mp3, transcript="hello there", consent=True)
    assert record.duration_s == pytest.approx(12.0, abs=0.5)
    with wave.open(str(record.audio_path), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
