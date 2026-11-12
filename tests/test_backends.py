import io
import wave

import httpx
import msgpack

from tests.support import write_reference
from voztudio.config import get_settings
from voztudio.domain import VoiceRecord
from voztudio.engine import FishSpeechEngine
from voztudio.factory import build_engine
from voztudio.qwen import Qwen3Backend
from voztudio.reference import prepare_reference
from voztudio.stylemap import HINDI_WARNING, QWEN_STYLE_WARNING, strip_qwen_style, vox_instruction
from voztudio.voxcpm import VoxCPM2Backend
from datetime import datetime, timezone


class _Chunk:
    def __init__(self, audio, rate):
        self.audio = audio
        self.sample_rate = rate


def test_qwen_style_is_removed_and_vox_maps_whisper():
    spoken, warnings = strip_qwen_style("Olá [whisper].", "calm")
    assert spoken == "Olá."
    assert warnings == (QWEN_STYLE_WARNING,)
    assert strip_qwen_style("Olá.", None) == ("Olá.", ())

    text, instruct, mapped_warnings = vox_instruction("Olá [whisper].", None)
    assert text == "Olá."
    assert instruct == "sussurrando"
    assert mapped_warnings == ()
    text, instruct, mapped_warnings = vox_instruction("Olá.", "calm, slow")
    assert text == "Olá."
    assert instruct == "tom calmo, mais devagar"
    text, instruct, mapped_warnings = vox_instruction("Olá [robot].", None)
    assert instruct == "robot"
    assert mapped_warnings == ("No VoxCPM2 mapping for [robot]; sent as (robot).",)


def test_prepare_reference_keeps_short_clips_and_trims_long_ones(tmp_path):
    short = write_reference(tmp_path / "short.wav", seconds=12)
    path, trimmed = prepare_reference(short, tmp_path / "cache", 15)
    assert trimmed is False
    assert path == short
    long = write_reference(tmp_path / "long.wav", seconds=40)
    clipped, trimmed = prepare_reference(long, tmp_path / "cache", 15)
    assert trimmed is True
    with wave.open(str(clipped), "rb") as handle:
        assert handle.getnframes() == 15 * 16000
    again, _trimmed = prepare_reference(long, tmp_path / "cache", 15)
    assert again == clipped


def test_qwen_mlx_clone_omits_voice_and_style(tmp_path):
    class Model:
        def generate(self, **kwargs):
            self.kwargs = kwargs
            return iter([_Chunk([0.1, -0.2], 24000)])

    model = Model()
    backend = Qwen3Backend(
        "qwen3",
        "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "mlx",
        tmp_path / "cache",
        model=model,
    )
    audio = write_reference(tmp_path / "ref.wav")
    result = backend.synthesize("Olá [whisper].", audio, "exact words", "calm", "pt")
    assert result.audio.sample_rate == 24000
    assert result.audio.samples == (0.1, -0.2)
    assert QWEN_STYLE_WARNING in result.warnings
    assert model.kwargs == {
        "text": "Olá.",
        "ref_audio": str(audio),
        "ref_text": "exact words",
        "lang_code": "portuguese",
        "stream": False,
    }


def test_qwen_pytorch_reuses_the_clone_prompt_and_warns_for_hindi(tmp_path):
    class Model:
        def __init__(self):
            self.prompts = []
            self.calls = []

        def create_voice_clone_prompt(self, ref_audio, ref_text):
            self.prompts.append((ref_audio, ref_text))
            return {"ref": ref_text}

        def generate_voice_clone(self, **kwargs):
            self.calls.append(kwargs)
            return ([0.5, 0.25], 24000)

    model = Model()
    backend = Qwen3Backend(
        "qwen3-fast",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "pytorch",
        tmp_path / "cache",
        model=model,
    )
    audio = write_reference(tmp_path / "ref.wav", seconds=40)
    first = backend.synthesize("Hello.", audio, "exact words", None, "hi")
    second = backend.synthesize("Again.", audio, "exact words", None, "en")
    assert first.audio.sample_rate == 24000
    assert HINDI_WARNING in first.warnings
    assert any("15 seconds" in warning for warning in first.warnings)
    assert HINDI_WARNING not in second.warnings
    assert len(model.prompts) == 1
    assert model.calls[0]["language"] == "auto"
    assert model.calls[0]["voice_clone_prompt"] == {"ref": "exact words"}
    assert model.calls[1]["language"] == "english"
    assert "ref_audio" not in model.calls[0]


def test_voxcpm_mlx_and_pytorch_style_calls(tmp_path):
    class MlxModel:
        def generate(self, **kwargs):
            self.kwargs = kwargs
            return iter([_Chunk([0.2, 0.3], 48000)])

    mlx_model = MlxModel()
    backend = VoxCPM2Backend(
        "mlx-community/VoxCPM2-8bit",
        "mlx",
        tmp_path / "cache",
        model=mlx_model,
    )
    audio = write_reference(tmp_path / "ref.wav")
    result = backend.synthesize("Olá [whisper].", audio, "exact words", None, "pt")
    assert result.audio.sample_rate == 48000
    assert mlx_model.kwargs["text"] == "Olá."
    assert mlx_model.kwargs["instruct"] == "sussurrando"
    assert mlx_model.kwargs["ref_audio"] == str(audio)

    class Inner:
        sample_rate = 48000

    class TorchModel:
        def __init__(self):
            self.tts_model = Inner()

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return [0.4, -0.4]

    torch_model = TorchModel()
    torch_backend = VoxCPM2Backend(
        "openbmb/VoxCPM2",
        "pytorch",
        tmp_path / "cache",
        model=torch_model,
    )
    torch_backend.synthesize("Olá [whisper].", audio, None, None, "pt")
    assert torch_model.kwargs["text"] == "(sussurrando)Olá."
    assert torch_model.kwargs["reference_wav_path"] == str(audio)
    assert torch_model.kwargs["cfg_value"] == 2.0
    assert torch_model.kwargs["inference_timesteps"] == 10


def test_fish_sends_only_the_first_30_seconds(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = msgpack.unpackb(request.content, raw=False)
        output = io.BytesIO()
        with wave.open(output, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x10" * 8)
        return httpx.Response(200, content=output.getvalue())

    source = write_reference(tmp_path / "long.wav", seconds=40)
    record = VoiceRecord(
        id="abc",
        audio_path=source,
        transcript="exact words",
        sample_rate=16000,
        created_at=datetime.now(timezone.utc),
        consent=True,
        duration_s=40,
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    engine = FishSpeechEngine("http://fish.local", client=client, cache_dir=tmp_path / "cache")
    buffer = engine.generate("Olá.", "pt", record, instructions="calm")
    sent = seen["body"]["references"][0]["audio"]
    assert sent != source.read_bytes()
    with wave.open(io.BytesIO(sent), "rb") as handle:
        assert handle.getnframes() == 30 * 16000
    assert seen["body"]["text"] == "[calm] Olá."
    assert seen["body"]["use_memory_cache"] == "on"
    assert buffer.sample_rate == 16000
    assert any("30 seconds" in warning for warning in engine.last_warnings)


def test_factory_builds_the_selected_backend(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VOZTUDIO_BACKEND", "qwen3-fast")
    monkeypatch.setenv("VOZTUDIO_RUNTIME", "pytorch")
    engine = build_engine(get_settings())
    assert engine.name == "qwen3-fast"
    assert engine.supports_style is False
    assert engine.reference_max_seconds == 15
    monkeypatch.setenv("VOZTUDIO_BACKEND", "fish")
    monkeypatch.delenv("FISH_MODEL_ID", raising=False)
    monkeypatch.delenv("VOZTUDIO_MODEL_ID", raising=False)
    fish = build_engine(get_settings())
    assert isinstance(fish, FishSpeechEngine)
    assert fish.supports_style is True
    assert fish.model_id == "fishaudio/s2-pro"
