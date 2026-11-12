from pathlib import Path

from tests.support import write_reference
from voztudio.engine import AudioBuffer
from voztudio.generation import GenerationService
from voztudio.storage import VoiceStore
from voztudio.ui import UI_NOTICE, create_ui, register_reference, synthesize

ROOT = Path(__file__).parents[1]


class FakeEngine:
    def generate(self, text, language, reference, instructions=None):
        return AudioBuffer(samples=(0.1, 0.2), sample_rate=16000)


def _service(tmp_path):
    voices = VoiceStore(tmp_path / "voices")
    return GenerationService(FakeEngine(), voices, tmp_path / "outputs")


def test_notice_states_license_and_portuguese_limit():
    assert "Built with Fish Audio" in UI_NOTICE
    assert "pt" in UI_NOTICE
    assert "português do Brasil" in UI_NOTICE


def test_ui_registers_only_with_consent_and_reports_errors(tmp_path):
    service = _service(tmp_path)
    audio = write_reference(tmp_path / "reference.wav")
    message, _updated = register_reference(service, str(audio), "hello there", False)
    assert "rights" in message
    assert service.store.list() == []
    message, _updated = register_reference(service, str(audio), "hello there", True)
    assert message.startswith("Registered voice")
    voice_id = service.store.list()[0].id
    path, error = synthesize(service, voice_id, "pt", "Olá.", "calm")
    assert error == ""
    assert path
    _path, error = synthesize(service, voice_id, "de", "Olá.", None)
    assert "Unsupported language" in error
    demo = create_ui(service)
    assert demo.title == "Voztudio"


def test_consent_checkbox_starts_unchecked():
    source = (ROOT / "src" / "voztudio" / "ui.py").read_text(encoding="utf-8")
    assert "value=False" in source
