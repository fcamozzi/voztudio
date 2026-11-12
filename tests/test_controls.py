from voztudio.asr import TranscriptUnavailable, choose_reference_transcript
from voztudio.domain import GenerationRequest
from voztudio.duration import cap_seconds, qwen_max_tokens, trim_to_duration, vox_max_tokens
from voztudio.engine import AudioBuffer, Synthesis
from voztudio.generation import GenerationService
from voztudio.pronounce import apply_pronunciations, parse_pronunciation_lines
from voztudio.storage import VoiceStore
from tests.support import write_reference


def test_generation_caps_grow_with_text_and_stay_under_the_upstream_defaults():
    short = vox_max_tokens("Os detalhes a gente passa antes da live.")
    long = vox_max_tokens("a " * 100)
    assert short < long < 2000
    assert qwen_max_tokens("Olá.") == 75
    assert qwen_max_tokens("a " * 100) < 2048
    assert cap_seconds("Oi.") < cap_seconds("Os detalhes a gente passa antes da live.")


def test_trim_to_duration_drops_only_the_extra_tail():
    short = trim_to_duration((0.2, 0.2), 16000, 5)
    assert short == ((0.2, 0.2), False)
    loud = tuple(0.4 for _ in range(200))
    trimmed, removed = trim_to_duration(loud, 100, 1.0)
    assert removed is True
    assert len(trimmed) <= 100
    assert len(trimmed) > 0


def test_pronunciations_replace_brand_names_outside_style_tags():
    mapping = parse_pronunciation_lines("FastHunting=Fést Rânting\n# comment\n")
    assert apply_pronunciations("Use FastHunting hoje.", mapping) == "Use Fést Rânting hoje."
    assert apply_pronunciations("Use fasthunting hoje.", mapping) == "Use Fést Rânting hoje."
    assert apply_pronunciations("[FastHunting] hoje", mapping) == "[FastHunting] hoje"


def test_long_reference_transcript_is_replaced_by_the_window(tmp_path):
    class Engine:
        reference_max_seconds = 20.0

        def __init__(self):
            self.cache_dir = tmp_path / "cache"
            self.calls = []

        def synthesize(self, text, reference_audio, reference_transcript, style, language):
            self.calls.append((text, reference_transcript, language, reference_audio))
            return Synthesis(AudioBuffer((0.1, 0.2), 16000), ())

    class ASR:
        def __init__(self):
            self.calls = []

        def transcribe(self, path, language):
            self.calls.append((path, language))
            return "primeiros vinte segundos"

    voices = VoiceStore(tmp_path / "voices")
    record = voices.register(
        write_reference(tmp_path / "long.wav", seconds=40),
        transcript="palavra " * 80,
        consent=True,
    )
    engine = Engine()
    asr = ASR()
    service = GenerationService(engine, voices, tmp_path / "out", transcriber=asr)
    result = service.generate(
        GenerationRequest("FastHunting hoje.", "pt", record.id, pronunciations="FastHunting=Fést Rânting")
    )
    assert engine.calls[0][0] == "Fést Rânting hoje."
    assert engine.calls[0][1] == "primeiros vinte segundos"
    assert engine.calls[0][2] == "pt"
    assert asr.calls and asr.calls[0][1] == "pt"
    assert result.reference_transcript == "primeiros vinte segundos"
    assert any("automatic transcript" in warning for warning in result.warnings)


def test_short_reference_transcript_stays_and_skips_asr(tmp_path):
    class Engine:
        reference_max_seconds = 20.0

        def __init__(self):
            self.cache_dir = tmp_path / "cache"

        def synthesize(self, text, reference_audio, reference_transcript, style, language):
            self.transcript = reference_transcript
            return Synthesis(AudioBuffer((0.1,), 16000), ())

    class ASR:
        def transcribe(self, path, language):
            raise AssertionError("ASR should not run for a short override")

    voices = VoiceStore(tmp_path / "voices")
    record = voices.register(
        write_reference(tmp_path / "ref.wav", seconds=40),
        transcript="exact words",
        consent=True,
    )
    engine = Engine()
    service = GenerationService(engine, voices, tmp_path / "out", transcriber=ASR())
    service.generate(GenerationRequest("Olá.", "pt", record.id))
    assert engine.transcript == "exact words"


def test_missing_asr_rejects_a_transcript_that_overruns_the_window():
    def unavailable():
        raise TranscriptUnavailable("no model")

    try:
        choose_reference_transcript("palavra " * 80, 20, unavailable)
    except ValueError as exc:
        assert "first 20 seconds" in str(exc)
    else:
        raise AssertionError("a long transcript was sent without ASR")
