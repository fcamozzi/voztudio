"""Transcribe the reference window that synthesis actually sends.

Apple Silicon uses mlx-audio's STT loader (``mlx_audio.stt.load`` then
``model.generate(path, language=...)``). Whisper checkpoints take an ISO
code such as ``pt``. Qwen3-ASR takes a language name such as ``Portuguese``.
Other machines use faster-whisper's ``WhisperModel.transcribe``.
"""

from __future__ import annotations

from pathlib import Path

from voztudio.models import MLX_ASR_MODEL

_QWEN_ASR_LANGUAGE = {
    "en": "English",
    "pt": "Portuguese",
    "es": "Spanish",
    "hi": "Hindi",
    "zh": "Chinese",
    "fr": "French",
}

# A careful transcript of the kept window stays under this many characters
# per second. Anything longer still describes audio that was cut off.
_CHARS_PER_WINDOW_SECOND = 22


class TranscriptUnavailable(RuntimeError):
    pass


def window_limit(seconds: float) -> int:
    return max(1, int(seconds * _CHARS_PER_WINDOW_SECOND))


def choose_reference_transcript(
    user_text: str | None,
    window_seconds: float,
    transcribe,
) -> tuple[str | None, str, tuple[str, ...]]:
    """Pick the transcript that matches the trimmed reference.

    A short user transcript is an override of that window. A transcript that
    is too long for the window is kept on the voice record but is not sent;
    the automatic transcript of the window is sent instead.
    """
    user = (user_text or "").strip()
    limit = window_limit(window_seconds)
    if user and len(user) <= limit:
        return user, "override", ()
    try:
        auto = (transcribe() or "").strip()
    except TranscriptUnavailable as exc:
        auto = ""
        unavailable = str(exc)
    else:
        unavailable = ""
    if auto:
        warnings = []
        if user:
            warnings.append(
                f"The pasted transcript is longer than the first {window_seconds:g} seconds, "
                "so the model receives an automatic transcript of that window."
            )
        return auto, "auto", tuple(warnings)
    if user:
        raise ValueError(
            f"The transcript has to cover only the first {window_seconds:g} seconds "
            f"of the reference (about {limit} characters). The pasted text is longer, "
            f"and automatic transcription is unavailable. {unavailable}".strip()
        )
    raise ValueError(
        "A transcript of the first "
        f"{window_seconds:g} seconds is required, and automatic transcription is unavailable. "
        f"{unavailable}".strip()
    )


class CachingTranscriber:
    def __init__(self, inner, cache_dir: Path):
        self.inner = inner
        self.cache_dir = Path(cache_dir)

    def transcribe(self, path: Path, language: str) -> str:
        path = Path(path)
        key = f"{path.stem}-{path.stat().st_mtime_ns}-{language}.txt"
        destination = self.cache_dir / "transcripts" / key
        if destination.is_file():
            cached = destination.read_text(encoding="utf-8").strip()
            if cached:
                return cached
        text = self.inner.transcribe(path, language).strip()
        if text:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
        return text


class LazyTranscriber:
    """Load Whisper on the first reference that actually needs it."""

    def __init__(self, runtime: str, model_id: str | None = None):
        self.runtime = runtime
        self.model_id = (model_id or "").strip()
        self._mlx = None
        self._whisper = None

    def transcribe(self, path: Path, language: str) -> str:
        language = (language or "pt").strip().lower() or "pt"
        if self.runtime == "mlx" or self._prefers_mlx():
            try:
                return self._mlx_transcribe(path, language)
            except TranscriptUnavailable:
                if self.runtime == "mlx":
                    raise
        return self._faster_transcribe(path, language)

    def _prefers_mlx(self) -> bool:
        model = self.model_id.lower()
        return model.startswith("mlx-community/") or "mlx" in model

    def _mlx_transcribe(self, path: Path, language: str) -> str:
        if self._mlx is None:
            try:
                from mlx_audio.stt import load
            except ImportError as exc:
                raise TranscriptUnavailable(
                    "mlx-audio STT is not installed. On Apple Silicon, scripts/install.sh "
                    "installs mlx-audio, which provides Whisper."
                ) from exc
            model_id = self.model_id or MLX_ASR_MODEL
            self._mlx = load(model_id)
            self._mlx_id = model_id
        spoken_language = language
        if "qwen3" in getattr(self, "_mlx_id", "").lower():
            spoken_language = _QWEN_ASR_LANGUAGE.get(language, language)
        result = self._mlx.generate(str(path), language=spoken_language)
        return _result_text(result)

    def _faster_transcribe(self, path: Path, language: str) -> str:
        if self._whisper is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise TranscriptUnavailable(
                    "faster-whisper is not installed. On Linux or CUDA, re-run "
                    "scripts/install.sh so the PyTorch backend installs it."
                ) from exc
            name = self.model_id or "tiny"
            if name.startswith("mlx-community/"):
                name = "tiny"
            self._whisper = WhisperModel(name, device="cpu", compute_type="int8")
        segments, _info = self._whisper.transcribe(str(path), language=language)
        return " ".join(segment.text.strip() for segment in segments if segment.text).strip()


def _result_text(result) -> str:
    if isinstance(result, str):
        return result.strip()
    text = getattr(result, "text", None)
    if text:
        return str(text).strip()
    return ""
