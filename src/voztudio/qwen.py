"""Qwen3-TTS Base voice clone, in process.

MLX, from mlx-audio ``Model.generate`` on a Base checkpoint::

    model.generate(text=..., ref_audio=path, ref_text=transcript, lang_code="portuguese")

Do not pass ``voice``. Base cloning ignores ``instruct``.

PyTorch, from the official ``qwen-tts`` package::

    Qwen3TTSModel.from_pretrained(...)
    model.create_voice_clone_prompt(ref_audio=..., ref_text=...)
    model.generate_voice_clone(text=..., language="portuguese", voice_clone_prompt=...)

Language names are the checkpoint's ``codec_language_id`` keys.
"""

from __future__ import annotations

import threading
from pathlib import Path

from voztudio.config import Settings
from voztudio.engine import AudioBuffer, Synthesis
from voztudio.duration import qwen_max_tokens
from voztudio.models import QWEN_LANGUAGES, REFERENCE_MAX_SECONDS, repo_for
from voztudio.reference import clip_warning, prepare_reference
from voztudio.runtime import (
    as_floats,
    load_mlx,
    load_qwen_torch,
    model_source,
    samples_from_mlx,
    torch_device,
)
from voztudio.stylemap import HINDI_WARNING, strip_qwen_style


class Qwen3Backend:
    def __init__(
        self,
        name: str,
        model_id: str,
        runtime: str,
        cache_dir: Path,
        requested_device: str = "auto",
        runtime_explicit: bool = False,
        model_dir: Path | None = None,
        model=None,
    ):
        if name not in {"qwen3", "qwen3-fast"}:
            raise ValueError(f"Unsupported Qwen3 backend: {name}")
        self.name = name
        self.model_id = model_id
        self.runtime = runtime
        self.cache_dir = Path(cache_dir)
        self.requested_device = requested_device
        self.runtime_explicit = runtime_explicit
        self.model_dir = model_dir
        self.supports_style = False
        self.reference_max_seconds = REFERENCE_MAX_SECONDS[name]
        self.resolved_device = "mlx" if runtime == "mlx" else requested_device
        self._model = model
        self._lock = threading.Lock()
        self._fallback_warning: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "Qwen3Backend":
        model_dir = _model_dir_override()
        return cls(
            name=settings.backend,
            model_id=settings.model_id,
            runtime=settings.runtime,
            cache_dir=settings.cache_dir,
            requested_device=settings.device,
            runtime_explicit=settings.runtime_explicit,
            model_dir=model_dir,
        )

    def synthesize(
        self,
        text: str,
        reference_audio: str | Path,
        reference_transcript: str | None,
        style: str | None,
        language: str,
    ) -> Synthesis:
        transcript = (reference_transcript or "").strip()
        if not transcript:
            raise ValueError(
                "Qwen3-TTS voice clone needs the reference transcript; "
                "register the voice with the exact words in the clip"
            )
        spoken, warnings = strip_qwen_style(text, style)
        if not spoken:
            raise ValueError("Text must not be empty after removing style tags")
        lang_code = QWEN_LANGUAGES.get(language, "auto")
        warning_list = list(warnings)
        if language == "hi":
            warning_list.append(HINDI_WARNING)
        path, trimmed = prepare_reference(
            Path(reference_audio), self.cache_dir, self.reference_max_seconds
        )
        if trimmed:
            warning_list.append(clip_warning(self.name, self.reference_max_seconds))
        with self._lock:
            model = self._model if self._model is not None else self._load()
            if self.runtime == "mlx":
                samples, rate = self._mlx(model, spoken, path, transcript, lang_code)
            else:
                samples, rate = self._pytorch(model, spoken, path, transcript, lang_code)
        if self._fallback_warning:
            warning_list.insert(0, self._fallback_warning)
        return Synthesis(AudioBuffer(samples, rate), tuple(warning_list))

    def health(self) -> tuple[bool, str]:
        try:
            if self._model is None:
                self._import_runtime()
            if self.runtime == "pytorch" and self._model is None:
                self.resolved_device = torch_device(self.requested_device)
            elif self.runtime == "mlx":
                self.resolved_device = "mlx"
        except Exception as exc:
            return False, str(exc)
        detail = f"{self.name} ready ({self.runtime}, {self.model_id}); weights load on first request"
        if self._fallback_warning:
            detail = f"{detail}. {self._fallback_warning}"
        return True, detail

    def _import_runtime(self) -> None:
        if self.runtime == "mlx":
            try:
                import mlx_audio.tts.utils  # noqa: F401
            except ImportError as exc:
                if self.runtime_explicit:
                    raise RuntimeError(
                        "mlx-audio is not installed. MLX runs on Apple Silicon. "
                        "On Linux or CUDA, unset VOZTUDIO_RUNTIME or set "
                        "VOZTUDIO_RUNTIME=pytorch and install qwen-tts."
                    ) from exc
                fallback = repo_for(self.name, "pytorch")
                self._fallback_warning = (
                    "mlx-audio is not installed, so Qwen3-TTS is using the official "
                    f"PyTorch weights {fallback}."
                )
                self.runtime = "pytorch"
                self.model_id = fallback
                self.resolved_device = self.requested_device
        if self.runtime == "pytorch":
            try:
                import qwen_tts  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "qwen-tts is not installed. On Apple Silicon, scripts/install.sh "
                    "installs mlx-audio. On Linux or CUDA, re-run scripts/install.sh "
                    "so it installs the official qwen-tts package."
                ) from exc

    def _load(self):
        self._import_runtime()
        source = model_source(
            self.name,
            self.runtime,
            self.model_id,
            self.cache_dir,
            self.model_dir,
        )
        if self.runtime == "mlx":
            self.resolved_device = "mlx"
            self._model = load_mlx(source)
        else:
            self.resolved_device = torch_device(self.requested_device)
            self._model = load_qwen_torch(source, self.resolved_device)
        return self._model

    @staticmethod
    def _mlx(model, text: str, path: Path, transcript: str, lang_code: str):
        # Base cloning: ref_audio + ref_text. voice and instruct are omitted.
        # max_tokens bounds the codec loop. The model still stops on codec EOS.
        results = model.generate(
            text=text,
            ref_audio=str(path),
            ref_text=transcript,
            lang_code=lang_code,
            stream=False,
            max_tokens=qwen_max_tokens(text),
        )
        return samples_from_mlx(results)

    def _pytorch(self, model, text: str, path: Path, transcript: str, lang_code: str):
        prompt = self._clone_prompt(model, path, transcript)
        limit = qwen_max_tokens(text)
        if prompt is None:
            wavs, rate = model.generate_voice_clone(
                text=text,
                language=lang_code,
                ref_audio=str(path),
                ref_text=transcript,
                max_new_tokens=limit,
            )
        else:
            wavs, rate = model.generate_voice_clone(
                text=text,
                language=lang_code,
                voice_clone_prompt=prompt,
                max_new_tokens=limit,
            )
        if not wavs:
            raise RuntimeError("Qwen3-TTS returned no audio")
        samples = as_floats(wavs[0])
        if not samples:
            raise RuntimeError("Qwen3-TTS returned empty audio")
        return samples, int(rate)

    def _clone_prompt(self, model, path: Path, transcript: str):
        create = getattr(model, "create_voice_clone_prompt", None)
        if not callable(create):
            return None
        cache = getattr(self, "_prompts", None)
        if cache is None:
            cache = {}
            self._prompts = cache
        key = (str(path), path.stat().st_mtime_ns, transcript, self.model_id)
        prompt = cache.get(key)
        if prompt is None:
            prompt = create(ref_audio=str(path), ref_text=transcript)
            cache[key] = prompt
        return prompt


def _model_dir_override() -> Path | None:
    import os

    raw = os.getenv("VOZTUDIO_MODEL_DIR", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()
