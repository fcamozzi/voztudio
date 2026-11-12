"""VoxCPM2 voice clone, in process.

MLX, from mlx-audio ``Model.generate``. A style note is ``instruct`` with no
parentheses; the library prefixes ``(instruct)``::

    model.generate(text=..., ref_audio=path, instruct="sussurrando")

``ref_text`` is accepted by the signature and is not used for conditioning.
The Python API clones from ``ref_audio``.

PyTorch, from the official ``voxcpm`` package. Controllable cloning keeps the
style channel, which continuation mode turns off::

    VoxCPM.from_pretrained(repo, load_denoiser=False, device=...)
    model.generate(
        text="(sussurrando)Olá.",
        reference_wav_path=path,
        cfg_value=2.0,
        inference_timesteps=10,
    )

``cfg_value`` and ``inference_timesteps`` are the upstream defaults.
"""

from __future__ import annotations

import threading
from pathlib import Path

from voztudio.config import Settings
from voztudio.engine import AudioBuffer, Synthesis
from voztudio.duration import vox_max_tokens
from voztudio.models import REFERENCE_MAX_SECONDS, repo_for
from voztudio.reference import clip_warning, prepare_reference
from voztudio.runtime import as_floats, load_mlx, load_vox_torch, model_source, samples_from_mlx, torch_device
from voztudio.stylemap import vox_instruction


class VoxCPM2Backend:
    def __init__(
        self,
        model_id: str,
        runtime: str,
        cache_dir: Path,
        requested_device: str = "auto",
        runtime_explicit: bool = False,
        model_dir: Path | None = None,
        model=None,
    ):
        self.name = "voxcpm2"
        self.model_id = model_id
        self.runtime = runtime
        self.cache_dir = Path(cache_dir)
        self.requested_device = requested_device
        self.runtime_explicit = runtime_explicit
        self.model_dir = model_dir
        self.supports_style = True
        self.reference_max_seconds = REFERENCE_MAX_SECONDS["voxcpm2"]
        self.resolved_device = "mlx" if runtime == "mlx" else requested_device
        self._model = model
        self._lock = threading.Lock()
        self._fallback_warning: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "VoxCPM2Backend":
        return cls(
            model_id=settings.model_id,
            runtime=settings.runtime,
            cache_dir=settings.cache_dir,
            requested_device=settings.device,
            runtime_explicit=settings.runtime_explicit,
            model_dir=_model_dir_override(),
        )

    def synthesize(
        self,
        text: str,
        reference_audio: str | Path,
        reference_transcript: str | None,
        style: str | None,
        language: str,
    ) -> Synthesis:
        # VoxCPM2 has no language argument. OpenBMB says the text is synthesized
        # directly, and the wrapper still stores the language label.
        del language
        spoken, instruct, warnings = vox_instruction(text, style)
        if not spoken:
            raise ValueError("Text must not be empty after removing style tags")
        path, trimmed = prepare_reference(
            Path(reference_audio), self.cache_dir, self.reference_max_seconds
        )
        warning_list = list(warnings)
        if trimmed:
            warning_list.append(clip_warning(self.name, self.reference_max_seconds))
        with self._lock:
            model = self._model if self._model is not None else self._load()
            if self.runtime == "mlx":
                samples, rate = self._mlx(model, spoken, path, instruct, reference_transcript)
            else:
                samples, rate = self._pytorch(model, spoken, path, instruct)
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
        detail = f"voxcpm2 ready ({self.runtime}, {self.model_id}); weights load on first request"
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
                        "VOZTUDIO_RUNTIME=pytorch and install voxcpm."
                    ) from exc
                fallback = repo_for("voxcpm2", "pytorch")
                self._fallback_warning = (
                    "mlx-audio is not installed, so VoxCPM2 is using the official "
                    f"PyTorch weights {fallback}."
                )
                self.runtime = "pytorch"
                self.model_id = fallback
                self.resolved_device = self.requested_device
        if self.runtime == "pytorch":
            try:
                import voxcpm  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "voxcpm is not installed. On Apple Silicon, "
                    "VOZTUDIO_BACKEND=voxcpm2 scripts/install.sh installs mlx-audio. "
                    "On Linux or CUDA, the same command installs the official voxcpm package."
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
            self._model = load_vox_torch(source, self.resolved_device)
        return self._model

    @staticmethod
    def _mlx(model, text: str, path: Path, instruct: str | None, transcript: str | None):
        # max_tokens is the patch cap. min_tokens stays at the upstream default
        # of 2 so the stop head can end the clip; the loop breaks when that
        # head predicts stop after min_tokens real patches.
        budget = f"({instruct}){text}" if instruct else text
        kwargs = {
            "text": text,
            "ref_audio": str(path),
            "max_tokens": vox_max_tokens(budget),
            "min_tokens": 2,
        }
        if instruct:
            kwargs["instruct"] = instruct
        cleaned = (transcript or "").strip()
        if cleaned:
            kwargs["ref_text"] = cleaned
        return samples_from_mlx(model.generate(**kwargs))

    @staticmethod
    def _pytorch(model, text: str, path: Path, instruct: str | None):
        spoken = f"({instruct}){text}" if instruct else text
        wav = model.generate(
            text=spoken,
            reference_wav_path=str(path),
            cfg_value=2.0,
            inference_timesteps=10,
            min_len=2,
            max_len=vox_max_tokens(spoken),
        )
        samples = as_floats(wav)
        if not samples:
            raise RuntimeError("VoxCPM2 returned empty audio")
        rate = int(model.tts_model.sample_rate)
        return samples, rate


def _model_dir_override() -> Path | None:
    import os

    raw = os.getenv("VOZTUDIO_MODEL_DIR", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()
