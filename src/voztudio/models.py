"""Backend ids, checkpoint repos, and reference-length caps.

Repo ids were checked against the upstream model cards and mlx-audio docs:

* Qwen3-TTS Base 8-bit MLX snapshots on mlx-community, plus the official
  PyTorch repos under Qwen/.
* VoxCPM2 8-bit is the quantized MLX snapshot documented by mlx-audio.
  The PyTorch package loads ``openbmb/VoxCPM2``.
* Fish Speech stays on ``fishaudio/s2-pro``.
"""

from __future__ import annotations

import platform

BACKENDS = ("qwen3", "qwen3-fast", "voxcpm2", "fish")
RUNTIMES = ("mlx", "pytorch", "fish-http")

# (backend, runtime) -> Hugging Face repo id.
MODEL_REPOS = {
    ("qwen3", "mlx"): "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    ("qwen3", "pytorch"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ("qwen3-fast", "mlx"): "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
    ("qwen3-fast", "pytorch"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ("voxcpm2", "mlx"): "mlx-community/VoxCPM2-8bit",
    ("voxcpm2", "pytorch"): "openbmb/VoxCPM2",
    ("fish", "fish-http"): "fishaudio/s2-pro",
}

CHECKPOINT_DIRS = {
    "qwen3": "qwen3",
    "qwen3-fast": "qwen3-fast",
    "voxcpm2": "voxcpm2",
    "fish": "s2-pro",
}

# Upload validation stays 10–90 s. These caps are what synthesis sends.
# Qwen3-TTS Base is described as 3-second rapid cloning; 15 s is the wrapper
# cap inside the 10–20 s window so a 90 s upload is not the whole prompt.
# VoxCPM's usage guide calls 5–30 s practical; the wrapper keeps the first 20 s.
# Fish Speech's recommended range in this project is 10–30 s.
REFERENCE_MAX_SECONDS = {
    "qwen3": 15.0,
    "qwen3-fast": 15.0,
    "voxcpm2": 20.0,
    "fish": 30.0,
}

# codec_language_id keys from the published Qwen3-TTS Base config.
# Hindi is not in that map. "auto" is the model's own fallback.
QWEN_LANGUAGES = {
    "en": "english",
    "pt": "portuguese",
    "es": "spanish",
    "zh": "chinese",
    "fr": "french",
    "hi": "auto",
}


def default_runtime() -> str:
    """MLX on Apple Silicon. PyTorch everywhere else, including Linux and CUDA."""
    machine = platform.machine().lower()
    if platform.system() == "Darwin" and machine in {"arm64", "aarch64"}:
        return "mlx"
    return "pytorch"


def repo_for(backend: str, runtime: str) -> str:
    if backend == "fish":
        return MODEL_REPOS[("fish", "fish-http")]
    key = (backend, "mlx" if runtime == "mlx" else "pytorch")
    try:
        return MODEL_REPOS[key]
    except KeyError as exc:
        raise ValueError(f"No checkpoint for backend {backend!r} runtime {runtime!r}") from exc


def checkpoint_name(backend: str) -> str:
    try:
        return CHECKPOINT_DIRS[backend]
    except KeyError as exc:
        raise ValueError(f"Unknown backend: {backend}") from exc
