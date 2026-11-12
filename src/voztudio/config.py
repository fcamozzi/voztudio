from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from voztudio.models import BACKENDS, default_runtime, repo_for

TARGET_LANGUAGES = ("en", "pt", "es", "hi", "zh", "fr")
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_DEFAULT_UPLOAD_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    cache_dir: Path
    fish_speech_root: Path
    model_id: str
    fish_speech_url: str
    device: str
    host: str
    port: int
    languages: tuple[str, ...] = TARGET_LANGUAGES
    fish_timeout_s: float = 900.0
    max_upload_bytes: int = _DEFAULT_UPLOAD_BYTES
    voice_min_seconds: float = 10.0
    voice_max_seconds: float = 90.0
    allow_public_bind: bool = False
    resolved_device: str = "auto"
    fish_api_key: str | None = None
    backend: str = "qwen3"
    runtime: str = "pytorch"
    runtime_explicit: bool = False


def _path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _optional(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def get_settings() -> Settings:
    # Explicit path: find_dotenv() walks parents and would pick up a developer
    # .env when tests change the working directory.
    load_dotenv(Path.cwd() / ".env", override=False)
    backend = os.getenv("VOZTUDIO_BACKEND", "qwen3").strip().lower() or "qwen3"
    if backend not in BACKENDS:
        raise SystemExit(
            f"Unknown VOZTUDIO_BACKEND={backend}. "
            "Expected one of: qwen3, qwen3-fast, voxcpm2, fish."
        )
    runtime_env = os.getenv("VOZTUDIO_RUNTIME", "").strip().lower()
    if backend == "fish":
        runtime = "fish-http"
        runtime_explicit = False
    else:
        runtime_explicit = bool(runtime_env)
        runtime = runtime_env or default_runtime()
        if runtime not in {"mlx", "pytorch"}:
            raise SystemExit(
                f"Unknown VOZTUDIO_RUNTIME={runtime}. Expected mlx or pytorch."
            )
    override = os.getenv("VOZTUDIO_MODEL_ID", "").strip()
    if backend == "fish":
        model_id = override or os.getenv("FISH_MODEL_ID", repo_for("fish", "fish-http"))
    else:
        model_id = override or repo_for(backend, runtime)
    return Settings(
        data_dir=_path("VOZTUDIO_DATA_DIR", "data"),
        cache_dir=_path("VOZTUDIO_CACHE_DIR", ".cache"),
        fish_speech_root=_path("FISH_SPEECH_ROOT", ".cache/fish-speech"),
        model_id=model_id,
        fish_speech_url=os.getenv("FISH_SPEECH_URL", "http://127.0.0.1:8080"),
        device=os.getenv("VOZTUDIO_DEVICE", "auto"),
        host=os.getenv("VOZTUDIO_HOST", "127.0.0.1"),
        port=int(os.getenv("VOZTUDIO_PORT", "7860")),
        fish_timeout_s=float(os.getenv("VOZTUDIO_FISH_TIMEOUT", "900")),
        max_upload_bytes=int(os.getenv("VOZTUDIO_MAX_UPLOAD_BYTES", str(_DEFAULT_UPLOAD_BYTES))),
        voice_min_seconds=float(os.getenv("VOZTUDIO_VOICE_MIN_SECONDS", "10")),
        voice_max_seconds=float(os.getenv("VOZTUDIO_VOICE_MAX_SECONDS", "90")),
        allow_public_bind=_flag("VOZTUDIO_ALLOW_PUBLIC_BIND"),
        resolved_device=os.getenv("VOZTUDIO_RESOLVED_DEVICE", "auto").strip() or "auto",
        fish_api_key=_optional("FISH_SPEECH_API_KEY"),
        backend=backend,
        runtime=runtime,
        runtime_explicit=runtime_explicit,
    )


def normalize_host(host: str) -> str:
    """Drop a single pair of brackets so `[::1]` is the same host as `::1`."""
    normalized = host.strip()
    if len(normalized) >= 2 and normalized[0] == "[" and normalized[-1] == "]":
        return normalized[1:-1]
    return normalized


def ensure_bind_allowed(host: str, allow_public: bool) -> None:
    """Refuse a non-loopback bind unless the operator opted in."""
    normalized = normalize_host(host)
    if normalized.lower() in _LOOPBACK_HOSTS or normalized == "::1":
        return
    if allow_public:
        print(
            f"Warning: binding Voztudio on {normalized}, which is reachable beyond this machine.",
            file=sys.stderr,
        )
        return
    raise SystemExit(
        f"Refusing to bind Voztudio on {normalized}. "
        "Set VOZTUDIO_ALLOW_PUBLIC_BIND=1 to allow a non-loopback address."
    )
