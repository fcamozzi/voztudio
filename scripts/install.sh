#!/usr/bin/env bash
# Install Voztudio. Safe on macOS system bash 3.2: no associative arrays,
# no mapfile, and no unquoted empty-array expansion under set -u.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib.sh
source "$ROOT/scripts/lib.sh"
cd "$ROOT"

load_env_file "$ROOT/.env"

VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
BACKEND="${VOZTUDIO_BACKEND:-qwen3}"
export VOZTUDIO_BACKEND="$BACKEND"

PYTHON_BIN="$(resolve_python)"
"$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)' \
  || { echo "Python 3.11 or 3.12 is required (found $PYTHON_BIN)" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "FFmpeg is required" >&2; exit 1; }
command -v git >/dev/null || { echo "Git is required" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required (install from https://docs.astral.sh/uv/)" >&2; exit 1; }

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# The wrapper venv never installs Fish Speech. Doing that in this venv
# downgraded numpy to 1.26.4 while pandas stayed on 3.x and broke `import gradio`.
uv pip install --python "$VENV_DIR/bin/python" --upgrade pip
uv pip install --python "$VENV_DIR/bin/python" -e "${ROOT}[test]"
uv pip install --python "$VENV_DIR/bin/python" "numpy>=2.0.2,<3"

install_fish() {
  local fish_venv fish_root fish_ref extra
  fish_venv="${FISH_VENV_DIR:-$ROOT/.venv-fish}"
  fish_root="${FISH_SPEECH_ROOT:-$ROOT/.cache/fish-speech}"
  fish_ref="${FISH_SPEECH_REF:-0f8a12e8a651062fdee9c33a894bdccd3a8513f1}"
  if [[ ! -x "$fish_venv/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$fish_venv"
  fi
  mkdir -p "$(dirname "$fish_root")"
  if [[ ! -d "$fish_root/.git" ]]; then
    git clone https://github.com/fishaudio/fish-speech.git "$fish_root"
  fi
  git -C "$fish_root" fetch --depth 1 origin "$fish_ref"
  git -C "$fish_root" checkout --detach "$fish_ref"
  extra="$(resolve_fish_extra)"
  # uv sync reads tool.uv.override-dependencies (protobuf) and the torch index
  # for the selected extra. --inexact keeps packages already in this venv.
  # pyaudio is only imported by the upstream playback client.
  (
    export VIRTUAL_ENV="$fish_venv"
    uv sync \
      --directory "$fish_root" \
      --python "$fish_venv/bin/python" \
      --active \
      --inexact \
      --no-dev \
      --extra "$extra" \
      --no-install-package pyaudio
  )
  echo "Fish Speech venv: $fish_venv"
  echo "Fish Speech source: $fish_root"
  echo "Fish Speech revision: $(git -C "$fish_root" rev-parse HEAD)"
  echo "PyTorch extra: $extra"
}

install_runtime() {
  local os_name arch runtime
  os_name="$(uname -s)"
  arch="$(uname -m)"
  runtime="${VOZTUDIO_RUNTIME:-}"
  if [[ -z "$runtime" ]]; then
    if [[ "$os_name" == "Darwin" && "$arch" == "arm64" ]]; then
      runtime="mlx"
    else
      runtime="pytorch"
    fi
  fi
  case "$runtime" in
    mlx)
      uv pip install --python "$VENV_DIR/bin/python" "mlx-audio>=0.6"
      ;;
    pytorch)
      if [[ "$BACKEND" == "voxcpm2" ]]; then
        uv pip install --python "$VENV_DIR/bin/python" "voxcpm>=2"
      else
        # qwen-tts pins transformers 4.57.3, which requires huggingface_hub<1.
        # mlx-audio wants huggingface_hub>=1, so the two runtimes stay apart.
        uv pip install --python "$VENV_DIR/bin/python" \
          "qwen-tts>=0.1.1" "huggingface_hub>=0.34,<1"
      fi
      uv pip install --python "$VENV_DIR/bin/python" "faster-whisper>=1.1"
      ;;
    *)
      echo "Unknown VOZTUDIO_RUNTIME=$runtime. Expected mlx or pytorch." >&2
      exit 1
      ;;
  esac
  # A backend package must not leave numpy on 1.26. Gradio's pandas needs >=2.
  uv pip install --python "$VENV_DIR/bin/python" "numpy>=2.0.2,<3"
}

hf_download() {
  if [[ -x "$VENV_DIR/bin/hf" ]]; then
    "$VENV_DIR/bin/hf" download "$@"
    return
  fi
  echo "The hf command is missing from $VENV_DIR. Re-run scripts/install.sh." >&2
  exit 1
}

download_checkpoint() {
  local repo dest
  repo="$1"
  dest="$2"
  if [[ "${VOZTUDIO_SKIP_DOWNLOAD:-}" == "1" ]]; then
    echo "Skipping weight download (VOZTUDIO_SKIP_DOWNLOAD=1)."
    echo "When you want the files: $VENV_DIR/bin/hf download $repo --local-dir $dest"
    return
  fi
  mkdir -p "$dest"
  hf_download "$repo" --local-dir "$dest"
}

resolve_checkpoint() {
  "$VENV_DIR/bin/python" -c '
import os
import sys
from voztudio.models import checkpoint_name, default_runtime, repo_for

backend = sys.argv[1]
requested = sys.argv[2]
runtime = requested or default_runtime()
if backend == "fish":
    runtime = "fish-http"
override = os.environ.get("VOZTUDIO_MODEL_ID", "").strip()
repo = override or repo_for(backend, runtime)
print(runtime)
print(repo)
print(checkpoint_name(backend))
' "$1" "$2"
}

case "$BACKEND" in
  fish)
    install_fish
    ;;
  qwen3|qwen3-fast|voxcpm2)
    install_runtime
    ;;
  *)
    echo "Unknown VOZTUDIO_BACKEND=$BACKEND. Expected qwen3, qwen3-fast, voxcpm2, or fish." >&2
    exit 1
    ;;
esac

"$VENV_DIR/bin/python" -c 'import gradio, numpy; parts = [int(item) for item in numpy.__version__.split(".")[:2]]; raise SystemExit(0 if parts >= [2, 0] else 1)' \
  || { echo "numpy in $VENV_DIR is older than 2.0. Gradio cannot start." >&2; exit 1; }

CHECKPOINT_INFO="$(resolve_checkpoint "$BACKEND" "${VOZTUDIO_RUNTIME:-}")"
RUNTIME_NAME="$(printf '%s\n' "$CHECKPOINT_INFO" | sed -n '1p')"
MODEL_REPO="$(printf '%s\n' "$CHECKPOINT_INFO" | sed -n '2p')"
CHECKPOINT_NAME="$(printf '%s\n' "$CHECKPOINT_INFO" | sed -n '3p')"
CACHE_DIR="${VOZTUDIO_CACHE_DIR:-$ROOT/.cache}"
CHECKPOINT_DIR="$CACHE_DIR/checkpoints/$CHECKPOINT_NAME"
download_checkpoint "$MODEL_REPO" "$CHECKPOINT_DIR"
if [[ "$RUNTIME_NAME" == "mlx" ]]; then
  hf_download "mlx-community/whisper-tiny-asr-fp16"
fi

cat <<MSG
Voztudio dependencies installed.
Backend: $BACKEND
Runtime: $RUNTIME_NAME
Checkpoint: $MODEL_REPO
Local snapshot: $CHECKPOINT_DIR
numpy: $("$VENV_DIR/bin/python" -c 'import numpy; print(numpy.__version__)')
Next: scripts/start.sh
MSG
