#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
FISH_SPEECH_ROOT="${FISH_SPEECH_ROOT:-$ROOT/.cache/fish-speech}"
FISH_SPEECH_REF="${FISH_SPEECH_REF:-0f8a12e8a651062fdee9c33a894bdccd3a8513f1}"

command -v "$PYTHON_BIN" >/dev/null || { echo "Python is required" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "FFmpeg is required" >&2; exit 1; }
command -v git >/dev/null || { echo "Git is required" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required (install from https://docs.astral.sh/uv/)" >&2; exit 1; }

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
uv pip install --python "$VENV_DIR/bin/python" --upgrade pip
uv pip install --python "$VENV_DIR/bin/python" -e "${ROOT}[test]"

mkdir -p "$(dirname "$FISH_SPEECH_ROOT")"
if [[ ! -d "$FISH_SPEECH_ROOT/.git" ]]; then
  git clone https://github.com/fishaudio/fish-speech.git "$FISH_SPEECH_ROOT"
fi
git -C "$FISH_SPEECH_ROOT" fetch --depth 1 origin "$FISH_SPEECH_REF"
git -C "$FISH_SPEECH_ROOT" checkout --detach "$FISH_SPEECH_REF"

# S2-Pro is the local default because it supports CPU fallback; override with FISH_EXTRA.
FISH_EXTRA="${FISH_EXTRA:-cpu}"
if [[ "${FISH_SKIP_PYAUDIO:-1}" == "1" ]]; then
  # pyaudio is only used by the upstream playback helper, not by its API server.
  # The managed image has no C compiler/PortAudio headers, so install the server
  # runtime without that optional playback package.
  uv pip install --python "$VENV_DIR/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.8.0 torchaudio==2.8.0
  uv pip install --python "$VENV_DIR/bin/python" -e "$FISH_SPEECH_ROOT" --no-deps
  uv pip install --python "$VENV_DIR/bin/python" \
    numpy transformers==5.19.0 'hydra-core>=1.3.2' 'natsort>=8.4.0' 'einops>=0.7.0' \
    'librosa>=0.10.1' 'kui>=1.6.0' 'uvicorn>=0.30.0' 'loguru>=0.6.0' 'pyrootutils>=1.0.4' \
    'resampy>=0.4.3' 'einx[torch]==0.2.2' pydub modelscope==1.17.1 \
    opencc-python-reimplemented==0.1.7 silero-vad ormsgpack 'tiktoken>=0.8.0' \
    'pydantic==2.9.2' cachetools descript-audio-codec safetensors tokenizers==0.23.2 \
    'soundfile>=0.12.1'
else
  uv pip install --python "$VENV_DIR/bin/python" --extra-index-url https://download.pytorch.org/whl/cpu -e "${FISH_SPEECH_ROOT}[$FISH_EXTRA]"
fi

cat <<MSG
Voztudio dependencies installed.
Fish Speech source: $FISH_SPEECH_ROOT
Fish Speech revision: $(git -C "$FISH_SPEECH_ROOT" rev-parse HEAD)
Next: download S2-Pro weights with:
  $VENV_DIR/bin/hf download fishaudio/s2-pro --local-dir "$ROOT/.cache/checkpoints/s2-pro"
MSG
