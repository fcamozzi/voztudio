#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib.sh
source "$ROOT/scripts/lib.sh"

VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
FISH_SPEECH_ROOT="${FISH_SPEECH_ROOT:-$ROOT/.cache/fish-speech}"
FISH_SPEECH_REF="${FISH_SPEECH_REF:-0f8a12e8a651062fdee9c33a894bdccd3a8513f1}"

PYTHON_BIN="$(resolve_python)"
"$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)' \
  || { echo "Python 3.11 or 3.12 is required (found $PYTHON_BIN)" >&2; exit 1; }
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

FISH_EXTRA="$(resolve_fish_extra)"
# uv sync reads tool.uv.override-dependencies (protobuf) and the torch index
# for the selected extra. --inexact keeps the voztudio packages already
# installed. pyaudio is only imported by the upstream playback client.
export VIRTUAL_ENV="$VENV_DIR"
uv sync \
  --directory "$FISH_SPEECH_ROOT" \
  --python "$VENV_DIR/bin/python" \
  --active \
  --inexact \
  --no-dev \
  --extra "$FISH_EXTRA" \
  --no-install-package pyaudio

cat <<MSG
Voztudio dependencies installed.
Fish Speech source: $FISH_SPEECH_ROOT
Fish Speech revision: $(git -C "$FISH_SPEECH_ROOT" rev-parse HEAD)
PyTorch extra: $FISH_EXTRA
Next: download S2-Pro weights with:
  $VENV_DIR/bin/hf download fishaudio/s2-pro --local-dir "$ROOT/.cache/checkpoints/s2-pro"
MSG
