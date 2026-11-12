#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/.cache/checkpoints/s2-pro}"
FISH_SPEECH_ROOT="${FISH_SPEECH_ROOT:-$ROOT/.cache/fish-speech}"
FISH_HOST="${FISH_HOST:-127.0.0.1}"
FISH_PORT="${FISH_PORT:-8080}"
VOZTUDIO_HOST="${VOZTUDIO_HOST:-127.0.0.1}"
VOZTUDIO_PORT="${VOZTUDIO_PORT:-7860}"
FISH_START_TIMEOUT="${FISH_START_TIMEOUT:-900}"

[[ -x "$VENV_DIR/bin/python" ]] || { echo "Run scripts/install.sh first" >&2; exit 1; }
[[ -f "$CHECKPOINT_ROOT/config.json" ]] || {
  echo "Fish Speech checkpoint missing: $CHECKPOINT_ROOT" >&2
  echo "Download it with: $VENV_DIR/bin/hf download fishaudio/s2-pro --local-dir $CHECKPOINT_ROOT" >&2
  exit 1
}

export FISH_SPEECH_URL="${FISH_SPEECH_URL:-http://$FISH_HOST:$FISH_PORT}"
export VOZTUDIO_HOST VOZTUDIO_PORT

"$VENV_DIR/bin/python" "$FISH_SPEECH_ROOT/tools/api_server.py" \
  --listen "$FISH_HOST:$FISH_PORT" \
  --llama-checkpoint-path "$CHECKPOINT_ROOT" \
  --decoder-checkpoint-path "$CHECKPOINT_ROOT/codec.pth" \
  --decoder-config-name modded_dac_vq &
FISH_PID=$!
cleanup() { kill "$FISH_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

for _ in $(seq 1 "$FISH_START_TIMEOUT"); do
  if curl -fsS "$FISH_SPEECH_URL/v1/health" >/dev/null 2>&1; then
    "$VENV_DIR/bin/python" -m voztudio.app
    exit $?
  fi
  if ! kill -0 "$FISH_PID" 2>/dev/null; then
    echo "Fish Speech server exited before readiness" >&2
    exit 1
  fi
  sleep 1
done

echo "Timed out waiting for Fish Speech at $FISH_SPEECH_URL" >&2
exit 1
