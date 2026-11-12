#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib.sh
source "$ROOT/scripts/lib.sh"
cd "$ROOT"

load_env_file "$ROOT/.env"

VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/.cache/checkpoints/s2-pro}"
FISH_SPEECH_ROOT="${FISH_SPEECH_ROOT:-$ROOT/.cache/fish-speech}"
FISH_HOST="${FISH_HOST:-127.0.0.1}"
FISH_PORT="${FISH_PORT:-8080}"
VOZTUDIO_HOST="${VOZTUDIO_HOST:-127.0.0.1}"
VOZTUDIO_PORT="${VOZTUDIO_PORT:-7860}"
FISH_START_TIMEOUT="${FISH_START_TIMEOUT:-900}"
FISH_LOG="${FISH_LOG:-$ROOT/.cache/fish-server.log}"

[[ -x "$VENV_DIR/bin/python" ]] || { echo "Run scripts/install.sh first" >&2; exit 1; }
ensure_loopback_host "$VOZTUDIO_HOST" "Voztudio"
ensure_loopback_host "$FISH_HOST" "Fish Speech"
[[ -f "$CHECKPOINT_ROOT/config.json" && -f "$CHECKPOINT_ROOT/codec.pth" ]] || {
  echo "Fish Speech checkpoint missing: $CHECKPOINT_ROOT" >&2
  echo "config.json and codec.pth are both required." >&2
  echo "Download them with: $VENV_DIR/bin/hf download fishaudio/s2-pro --local-dir $CHECKPOINT_ROOT" >&2
  exit 1
}

export FISH_SPEECH_URL="${FISH_SPEECH_URL:-$(fish_base_url "$FISH_HOST" "$FISH_PORT")}"
export VOZTUDIO_HOST VOZTUDIO_PORT

REQUESTED_DEVICE="$("$VENV_DIR/bin/python" - <<'PY'
import os
import torch

requested = os.environ.get("VOZTUDIO_DEVICE", "auto").strip() or "auto"
if requested != "auto":
    print(requested)
elif torch.cuda.is_available():
    print("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
)"

mkdir -p "$(dirname "$FISH_LOG")"
API_ARGS=()
if [[ -n "${FISH_SPEECH_API_KEY:-}" ]]; then
  API_ARGS=(--api-key "$FISH_SPEECH_API_KEY")
fi
PYTHONUNBUFFERED=1 "$VENV_DIR/bin/python" "$FISH_SPEECH_ROOT/tools/api_server.py" \
  --listen "$(format_listen "$FISH_HOST" "$FISH_PORT")" \
  --llama-checkpoint-path "$CHECKPOINT_ROOT" \
  --decoder-checkpoint-path "$CHECKPOINT_ROOT/codec.pth" \
  --decoder-config-name modded_dac_vq \
  --device "$REQUESTED_DEVICE" \
  "${API_ARGS[@]}" >"$FISH_LOG" 2>&1 &
FISH_PID=$!
cleanup() { kill "$FISH_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 "$FISH_START_TIMEOUT"); do
  if curl -fsS "$FISH_SPEECH_URL/v1/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$FISH_PID" 2>/dev/null; then
    echo "Fish Speech server exited before readiness. Last log lines:" >&2
    tail -n 40 "$FISH_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done

if [[ "$ready" != "1" ]]; then
  echo "Timed out waiting for Fish Speech at $FISH_SPEECH_URL" >&2
  tail -n 40 "$FISH_LOG" >&2 || true
  exit 1
fi

VOZTUDIO_RESOLVED_DEVICE="$(resolve_logged_device "$REQUESTED_DEVICE" "$FISH_LOG")"
export VOZTUDIO_RESOLVED_DEVICE
"$VENV_DIR/bin/python" -m voztudio.app
exit $?
