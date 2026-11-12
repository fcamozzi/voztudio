#!/usr/bin/env bash
# Shared helpers for install.sh and start.sh. Safe to source; it does not
# install packages or start processes.

# Print the interpreter path. PYTHON_BIN wins when it is on PATH; otherwise
# python3, then python. Fails when none of them exist.
resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
      command -v "$PYTHON_BIN"
      return 0
    fi
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    return 1
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  echo "Python 3.11 or 3.12 is required" >&2
  return 1
}

# FISH_EXTRA wins. A working nvidia-smi selects the cu128 PyTorch extra;
# otherwise the CPU extra. uv sync uses this so the wheel matches the machine.
resolve_fish_extra() {
  if [[ -n "${FISH_EXTRA:-}" ]]; then
    printf '%s\n' "$FISH_EXTRA"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' cu128
    return 0
  fi
  printf '%s\n' cpu
}

# Export KEY=VALUE lines from a dotenv file only when KEY is unset.
# Comments and blank lines are ignored. Quotes around a value are removed.
# The file is not sourced, so it cannot run shell commands or override
# variables that the caller already exported.
load_env_file() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
    if [[ "$line" == export\ * ]]; then
      line="${line#export }"
      line="${line#"${line%%[![:space:]]*}"}"
    fi
    [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi
    if [[ -z "${!key+x}" ]]; then
      export "$key=$value"
    fi
  done <"$file"
}

# Allow loopback hosts unless VOZTUDIO_ALLOW_PUBLIC_BIND is 1 or true.
# On opt-in, print one warning and continue.
ensure_loopback_host() {
  local host="$1"
  local label="$2"
  host="${host#[}"
  host="${host%]}"
  local lower
  lower="$(printf '%s' "$host" | tr '[:upper:]' '[:lower:]')"
  if [[ "$lower" == "127.0.0.1" || "$lower" == "localhost" || "$host" == "::1" ]]; then
    return 0
  fi
  local allow
  allow="$(printf '%s' "${VOZTUDIO_ALLOW_PUBLIC_BIND:-}" | tr '[:upper:]' '[:lower:]')"
  if [[ "$allow" == "1" || "$allow" == "true" ]]; then
    echo "Warning: binding ${label} on ${host}, which is reachable beyond this machine." >&2
    return 0
  fi
  echo "Refusing to bind ${label} on ${host}. Set VOZTUDIO_ALLOW_PUBLIC_BIND=1 to allow a non-loopback address." >&2
  return 1
}

# Base URL for curl and the wrapper. IPv6 hosts stay bracketed.
fish_base_url() {
  local host="$1"
  local port="$2"
  printf 'http://%s\n' "$(format_listen "$host" "$port")"
}

# Fish Speech splits --listen on the last colon only when the host is bracketed.
format_listen() {
  local host="$1"
  local port="$2"
  host="${host#[}"
  host="${host%]}"
  if [[ "$host" == *:* ]]; then
    printf '[%s]:%s\n' "$host" "$port"
  else
    printf '%s:%s\n' "$host" "$port"
  fi
}

# ModelManager can override --device. The startup log is the device that ran.
resolve_logged_device() {
  local requested="$1"
  local log="$2"
  if [[ -f "$log" ]] && grep -q "running on mps" "$log"; then
    printf '%s\n' mps
  elif [[ -f "$log" ]] && grep -q "running on CPU" "$log"; then
    printf '%s\n' cpu
  else
    printf '%s\n' "$requested"
  fi
}
