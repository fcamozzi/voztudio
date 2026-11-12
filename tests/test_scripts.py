import shutil
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _bash(script: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/usr/bin/bash", "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_start_script_keeps_cleanup_trap_active():
    text = (ROOT / "scripts/start.sh").read_text()
    assert 'exec "$VENV_DIR/bin/python" -m voztudio.app' not in text
    assert '"$VENV_DIR/bin/python" -m voztudio.app' in text
    assert "codec.pth" in text
    assert "FISH_START_TIMEOUT:-900" in text
    assert "--api-key" in text
    assert "load_env_file" in text
    assert "set -a" not in text


def test_install_script_uses_uv_sync_and_python_resolver():
    text = (ROOT / "scripts/install.sh").read_text()
    assert "resolve_python" in text
    assert "uv sync" in text
    assert "--no-install-package pyaudio" in text
    assert "--extra" in text
    assert "transformers==5.19.0" not in text
    assert '"${FISH_SPEECH_ROOT}[$FISH_EXTRA]"' not in text


def test_resolve_python_prefers_python3_on_path(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    python3 = bindir / "python3"
    python3.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python3.chmod(python3.stat().st_mode | stat.S_IEXEC)
    result = _bash(
        "source scripts/lib.sh && resolve_python",
        {"PATH": str(bindir), "HOME": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("/python3")


def test_resolve_python_fails_when_neither_binary_exists(tmp_path):
    result = _bash(
        "source scripts/lib.sh && resolve_python",
        {"PATH": str(tmp_path), "HOME": str(tmp_path)},
    )
    assert result.returncode != 0
    assert "Python 3.11 or 3.12 is required" in result.stderr


def test_resolve_fish_extra_and_listen_and_device_log(tmp_path):
    isolated = {"PATH": str(tmp_path), "HOME": str(tmp_path)}
    extra = _bash("source scripts/lib.sh && resolve_fish_extra", isolated)
    assert extra.stdout.strip() == "cpu"
    chosen = _bash("source scripts/lib.sh && FISH_EXTRA=cu126 resolve_fish_extra", isolated)
    assert chosen.stdout.strip() == "cu126"
    system = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    if shutil.which("nvidia-smi", path=system["PATH"]) is None:
        detected = _bash("source scripts/lib.sh && resolve_fish_extra", system)
        assert detected.stdout.strip() == "cpu"
    listen = _bash(
        'source scripts/lib.sh && format_listen ::1 8080 && format_listen 127.0.0.1 7860 && '
        'fish_base_url ::1 8080',
        system,
    )
    assert listen.stdout.splitlines() == ["[::1]:8080", "127.0.0.1:7860", "http://[::1]:8080"]
    log = tmp_path / "fish.log"
    log.write_text("CUDA is not available, running on CPU.\n", encoding="utf-8")
    device = _bash(
        f'source scripts/lib.sh && resolve_logged_device cuda "{log}"',
        system,
    )
    assert device.returncode == 0, device.stderr
    assert device.stdout.strip() == "cpu"


def test_env_file_does_not_override_or_execute(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# comment\nKEEP=fromfile\nNEW=fromfile\nexport QUOTED="hello world"\n$(echo pwned)\n',
        encoding="utf-8",
    )
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "KEEP": "original"}
    result = _bash(
        f'source scripts/lib.sh && load_env_file "{env_file}" && printf "%s\\n" "$KEEP" "$NEW" "$QUOTED"',
        env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["original", "fromfile", "hello world"]


def test_loopback_guard_refuses_public_bind_without_opt_in(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    refused = _bash(
        "source scripts/lib.sh && ensure_loopback_host 0.0.0.0 Fish",
        env,
    )
    assert refused.returncode != 0
    assert "VOZTUDIO_ALLOW_PUBLIC_BIND" in refused.stderr
    allowed = _bash(
        "source scripts/lib.sh && ensure_loopback_host 127.0.0.1 Fish && "
        "ensure_loopback_host LocalHost Fish && ensure_loopback_host ::1 Fish && "
        "VOZTUDIO_ALLOW_PUBLIC_BIND=1 ensure_loopback_host 0.0.0.0 Fish",
        env,
    )
    assert allowed.returncode == 0, allowed.stderr
    assert "Warning" in allowed.stderr
