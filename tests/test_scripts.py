from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_start_script_keeps_cleanup_trap_active():
    text = (ROOT / "scripts/start.sh").read_text()
    assert 'exec "$VENV_DIR/bin/python" -m voztudio.app' not in text
    assert '"$VENV_DIR/bin/python" -m voztudio.app' in text


def test_install_script_uses_braced_fish_root_extra_path():
    text = (ROOT / "scripts/install.sh").read_text()
    assert '"${FISH_SPEECH_ROOT}[$FISH_EXTRA]"' in text
