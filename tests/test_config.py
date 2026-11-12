import pytest

from voztudio.config import ensure_bind_allowed, get_settings


def test_defaults_use_loopback_and_target_languages(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = get_settings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 7860
    assert settings.languages == ("en", "pt", "es", "hi", "zh", "fr")
    assert settings.data_dir == tmp_path / "data"
    assert settings.resolved_device == "auto"
    assert settings.voice_min_seconds == 10
    assert settings.voice_max_seconds == 90


def test_environment_overrides_are_parsed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VOZTUDIO_DATA_DIR", str(tmp_path / "custom-data"))
    monkeypatch.setenv("VOZTUDIO_DEVICE", "cpu")
    monkeypatch.setenv("VOZTUDIO_PORT", "9001")
    monkeypatch.setenv("VOZTUDIO_RESOLVED_DEVICE", "cpu")
    settings = get_settings()
    assert settings.data_dir == tmp_path / "custom-data"
    assert settings.device == "cpu"
    assert settings.port == 9001
    assert settings.resolved_device == "cpu"


def test_dotenv_does_not_override_existing_variables(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "VOZTUDIO_PORT=9001\nVOZTUDIO_DEVICE=cpu\nFISH_SPEECH_API_KEY=secret\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VOZTUDIO_DEVICE", raising=False)
    monkeypatch.delenv("FISH_SPEECH_API_KEY", raising=False)
    monkeypatch.setenv("VOZTUDIO_PORT", "7860")
    settings = get_settings()
    assert settings.port == 7860
    assert settings.device == "cpu"
    assert settings.fish_api_key == "secret"


def test_bind_allows_loopback_and_warns_when_opted_in(capsys):
    ensure_bind_allowed("127.0.0.1", allow_public=False)
    ensure_bind_allowed("LocalHost", allow_public=False)
    ensure_bind_allowed("::1", allow_public=False)
    ensure_bind_allowed("[::1]", allow_public=False)
    with pytest.raises(SystemExit, match="VOZTUDIO_ALLOW_PUBLIC_BIND"):
        ensure_bind_allowed("0.0.0.0", allow_public=False)
    ensure_bind_allowed("0.0.0.0", allow_public=True)
    assert "Warning" in capsys.readouterr().err
