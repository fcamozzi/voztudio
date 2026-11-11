from voztudio.config import get_settings


def test_defaults_use_loopback_and_target_languages(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = get_settings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 7860
    assert settings.languages == ("en", "pt", "es", "hi", "zh", "fr")
    assert settings.data_dir == tmp_path / "data"


def test_environment_overrides_are_parsed(monkeypatch, tmp_path):
    monkeypatch.setenv("VOZTUDIO_DATA_DIR", str(tmp_path / "custom-data"))
    monkeypatch.setenv("VOZTUDIO_DEVICE", "cpu")
    monkeypatch.setenv("VOZTUDIO_PORT", "9001")
    settings = get_settings()
    assert settings.data_dir == tmp_path / "custom-data"
    assert settings.device == "cpu"
    assert settings.port == 9001
