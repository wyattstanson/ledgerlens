from pathlib import Path

import pytest

from ledgerlens.settings import REPO_ROOT, Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("LEDGERLENS_DB", "LEDGERLENS_SEED", "LEDGERLENS_MIN_GROUP_N", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_defaults_come_from_settings_toml_and_resolve_against_repo_root():
    settings = Settings()
    assert settings.seed == 42
    assert settings.min_group_n == 5
    assert settings.db == (REPO_ROOT / "data" / "warehouse.duckdb").resolve()
    assert settings.anthropic_api_key is None


def test_environment_overrides_toml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("LEDGERLENS_DB", str(tmp_path / "other.duckdb"))
    monkeypatch.setenv("LEDGERLENS_SEED", "7")
    settings = Settings()
    assert settings.db == tmp_path / "other.duckdb"
    assert settings.seed == 7


def test_relative_env_path_resolves_against_repo_root_not_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LEDGERLENS_DB", "data/x.duckdb")
    assert Settings().db == (REPO_ROOT / "data" / "x.duckdb").resolve()


def test_api_key_is_secret_and_never_printed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    settings = Settings()
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-test-not-real"
    assert "sk-test-not-real" not in repr(settings)


def test_min_group_n_cannot_be_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LEDGERLENS_MIN_GROUP_N", "0")
    with pytest.raises(ValueError, match="min_group_n"):
        Settings()
