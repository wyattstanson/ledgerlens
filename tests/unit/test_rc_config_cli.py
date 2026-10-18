from pathlib import Path

import pytest

from ledgerlens import cli
from ledgerlens.core.db import connect
from ledgerlens.ratecards.config import RateCardsConfig, WorldConfig
from ledgerlens.settings import REPO_ROOT, Settings


def test_code_defaults_match_the_shipped_yaml():
    # Two sources of truth drifted once (line_probability 0.8 vs 0.85); this keeps them equal.
    assert RateCardsConfig.from_yaml(REPO_ROOT / "config" / "ratecards.yaml") == RateCardsConfig()


def test_cli_honours_an_explicit_seed_of_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    settings = Settings(db=tmp_path / "w.duckdb", seed=42)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    small = RateCardsConfig(world=WorldConfig(vendors=8, months=14, duplicate_lines=0, distinct_titles=64))
    monkeypatch.setattr(cli.RateCardsConfig, "from_yaml", classmethod(lambda cls, path: small))
    assert cli.main(["generate", "--seed", "0"]) == 0
    with connect(settings.db) as conn:
        assert conn.execute("select seed from raw_ratecards.world_meta").fetchone() == (0,)
