"""Runtime settings: config/settings.toml defaults, overridden by LEDGERLENS_* environment variables.

Secrets (ANTHROPIC_API_KEY) come only from the environment and are held as SecretStr.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_TOML = REPO_ROOT / "config" / "settings.toml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEDGERLENS_", toml_file=SETTINGS_TOML, extra="ignore")

    db: Path = Path("data/warehouse.duckdb")
    dbt_project_dir: Path = Path("transform")
    config_dir: Path = Path("config")
    seed: int = 42
    min_group_n: int = Field(default=5, ge=1)
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "LEDGERLENS_ANTHROPIC_API_KEY")
    )

    @field_validator("db", "dbt_project_dir", "config_dir")
    @classmethod
    def _resolve_against_repo(cls, value: Path) -> Path:
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence: explicit kwargs > environment > settings.toml > field defaults.
        return init_settings, env_settings, TomlConfigSettingsSource(settings_cls)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
