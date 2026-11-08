"""Settings parsing and the rule that a database URL is mandatory."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_cors_origins_split_and_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test ,")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.cors_origin_list == ["http://a.test", "http://b.test"]


def test_database_url_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]
