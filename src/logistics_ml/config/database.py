from __future__ import annotations

from dataclasses import dataclass

from broadway.config.schema import DatabaseServiceConfig

from ._config import _config as _cfg


@dataclass(frozen=True)
class DatabaseConfig:
    url: str


def _build(cfg: DatabaseServiceConfig) -> DatabaseConfig:
    return DatabaseConfig(url=cfg.url)


database: DatabaseConfig = _build(_cfg.database_service)
