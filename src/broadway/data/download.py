"""Fetch raw files from URLs into the data directory."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

from broadway.config.schema import EnvironmentConfig

logger = logging.getLogger(__name__)


def download(url: str, env: EnvironmentConfig) -> Path:
    filename = Path(unquote(urlparse(url).path)).name or "download"
    dest_dir = Path(env.data_dir) / env.raw_subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    logger.info(f"downloading {url} → {dest}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, "wb") as f:
        f.writelines(response.iter_content(chunk_size=env.download_chunk_size))
    return dest
