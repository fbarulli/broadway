from __future__ import annotations

import yaml

from broadway.config import loader
from broadway.lineage.models import SampleSpec


def load_sample(name: str) -> SampleSpec:
    path = loader.CONFIGS_DIR / "sample" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"sample config not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"sample config must be a mapping: {path}")
    return SampleSpec(**data)
