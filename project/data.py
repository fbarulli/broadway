"""Demo dataset loaders (main branch, synthetic demo).

Mirror of the taxi branch's ``project/data.py`` contract (contract-driven
loaders + constants) so the shared worker image layout and CI boot checks
resolve on main. Backed by the synthetic demo dataset; no taxi content.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from broadway.config.schema import DatasetContract
from broadway.contracts.pandera import build_raw_schema

_DATASET_YAML = Path("configs/dataset/test.yaml")
_contract = DatasetContract(**yaml.safe_load(_DATASET_YAML.read_text()))

DATA_PATH = Path(_contract.path)


def read_demo_data() -> pd.DataFrame:
    """Synthetic demo dataset, validated against the demo contract."""
    raw = pd.read_csv(DATA_PATH)
    return build_raw_schema(_contract).validate(raw)


def read_demo_sample(sample: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Seeded random sample of the demo dataset."""
    df = read_demo_data()
    if sample is not None:
        df = df.sample(n=sample, random_state=seed)
    return df
