"""Typed container returned by the named-sample loader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from broadway.lineage.models import SampleSpec


@dataclass
class Sample:
    df: pd.DataFrame
    spec: SampleSpec
    provenance: dict[str, Any]
