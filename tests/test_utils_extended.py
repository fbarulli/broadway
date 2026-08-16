"""Fail-loud validation helpers in broadway.utils (config keys + NaN/Inf guard)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from broadway.utils import require_finite, require_keys


def test_require_keys_missing_raises_with_key_name() -> None:
    with pytest.raises(ValueError, match="alpha"):
        require_keys({"beta": 1}, ["alpha", "beta"], "train")


def test_require_keys_clean_config_passes() -> None:
    require_keys({"alpha": 1, "beta": 2}, ["alpha", "beta"], "train")


def test_require_finite_nan_raises() -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan]})
    with pytest.raises(ValueError, match="NaN"):
        require_finite(frame, "train")


def test_require_finite_inf_raises() -> None:
    frame = pd.DataFrame({"a": [1.0, np.inf]})
    with pytest.raises(ValueError, match="Inf"):
        require_finite(frame, "train")


def test_require_finite_clean_frame_passes() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    require_finite(frame, "train")
