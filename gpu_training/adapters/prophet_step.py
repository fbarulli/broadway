"""Prophet step adapter for the memory probe (CPU only).

Implements the probe contract::

    step(batch_size, seq_len, device, dtype) -> None

Prophet has no mini-batch and no torch parameters: one ``fit`` over a series of
``seq_len`` points IS the training step, and the probe's CPU path records the
process RSS. Asking for a CUDA device fails loud — Prophet is never a VRAM
probe candidate. The package imports lazily and absence fails loud with the
install hint.
"""

from __future__ import annotations

_HINT = "the Prophet probe adapter needs prophet: pip install prophet"


def step(batch_size: int, seq_len: int, device: str, dtype: str) -> None:
    """Fit Prophet on one synthetic series; return None (no torch parameter count)."""
    _ = (batch_size, dtype)
    if device != "cpu":
        raise ValueError("prophet is CPU-only; run the probe with --device cpu")
    try:
        import numpy as np
        import pandas as pd
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError(_HINT) from exc
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01", periods=seq_len, freq="D"),
            "y": np.random.default_rng(0).normal(size=seq_len),
        }
    )
    Prophet().fit(frame)
