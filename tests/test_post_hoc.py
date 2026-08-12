from __future__ import annotations

import numpy as np
import pandas as pd

from broadway.stats.post_hoc import games_howell


def _df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frames = [
        pd.DataFrame({"group": "A", "dv": rng.normal(0.0, 1.0, 20)}),
        pd.DataFrame({"group": "B", "dv": rng.normal(1.0, 1.0, 20)}),
        pd.DataFrame({"group": "C", "dv": rng.normal(2.0, 1.0, 20)}),
    ]
    return pd.concat(frames, ignore_index=True)


def test_games_howell_columns() -> None:
    result = games_howell(_df(), dv="dv", between="group")
    for col in ["A", "B", "diff", "pval", "cohens_d", "hedges_g", "effect_size_note"]:
        assert col in result.columns
    assert len(result) == 3


def test_games_howell_effect_size_note_small_n() -> None:
    result = games_howell(_df(), dv="dv", between="group")
    assert set(result["effect_size_note"]) == {"use hedges_g"}


def test_games_howell_effect_size_note_large_n() -> None:
    rng = np.random.default_rng(3)
    frames = [
        pd.DataFrame({"group": "X", "dv": rng.normal(0.0, 1.0, 40)}),
        pd.DataFrame({"group": "Y", "dv": rng.normal(0.5, 1.0, 40)}),
    ]
    result = games_howell(pd.concat(frames, ignore_index=True), dv="dv", between="group")
    assert set(result["effect_size_note"]) == {"use cohens_d"}
