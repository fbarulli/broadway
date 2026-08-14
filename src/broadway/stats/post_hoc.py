"""Games-Howell pairwise with Cohen's d / Hedges' g effect sizes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pingouin as pg

from broadway.stats.effect_size import cohens_d, hedges_g


def games_howell(df: pd.DataFrame, dv: str, between: str, small_n: int = 30) -> pd.DataFrame:
    result = pg.pairwise_gameshowell(data=df, dv=dv, between=between)

    grouped = {name: group[dv].to_numpy() for name, group in df.groupby(between)}

    cohens: list[float] = []
    hedges: list[float] = []
    notes: list[str] = []
    for a, b in zip(result["A"], result["B"]):
        a_vals = grouped[a]
        b_vals = grouped[b]
        cohens.append(cohens_d(a_vals, b_vals))
        hedges.append(hedges_g(a_vals, b_vals))
        if len(a_vals) < small_n or len(b_vals) < small_n:
            notes.append("use hedges_g")
        else:
            notes.append("use cohens_d")

    result["cohens_d"] = cohens
    result["hedges_g"] = hedges
    result["effect_size_note"] = notes
    return result
