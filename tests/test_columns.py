from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from broadway.discover.columns import run


def test_columns_prints_name_dtype_and_writes_no_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parquet = tmp_path / "data.parquet"
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "b": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "c": [1, 2],
        }
    )
    df.to_parquet(parquet, index=False)

    before = set(tmp_path.rglob("*"))
    run(str(parquet))
    out = capsys.readouterr().out

    assert "a: float64" in out
    assert "b: datetime64" in out
    assert "c: int64" in out
    assert set(tmp_path.rglob("*")) == before
