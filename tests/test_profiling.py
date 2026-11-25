"""Public behavior of the profiling seam (PROFILING.md §1-§4)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from broadway.observability import profiling
from broadway.observability.profiling import ProfileConfig, ProfileRecord, StepProfiler


def _config(tmp_path: Path, enabled: bool) -> ProfileConfig:
    return ProfileConfig(
        enabled=enabled,
        output_dir=str(tmp_path / "profiles"),
        runs=1,
        noise_band_pct=5.0,
        budgets={},
    )


def test_enabled_profiler_writes_a_record_with_the_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(profiling, "current_run_id", lambda: "run-id-probe")
    output = tmp_path / "profiles" / "etl.json"

    StepProfiler(_config(tmp_path, enabled=True)).run(
        "etl", "taxi", lambda: sum(range(1000))
    )

    record = ProfileRecord.model_validate_json(output.read_text(encoding="utf-8"))
    assert record.step == "etl"
    assert record.run_id == "run-id-probe"
    assert record.dataset == "taxi"
    assert record.runs == 1
    assert record.wall_p50 > 0.0
    assert record.wall_p95 == record.wall_p50
    assert record.cpu_seconds > 0.0
    assert record.peak_rss_mb > 0.0
    assert record.rows is None
    assert record.throughput_per_1k is None
    assert datetime.fromisoformat(record.created_at).tzinfo is not None


def test_disabled_profiler_runs_the_callable_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(profiling.PROFILE_ENV, raising=False)
    calls: list[int] = []

    StepProfiler(_config(tmp_path, enabled=False)).run(
        "etl", "taxi", lambda: calls.append(1)
    )

    assert calls == [1]
    assert not (tmp_path / "profiles").exists()
