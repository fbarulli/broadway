"""Public behavior of the profiling seam (PROFILING.md §1-§4)."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pytest

from broadway.observability import profiling
from broadway.observability.profiling import (
    ProfileConfig,
    ProfileRecord,
    StepProfiler,
    phase,
    timed,
)


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
        "etl", "amazon", lambda: sum(range(1000))
    )

    record = ProfileRecord.model_validate_json(output.read_text(encoding="utf-8"))
    assert record.step == "etl"
    assert record.run_id == "run-id-probe"
    assert record.dataset == "amazon"
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
        "etl", "amazon", lambda: calls.append(1)
    )

    assert calls == [1]
    assert not (tmp_path / "profiles").exists()


def test_phase_records_only_into_the_active_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(profiling.PROFILE_ENV, raising=False)
    output = tmp_path / "profiles" / "train.json"

    with phase("model_fit"):  # no active profile: inert, writes nothing
        pass
    assert not output.exists()

    def probe() -> None:
        with phase("model_fit"):
            time.sleep(0.01)

    StepProfiler(_config(tmp_path, enabled=True)).run("train", "amazon", probe)

    record = ProfileRecord.model_validate_json(output.read_text(encoding="utf-8"))
    assert record.phases["model_fit"].wall_seconds >= 0.01
    assert record.phases["model_fit"].cpu_seconds >= 0.0


def test_timed_reports_elapsed_wall_seconds() -> None:
    with timed() as elapsed:
        time.sleep(0.02)
    assert elapsed() >= 0.02
