"""Public behavior of the gpu_training memory probe and HPO fan-out."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str) -> types.ModuleType:
    """Load a gpu_training module by path (the lane is not an installed package)."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


memory_probe = _load("gpu_training.memory_probe", "gpu_training/memory_probe.py")
run_hpo = _load("gpu_training.run_hpo", "gpu_training/run_hpo.py")


def _fake_step(batch_size: int, seq_len: int, device: str, dtype: str) -> int:
    """Stand-in adapter: no torch, returns a deterministic parameter count."""
    _ = (batch_size, seq_len, device, dtype)
    return 42


def _probe_config(tmp_path: Path) -> memory_probe.ProbeConfig:
    return memory_probe.ProbeConfig(
        step="tests.test_gpu_training:_fake_step",
        device="cpu",
        dtype="fp32",
        seq_len=8,
        batches=[1, 2, 4],
        headroom=0.9,
        output=str(tmp_path / "baseline.json"),
    )


def test_probe_writes_a_cpu_baseline(tmp_path: Path) -> None:
    baseline = memory_probe.MemoryProbe(_probe_config(tmp_path), _fake_step).run()
    assert baseline.max_batch == 4
    assert baseline.params == 42
    assert baseline.concurrent_trials is None
    assert baseline.per_trial_mb is not None


def _run_config(**overrides: object) -> run_hpo.HpoRunConfig:
    values: dict[str, object] = {
        "command": "run --n-trials {n_trials} --micro-batch {micro_batch} --worker {worker} --gpu {gpu}",
        "n_trials": 5,
        "study": "laya",
        "storage_env": "OPTUNA_STORAGE_URL",
        "gpus": [0],
        "mps_slots_per_gpu": 2,
        "workers": None,
    }
    values.update(overrides)
    return run_hpo.HpoRunConfig(**values)


def test_plan_splits_the_trial_budget_over_measured_concurrency() -> None:
    baseline = {"max_batch": 8, "concurrent_trials": 2}
    plans = run_hpo._plan(_run_config(), baseline)
    assert len(plans) == 2
    assert [plan.n_trials for plan in plans] == [3, 3]
    assert all("--micro-batch 8" in plan.command for plan in plans)
    assert [plan.gpu for plan in plans] == [0, 0]


def test_plan_requires_a_fitting_batch() -> None:
    with pytest.raises(ValueError, match="no fitting batch"):
        run_hpo._plan(_run_config(), {"max_batch": None, "concurrent_trials": 1})
