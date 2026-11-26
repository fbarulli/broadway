"""Public behavior of the gpu_training memory probe, HPO fan-out and worker kernel."""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
import time
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pydantic import ValidationError

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str) -> types.ModuleType:
    """Load a gpu_training module by path (the lane is not an installed package)."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# lane_contract first: the entries import it by its top-level name.
lane_contract = _load("lane_contract", "gpu_training/lane_contract.py")
hpo_study = _load("hpo_study", "gpu_training/hpo_study.py")
memory_probe = _load("gpu_training.memory_probe", "gpu_training/memory_probe.py")
run_hpo = _load("gpu_training.run_hpo", "gpu_training/run_hpo.py")
hpo_kernel = _load("gpu_training.hpo_kernel", "gpu_training/hpo_kernel.py")

STUDY_CONFIG = _ROOT / "gpu_training/config/hpo_kernel.yaml"
KAGGLE_PROFILE = _ROOT / "gpu_training/config/hpo_run_kaggle.yaml"


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


def test_probe_rejects_an_unknown_device(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        memory_probe.ProbeConfig(
            step="tests.test_gpu_training:_fake_step",
            device="tpu",
            dtype="fp32",
            seq_len=8,
            batches=[1],
            headroom=0.9,
            output=str(tmp_path / "baseline.json"),
        )


def test_probe_reports_no_fitting_batch_when_every_step_exceeds_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 2 MiB total × 0.9 headroom is below any process's cumulative RSS.
    monkeypatch.setattr(memory_probe, "_cpu_snapshot", lambda: ("cpu", 1.0, 2.0))
    baseline = memory_probe.MemoryProbe(_probe_config(tmp_path), _fake_step).run()
    assert baseline.max_batch is None
    assert baseline.per_trial_mb is None


def _run_config(**overrides: object) -> run_hpo.HpoRunConfig:
    values: dict[str, object] = {
        "command": (
            "run --n-trials {n_trials} --micro-batch {micro_batch} --worker {worker} "
            "--gpu {gpu} --seed {seed} --objective {objective}"
        ),
        "kernel_config": str(STUDY_CONFIG),
        "seed": 1729,
        "objective": "adapters.taxi:build",
        "storage_env": "OPTUNA_STORAGE_URL",
        "gpus": [0],
        "mps_slots_per_gpu": 2,
        "workers": None,
    }
    values.update(overrides)
    return run_hpo.HpoRunConfig(**values)


def test_plan_splits_the_trial_budget_over_measured_concurrency() -> None:
    baseline = {"max_batch": 8, "concurrent_trials": 2}
    plans = run_hpo._plan(_run_config(), baseline, 5)
    assert len(plans) == 2
    assert [plan.n_trials for plan in plans] == [3, 2]
    assert sum(plan.n_trials for plan in plans) == 5
    assert all("--micro-batch 8" in plan.command for plan in plans)
    assert all("--seed 1729" in plan.command for plan in plans)
    assert [plan.gpu for plan in plans] == [0, 0]


def test_plan_multiplies_per_gpu_concurrency_by_device_count() -> None:
    baseline = {
        "max_batch": 8,
        "concurrent_trials": 1,
        "concurrent_trials_per_gpu": 1,
        "device_count": 2,
    }
    plans = run_hpo._plan(_run_config(gpus=[0, 1], mps_slots_per_gpu=1), baseline, 2)
    assert len(plans) == 2
    assert [plan.gpu for plan in plans] == [0, 1]


def test_plan_requires_a_fitting_batch() -> None:
    with pytest.raises(ValueError, match="no fitting batch"):
        run_hpo._plan(_run_config(), {"max_batch": None, "concurrent_trials": 1}, 5)


def _taxi_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Tmp parquet + config overlay for the real taxi dataset/experiment contracts.

    The declared data path is overlaid, exactly as a Kaggle mount would be; the
    contracts themselves are the committed project ones (no hand-written columns).
    """
    parquet = tmp_path / "training_data.parquet"
    pq.write_table(
        pa.table(
            {
                "dropoff_location_id": pa.array([1, 2, 3], type=pa.int32()),
                "fare_amount": pa.array([10.0, 20.0, 30.0], type=pa.float64()),
                "passenger_count": pa.array([1.0, 2.0, 1.0], type=pa.float64()),
                "pickup_datetime": pa.array(
                    [0, 86_400, 172_800], type=pa.timestamp("ns")
                ),
                "pickup_location_id": pa.array([1, 2, 3], type=pa.int32()),
                "trip_distance": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
                "trip_duration_minutes": pa.array([5.0, 10.0, 15.0], type=pa.float64()),
            }
        ),
        parquet,
    )
    overlay = tmp_path / "overlay"
    for kind in ("dataset", "experiment"):
        source = yaml.safe_load(
            (_ROOT / "project/config" / kind / "taxi.yaml").read_text(encoding="utf-8")
        )
        if kind == "dataset":
            source["path"] = str(parquet)
        target = overlay / kind / "taxi.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(source), encoding="utf-8")
    monkeypatch.setenv("BROADWAY_CONFIG_OVERLAY_DIR", str(overlay))
    return parquet


def _study_config(tmp_path: Path) -> Path:
    """The shipped study definition, with its local study inside tmp_path."""
    source = yaml.safe_load(STUDY_CONFIG.read_text(encoding="utf-8"))
    source["local_storage_url"] = f"sqlite:///{tmp_path / 'study.db'}"
    path = tmp_path / "hpo_kernel.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    return path


def _fake_objective_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Hermetic adapter: `build(study) -> objective`, no data, no torch."""
    module = tmp_path / "fake_objective.py"
    module.write_text(
        "def build(study):\n"
        "    return lambda params, trial=None: float(sum(params.values()))\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return "fake_objective:build"


def _write_run_inputs(
    tmp_path: Path, command: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    _taxi_fixture(tmp_path, monkeypatch)
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"max_batch": 8, "concurrent_trials": 2}', encoding="utf-8")
    config_path = tmp_path / "hpo_run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "command": command,
                "kernel_config": str(STUDY_CONFIG),
                "seed": 1729,
                "objective": "adapters.taxi:build",
                "storage_env": "OPTUNA_STORAGE_URL",
                "gpus": [0],
                "mps_slots_per_gpu": 2,
                "workers": 2,
                "n_trials": 4,
            }
        ),
        encoding="utf-8",
    )
    return baseline, config_path


def test_dry_run_plan_redacts_the_storage_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "postgresql://optuna:TOPSECRET123@db/study"
    monkeypatch.setenv("OPTUNA_STORAGE_URL", secret)
    baseline, config_path = _write_run_inputs(
        tmp_path, "run --storage {storage} --worker {worker}", monkeypatch
    )
    assert run_hpo.main(["--baseline", str(baseline), "--config", str(config_path)]) == 0
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert "<redacted>" in captured.out


def test_execute_starts_every_worker_before_collecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two 0.4s children: parallel ≈ 0.45s, serial ≈ 0.9s. 0.75s splits them with
    # margin both ways; only a heavily loaded host could flake here.
    monkeypatch.setenv("OPTUNA_STORAGE_URL", "secret")
    baseline, config_path = _write_run_inputs(
        tmp_path,
        f'{sys.executable} -c "import time; time.sleep(0.4)" --worker {{worker}}',
        monkeypatch,
    )
    started = time.perf_counter()
    code = run_hpo.main(["--baseline", str(baseline), "--config", str(config_path), "--execute"])
    elapsed = time.perf_counter() - started
    assert code == 0
    assert elapsed < 0.75


def test_kaggle_profile_plans_one_worker_per_t4_and_redacts_the_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _taxi_fixture(tmp_path, monkeypatch)
    secret = "postgresql://optuna:TOPSECRET123@db/study"
    monkeypatch.setenv("OPTUNA_STORAGE_URL", secret)
    baseline = tmp_path / "baseline.json"
    measured = {
        "max_batch": 16,
        "concurrent_trials": 2,
        "concurrent_trials_per_gpu": 1,
        "device_count": 2,
    }
    baseline.write_text(json.dumps(measured), encoding="utf-8")

    assert run_hpo.main(["--baseline", str(baseline), "--config", str(KAGGLE_PROFILE)]) == 0
    captured = capsys.readouterr()
    commands = [line for line in captured.out.splitlines() if "hpo_kernel.py" in line]
    assert len(commands) == 2
    assert secret not in captured.out + captured.err
    assert all("<redacted>" in line for line in commands)

    config = run_hpo.HpoRunConfig.from_yaml(KAGGLE_PROFILE)
    budget = hpo_study.HpoStudyConfig.from_yaml(STUDY_CONFIG).budget()
    plans = run_hpo._plan(config, measured, budget)
    assert [plan.gpu for plan in plans] == [0, 1]
    assert sum(plan.n_trials for plan in plans) == budget
    assert all(plan.n_trials > 0 for plan in plans)


def test_kernel_boundary_accepts_the_launcher_rendered_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "postgresql://optuna:TOPSECRET123@db/study"
    monkeypatch.setenv("OPTUNA_STORAGE_URL", secret)
    measured = {
        "max_batch": 16,
        "concurrent_trials": 2,
        "concurrent_trials_per_gpu": 1,
        "device_count": 2,
    }
    config = run_hpo.HpoRunConfig.from_yaml(KAGGLE_PROFILE)
    budget = hpo_study.HpoStudyConfig.from_yaml(STUDY_CONFIG).budget()
    plan = run_hpo._plan(config, measured, budget)[0]

    interpreter, script, *argv = shlex.split(plan.command)
    assert interpreter.endswith("python")
    assert script.endswith("hpo_kernel.py")
    invocation = hpo_kernel.parse_argv(argv)
    assert invocation.n_trials == plan.n_trials
    assert invocation.micro_batch == 16
    assert invocation.worker == 0
    assert invocation.seed == config.seed
    assert invocation.objective == config.objective
    assert invocation.storage == secret
    # The study identity (model, space, dataset) travels through --config.
    assert hpo_study.HpoStudyConfig.from_yaml(invocation.config) == hpo_study.HpoStudyConfig.from_yaml(
        STUDY_CONFIG
    )


def test_per_worker_seeds_never_share_a_cold_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    study = hpo_study.HpoStudyConfig.from_yaml(STUDY_CONFIG)
    spec = study.model_spec()

    def objective(params: dict[str, float | int], trial: object | None = None) -> float:
        _ = trial
        return float(sum(params.values()))

    def cold_start(worker: int) -> list[dict[str, float | int]]:
        invocation = hpo_kernel.parse_argv(
            [
                "--config", str(STUDY_CONFIG),
                "--n-trials", "4",
                "--micro-batch", "8",
                "--worker", str(worker),
                "--seed", "1729",
                "--objective", "adapters.taxi:build",
            ]
        )  # fmt: skip
        run = hpo_kernel.run_model_study(
            spec, objective, 4, invocation.sampler_seed(), storage_url=None
        )
        return [trial.params for trial in run.trials]

    # seed + worker: two cold workers diverge; one seed would duplicate (control).
    assert cold_start(0) != cold_start(1)
    assert cold_start(0) == cold_start(0)

    # ...and the kernel routes that derived seed into the canonical TPE constructor.
    captured: list[int | None] = []
    real_sampler = hpo_kernel.optuna.samplers.TPESampler

    def recording_sampler(**kwargs: object) -> object:
        captured.append(kwargs.get("seed"))  # type: ignore[arg-type]
        return real_sampler(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(hpo_kernel.optuna.samplers, "TPESampler", recording_sampler)
    monkeypatch.delenv("OPTUNA_STORAGE_URL", raising=False)
    _taxi_fixture(tmp_path, monkeypatch)
    code = hpo_kernel.main(
        [
            "--config", str(_study_config(tmp_path)),
            "--n-trials", "2",
            "--micro-batch", "8",
            "--worker", "1",
            "--seed", "1729",
            "--objective", _fake_objective_module(tmp_path, monkeypatch),
        ]
    )  # fmt: skip
    assert code == 0
    assert captured == [1730]
    record = json.loads(capsys.readouterr().out)
    assert record["study"] == f"hpo-{study.model}"
    assert record["dataset"]["rows"] == 3
    assert record["storage"] == "config"  # local SQLite fallback, no URL printed


def test_preflight_requires_the_declared_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    parquet = _taxi_fixture(tmp_path, monkeypatch)
    study = hpo_study.HpoStudyConfig.from_yaml(STUDY_CONFIG)
    identity = study.identity()
    assert identity.path == str(parquet)
    assert identity.rows == 3
    assert identity.schema_contract == "engineered"

    baseline, config_path = _write_run_inputs(tmp_path, "run --worker {worker}", monkeypatch)
    parquet.unlink()
    assert run_hpo.main(["--baseline", str(baseline), "--config", str(config_path)]) == 1
    captured = capsys.readouterr()
    assert "FileNotFoundError" in captured.err
    assert "declares" in captured.err
