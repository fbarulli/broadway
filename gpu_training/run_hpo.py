"""Read the measured baseline and fan out HPO workers without OOM.

The baseline says which micro-batch fits and how many trials the host can run.
This launcher splits the study's trial budget across that many workers, assigns
each a GPU/MPS slot, and renders the command from config. Dry-run is the default
posture; ``--execute`` starts every worker at once, then collects exit codes.

Before either mode the launcher runs the real-data preflight from the study
config: parses the project dataset/experiment configs, then proves the declared
parquet exists with its declared columns. A missing dataset fails loud — the
lane never runs on absent data.

The storage URL is a secret: it is redacted in everything printed, and only the
child argv carries the real value.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import yaml
from hpo_study import DatasetIdentity, HpoStudyConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator

REDACTED = "<redacted>"  # what every printed command shows in place of the storage URL
# argv token for "no storage URL": the worker then uses its config's local study.
ABSENT_STORAGE = '""'


class HpoRunConfig(BaseModel):
    """Boundary-validated fan-out profile (``gpu_training/config/hpo_run*.yaml``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: str
    # The study definition (search space, model, dataset, storage policy).
    kernel_config: str
    seed: int
    objective: str
    storage_env: str
    gpus: list[int] = Field(min_length=1)
    mps_slots_per_gpu: int = Field(ge=1)
    workers: int | None = None
    # Optional operator override; null means "the HPO spec's budget".
    n_trials: int | None = Field(default=None, gt=0)
    # False only for a host-local profile whose workers share the kernel's
    # local SQLite study (no shared URL exists) — the 2xT4 Kaggle case.
    storage_required: bool = True

    @field_validator("objective")
    @classmethod
    def _reference(cls, value: str) -> str:
        module, separator, attribute = value.partition(":")
        if not separator or not module or not attribute:
            raise ValueError(f"objective must be a 'module:function' reference, got {value!r}")
        return value

    @classmethod
    def from_yaml(cls, path: Path) -> HpoRunConfig:
        return cls(**_layered(path))


def _mapping(path: Path) -> dict[str, object]:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"run config must be a mapping: {path}")
    return data


def _layered(path: Path) -> dict[str, object]:
    """Merge ``extends`` layers: the base first, this file's keys on top.

    A hardware profile declares only what the hardware changes — the base stays
    the single declaration of the command, seed, objective and storage policy.
    A layer's relative ``kernel_config`` resolves against the file that declares
    it, so a profile in another directory inherits a working base.
    """
    data = _mapping(path)
    kernel_config = data.get("kernel_config")
    if isinstance(kernel_config, str) and not Path(kernel_config).is_absolute():
        data["kernel_config"] = str((path.parent / kernel_config).resolve())
    base = data.pop("extends", None)
    if base is None:
        return data
    if not isinstance(base, str):
        raise ValueError(f"extends must be a config path, got {base!r}")
    return {**_layered(path.parent / base), **data}


@dataclass(frozen=True)
class WorkerPlan:
    """One worker's assigned slot and trial budget.

    ``command`` is the exact argv handed to the child (it carries the storage
    URL); ``display`` is the same command with that URL redacted and is the
    only render that may be printed.
    """

    worker: int
    gpu: int
    n_trials: int
    command: str
    display: str


def _workers(config: HpoRunConfig, baseline: dict[str, object]) -> int:
    per_gpu = baseline.get("concurrent_trials_per_gpu")
    device_count = baseline.get("device_count")
    measured: object
    if isinstance(per_gpu, (int, float)) and isinstance(device_count, (int, float)):
        measured = int(per_gpu) * int(device_count)
    else:
        measured = baseline.get("concurrent_trials")
    ceiling = len(config.gpus) * config.mps_slots_per_gpu
    chosen = config.workers or (int(measured) if isinstance(measured, (int, float)) else 1)
    return max(1, min(chosen, ceiling))


def _distribute(n_trials: int, workers: int) -> list[int]:
    """Split the budget exactly: floor each worker, hand the remainder to the first ones."""
    base, remainder = divmod(n_trials, workers)
    return [base + (1 if worker < remainder else 0) for worker in range(workers)]


def _plan(config: HpoRunConfig, baseline: dict[str, object], budget: int) -> list[WorkerPlan]:
    workers = min(_workers(config, baseline), budget)
    micro_batch = baseline.get("max_batch")
    if micro_batch is None:
        raise ValueError("baseline has no fitting batch: every configured micro-batch failed")
    storage = os.environ.get(config.storage_env)
    plans: list[WorkerPlan] = []
    for worker, n_trials in enumerate(_distribute(budget, workers)):
        gpu = config.gpus[worker % len(config.gpus)]
        values = {
            "kernel_config": config.kernel_config,
            "n_trials": n_trials,
            "micro_batch": micro_batch,
            "worker": worker,
            "gpu": gpu,
            "seed": config.seed,
            "objective": config.objective,
            "storage_env": config.storage_env,
        }
        plans.append(
            WorkerPlan(
                worker=worker,
                gpu=gpu,
                n_trials=n_trials,
                command=config.command.format(
                    storage=storage or ABSENT_STORAGE, **values
                ),
                display=config.command.format(
                    storage=REDACTED if storage else ABSENT_STORAGE, **values
                ),
            )
        )
    return plans


def _start(plan: WorkerPlan) -> subprocess.Popen[bytes]:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(plan.gpu),
        "OMP_NUM_THREADS": "1",
    }
    print(f"[worker {plan.worker}] gpu={plan.gpu} trials={plan.n_trials}: {plan.display}", flush=True)
    return subprocess.Popen(shlex.split(plan.command), env=environment)


def _execute(plans: list[WorkerPlan], config: HpoRunConfig) -> list[int]:
    """Start every worker before waiting on any so the fan-out is concurrent."""
    if config.storage_env not in os.environ:
        if config.storage_required:
            raise ValueError(
                f"{config.storage_env} is unset: this profile requires the shared study URL "
                "from the secret store (set storage_required: false for a host-local study)"
            )
        print(
            f"{config.storage_env} is unset: workers share the kernel's local SQLite study",
            file=sys.stderr,
        )
    processes: list[subprocess.Popen[bytes]] = []
    try:
        for plan in plans:
            processes.append(_start(plan))
    except Exception:  # never leave earlier workers orphaned on a failed start
        for process in processes:
            process.terminate()
        raise
    return [process.wait() for process in processes]


def _report(identity: DatasetIdentity, study: HpoStudyConfig, budget: int, plans: list[WorkerPlan]) -> None:
    print(
        f"dataset '{identity.name}': {identity.path} ({identity.rows} rows, "
        f"contract {identity.schema_contract})",
        file=sys.stderr,
    )
    print(
        f"study hpo-{study.model}: {budget} trials from {study.hpo_spec} → "
        f"{len(plans)} worker(s), {sum(plan.n_trials for plan in plans)} trials",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fan out HPO workers from a memory baseline.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/hpo_run.yaml"))
    parser.add_argument("--execute", action="store_true", help="start the workers")
    parser.add_argument("--dry-run", action="store_true", help="print the plans only (default)")
    arguments = parser.parse_args(argv)
    if arguments.execute and arguments.dry_run:
        parser.error("--execute and --dry-run are mutually exclusive")

    try:
        baseline = json.loads(arguments.baseline.read_text(encoding="utf-8"))
        config = HpoRunConfig.from_yaml(arguments.config)
        study = HpoStudyConfig.from_yaml(Path(config.kernel_config))
        identity = study.identity()  # preflight: configs parse, declared data present
        budget = config.n_trials or study.budget()
        plans = _plan(config, baseline, budget)
        _report(identity, study, budget, plans)
        if not arguments.execute:
            for plan in plans:
                print(plan.display)
            return 0
        codes = _execute(plans, config)
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return 1

    failed = [code for code in codes if code != 0]
    if failed:
        print(f"FAILED workers: {len(failed)}/{len(codes)} (codes {failed})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
