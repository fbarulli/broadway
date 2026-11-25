"""Read the measured baseline and fan out HPO workers without OOM.

The baseline says which micro-batch fits and how many trials the host can run.
This launcher splits the trial budget across that many workers, assigns each a
GPU/MPS slot, and renders the command from config. Dry-run is the default
posture; ``--execute`` starts every worker at once, then collects exit codes.
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
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

REDACTED = "<redacted>"  # what every printed command shows in place of the storage URL


class HpoRunConfig(BaseModel):
    """Boundary-validated fan-out config (``gpu_training/config/hpo_run.yaml``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: str
    n_trials: int = Field(gt=0)
    study: str
    storage_env: str
    gpus: list[int] = Field(min_length=1)
    mps_slots_per_gpu: int = Field(ge=1)
    workers: int | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> HpoRunConfig:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"run config must be a mapping: {path}")
        return cls(**data)


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


def _plan(config: HpoRunConfig, baseline: dict[str, object]) -> list[WorkerPlan]:
    workers = min(_workers(config, baseline), config.n_trials)
    micro_batch = baseline.get("max_batch")
    if micro_batch is None:
        raise ValueError("baseline has no fitting batch: every configured micro-batch failed")
    storage = os.environ.get(config.storage_env, "<unset>")
    plans: list[WorkerPlan] = []
    for worker, n_trials in enumerate(_distribute(config.n_trials, workers)):
        gpu = config.gpus[worker % len(config.gpus)]
        values = {
            "study": config.study,
            "n_trials": n_trials,
            "micro_batch": micro_batch,
            "worker": worker,
            "gpu": gpu,
            "storage_env": config.storage_env,
        }
        plans.append(
            WorkerPlan(
                worker=worker,
                gpu=gpu,
                n_trials=n_trials,
                command=config.command.format(storage=storage, **values),
                display=config.command.format(storage=REDACTED, **values),
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


def _execute(plans: list[WorkerPlan], storage_env: str) -> list[int]:
    """Start every worker before waiting on any so the fan-out is concurrent."""
    if storage_env not in os.environ:
        raise ValueError(
            f"{storage_env} is unset: the shared study URL must come from the secret store"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fan out HPO workers from a memory baseline.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/hpo_run.yaml"))
    parser.add_argument("--execute", action="store_true", help="start the workers")
    parser.add_argument("--dry-run", action="store_true", help="print the plans only (default)")
    arguments = parser.parse_args(argv)
    if arguments.execute and arguments.dry_run:
        parser.error("--execute and --dry-run are mutually exclusive")

    baseline = json.loads(arguments.baseline.read_text(encoding="utf-8"))
    config = HpoRunConfig.from_yaml(arguments.config)
    plans = _plan(config, baseline)
    print(
        f"baseline: max_batch={baseline.get('max_batch')} "
        f"concurrent_trials={baseline.get('concurrent_trials')} → {len(plans)} worker(s), "
        f"{sum(p.n_trials for p in plans)} trials",
        file=sys.stderr,
    )
    if not arguments.execute:
        for plan in plans:
            print(plan.display)
        return 0

    codes = _execute(plans, config.storage_env)
    failed = [code for code in codes if code != 0]
    if failed:
        print(f"FAILED workers: {len(failed)}/{len(codes)} (codes {failed})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
