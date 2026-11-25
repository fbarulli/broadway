"""Read the measured baseline and fan out HPO workers without OOM.

The baseline says which micro-batch fits and how many trials the host can run.
This launcher splits the trial budget across that many workers, assigns each a
GPU/MPS slot, and renders the command from config. Dry-run is the default
posture; ``--execute`` starts the workers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


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
    """One worker's assigned slot and trial budget."""

    worker: int
    gpu: int
    n_trials: int
    command: str


def _workers(config: HpoRunConfig, baseline: dict[str, object]) -> int:
    measured = baseline.get("concurrent_trials")
    ceiling = len(config.gpus) * config.mps_slots_per_gpu
    chosen = config.workers or (int(measured) if measured else 1)
    return max(1, min(chosen, ceiling))


def _plan(config: HpoRunConfig, baseline: dict[str, object]) -> list[WorkerPlan]:
    workers = _workers(config, baseline)
    per_worker = math.ceil(config.n_trials / workers)
    micro_batch = baseline.get("max_batch")
    if micro_batch is None:
        raise ValueError("baseline has no fitting batch: every configured micro-batch failed")
    storage = os.environ.get(config.storage_env, "<unset>")
    plans: list[WorkerPlan] = []
    for worker in range(workers):
        gpu = config.gpus[worker % len(config.gpus)]
        command = config.command.format(
            study=config.study,
            storage=storage,
            n_trials=per_worker,
            micro_batch=micro_batch,
            worker=worker,
            gpu=gpu,
        )
        plans.append(WorkerPlan(worker=worker, gpu=gpu, n_trials=per_worker, command=command))
    return plans


def _execute(plan: WorkerPlan, storage_env: str) -> int:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(plan.gpu),
        "OMP_NUM_THREADS": "1",
    }
    if storage_env not in os.environ:
        raise ValueError(
            f"{storage_env} is unset: the shared study URL must come from the secret store"
        )
    print(f"[worker {plan.worker}] gpu={plan.gpu} trials={plan.n_trials}: {plan.command}", flush=True)
    return subprocess.run(shlex.split(plan.command), env=environment, check=False).returncode


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
            print(plan.command)
        return 0

    codes = [_execute(plan, config.storage_env) for plan in plans]
    failed = [code for code in codes if code != 0]
    if failed:
        print(f"FAILED workers: {len(failed)}/{len(codes)} (codes {failed})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
