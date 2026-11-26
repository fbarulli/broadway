"""Run ONE Optuna worker over the canonical broadway HPO study.

Model-agnostic: ``--objective module:function`` mirrors the memory probe's
``--step module:function`` boundary, so no vendor or model code lives in the
lane. The reference is a factory called as ``build(study) -> objective`` where
``study`` is the validated study config (project config names, no secrets) and
the returned objective has the canonical broadway signature
``objective(params, trial) -> float``. The kernel then runs the study through
``broadway.training.hpo.run_model_study`` — never a sampler loop of its own.

Per-worker seed: the canonical TPE sampler is seeded ``seed + worker``. Two
workers on one study that inherit a single seed draw the same cold-start
sequence (TPE's startup draws come from one RNG stream and ``constant_liar``
only guards already-observed points), so a 2xT4 run burned half its budget on
duplicate trials. The seed offset is the fix.

Storage: ``--storage`` wins, then the study config's ``storage_env`` variable,
then its local SQLite study — the fallback the 2xT4 run used when no shared URL
existed. The URL is a secret: it is never printed, and a traceback is scrubbed
before it reaches stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import optuna
from hpo_study import DatasetIdentity, HpoStudyConfig, SharedStudyStorage
from lane_contract import resolve_callable
from pydantic import BaseModel, ConfigDict, Field, field_validator

from broadway.training.hpo import Objective, run_model_study

REDACTED = "<redacted>"  # what a scrubbed traceback shows in place of the storage URL


class WorkerInvocation(BaseModel):
    """One worker's argv, validated at the boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: Path
    n_trials: int = Field(gt=0)
    micro_batch: int = Field(gt=0)
    worker: int = Field(ge=0)
    seed: int
    objective: str
    storage: str | None = None

    @field_validator("objective")
    @classmethod
    def _reference(cls, value: str) -> str:
        module, separator, attribute = value.partition(":")
        if not separator or not module or not attribute:
            raise ValueError(f"objective must be a 'module:function' reference, got {value!r}")
        return value

    def sampler_seed(self) -> int:
        """This worker's TPE seed: the run seed offset by the worker index."""
        return self.seed + self.worker


def parse_argv(argv: list[str] | None = None) -> WorkerInvocation:
    parser = argparse.ArgumentParser(description="Run one Optuna HPO worker.")
    parser.add_argument("--config", type=Path, default=Path("config/hpo_kernel.yaml"))
    parser.add_argument("--n-trials", type=int, required=True)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--objective", required=True, help="module:function factory")
    parser.add_argument("--storage", default=None, help="Optuna storage URL (never printed)")
    # The parsed namespace is re-validated by the model above, at the boundary.
    return WorkerInvocation(**vars(parser.parse_args(argv)))


def _storage_target(invocation: WorkerInvocation, study: HpoStudyConfig) -> tuple[str, str]:
    """(URL, source) — argv, then the study config's env var, then the local study."""
    if invocation.storage:
        return invocation.storage, "argv"
    from_env = os.environ.get(study.storage_env)
    if from_env:
        return from_env, study.storage_env
    return study.local_storage_url, "config"


def _redact(trace: str, secret: str) -> str:
    """Scrub the storage URL out of a traceback before it reaches stderr."""
    return trace.replace(secret, REDACTED) if secret else trace


class HpoWorker:
    """One worker: the study's search space, this worker's seed, one storage target."""

    def __init__(
        self,
        invocation: WorkerInvocation,
        study: HpoStudyConfig,
        objective: Objective,
        identity: DatasetIdentity,
        direction: str,
        storage_url: str,
        storage_source: str,
    ) -> None:
        self._invocation = invocation
        self._study = study
        self._objective = objective
        self._identity = identity
        self._direction = direction
        self._storage_url = storage_url
        self._storage_source = storage_source

    @classmethod
    def build(
        cls,
        invocation: WorkerInvocation,
        study: HpoStudyConfig,
        storage_url: str,
        storage_source: str,
    ) -> HpoWorker:
        """Resolve the adapter factory and prove the data BEFORE any trial runs."""
        identity = study.identity()  # cheap: parquet metadata, no data read
        adapter = resolve_callable(invocation.objective, "objective")(study.model_dump())
        if not callable(adapter):
            raise TypeError(f"objective factory {invocation.objective!r} returned {type(adapter)}")
        SharedStudyStorage.prepare(storage_url)
        return cls(
            invocation=invocation,
            study=study,
            objective=_bind_micro_batch(adapter, study.micro_batch_param, invocation.micro_batch),
            identity=identity,
            direction=study.hpo_config().direction,
            storage_url=storage_url,
            storage_source=storage_source,
        )

    def run(self) -> dict[str, Any]:
        invocation = self._invocation
        started = time.perf_counter()
        study = run_model_study(
            self._study.model_spec(),
            self._objective,
            n_trials=invocation.n_trials,
            random_state=invocation.sampler_seed(),
            storage_url=self._storage_url,
            direction=self._direction,
        )
        elapsed_s = time.perf_counter() - started
        return {
            "study": study.study_name,
            "model": self._study.model,
            "worker": invocation.worker,
            "sampler_seed": invocation.sampler_seed(),
            "n_trials": invocation.n_trials,
            "micro_batch": invocation.micro_batch,
            "storage": self._storage_source,
            "dataset": self._identity.model_dump(),
            "best_value": float(study.best_value),
            "best_params": dict(study.best_params),
            "elapsed_s": round(elapsed_s, 3),
        }


def _bind_micro_batch(
    objective: Objective, parameter: str | None, micro_batch: int
) -> Objective:
    """Hand the measured batch to the adapter as a fixed param, under its own name."""
    if parameter is None:
        return objective

    def bound(params: dict[str, float | int], trial: optuna.Trial | None = None) -> float:
        return objective({**params, parameter: micro_batch}, trial)

    return bound


def main(argv: list[str] | None = None) -> int:
    invocation = parse_argv(argv)
    try:
        study = HpoStudyConfig.from_yaml(invocation.config)
    except Exception:  # config holds no secret: the URL arrives later
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    storage_url, storage_source = _storage_target(invocation, study)
    try:
        result = HpoWorker.build(invocation, study, storage_url, storage_source).run()
    except Exception:
        print(_redact(traceback.format_exc(), storage_url), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
