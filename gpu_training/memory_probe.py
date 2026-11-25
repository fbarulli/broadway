"""Measure a training step's memory footprint and write an HPO baseline.

The probe is model-agnostic: ``--step module:function`` must implement

    step(batch_size, seq_len, device, dtype) -> param_count | None

and run one real forward + backward + optimizer step. The probe sweeps the
configured micro-batches, records peak VRAM (CUDA) or peak RSS (CPU), finds the
largest batch under the headroom, and derives how many trials the host can run.

No threshold or path is coded here: everything comes from the config (SSOT).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import resource
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

StepFn = Callable[[int, int, str, str], int | None]
Device = Literal["cuda", "cpu"]
Dtype = Literal["fp32", "fp16", "bf16"]
_MIB = 1_000_000.0


class ProbeConfig(BaseModel):
    """Boundary-validated probe config (``gpu_training/config/memory_probe.yaml``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: str
    device: Device
    dtype: Dtype
    seq_len: int = Field(gt=0)
    batches: list[int] = Field(min_length=1)
    headroom: float = Field(gt=0.0, le=1.0)
    output: str

    @classmethod
    def from_yaml(cls, path: Path) -> ProbeConfig:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"probe config must be a mapping: {path}")
        return cls(**data)


class BatchMeasurement(BaseModel):
    """One micro-batch attempt."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int
    ok: bool
    peak_alloc_mb: float | None = None
    peak_reserved_mb: float | None = None
    free_mb: float | None = None
    total_mb: float | None = None
    peak_rss_mb: float | None = None
    error: str | None = None


class Baseline(BaseModel):
    """What the HPO launcher needs: the fitting batch and the concurrency bound."""

    model_config = ConfigDict(extra="forbid")

    device: Device
    device_name: str
    total_mb: float | None
    free_mb: float | None
    params: int | None
    seq_len: int
    dtype: Dtype
    headroom: float
    batches: list[BatchMeasurement]
    max_batch: int | None
    per_trial_mb: float | None
    concurrent_trials: int | None
    created_at: str


def _meminfo() -> dict[str, float]:
    """Host RAM from /proc/meminfo, in MiB (Linux)."""
    values: dict[str, float] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, _, rest = line.partition(":")
        if key in {"MemTotal", "MemAvailable"} and rest.strip():
            values[key] = float(rest.split()[0]) / 1024.0
    return values


def _cuda_snapshot() -> tuple[str, float, float]:
    import torch

    properties = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    return properties.name, free / _MIB, total / _MIB


def _cpu_snapshot() -> tuple[str, float, float]:
    memory = _meminfo()
    return "cpu", memory.get("MemAvailable", 0.0), memory.get("MemTotal", 0.0)


class MemoryProbe:
    """Runs the configured sweep and writes one baseline record."""

    def __init__(self, config: ProbeConfig, step: StepFn) -> None:
        self._config = config
        self._step = step

    @classmethod
    def from_config(cls, config: ProbeConfig) -> MemoryProbe:
        return cls(config, _resolve_step(config.step))

    def run(self) -> Baseline:
        name, free, total = self._snapshot()
        measurements: list[BatchMeasurement] = []
        params: int | None = None
        oom = False
        for batch in sorted(self._config.batches):
            if oom:
                measurements.append(
                    BatchMeasurement(
                        batch_size=batch,
                        ok=False,
                        error="skipped after OOM (batch is monotonic)",
                    )
                )
                continue
            measurement = self._measure(batch, total)
            measurements.append(measurement)
            if not measurement.ok:
                oom = measurement.error is not None and "out of memory" in measurement.error.lower()
                continue
            if params is None:
                params = self._last_params
        max_batch = self._max_batch(measurements, total)
        per_trial = self._per_trial(measurements, max_batch)
        concurrent = self._concurrency(free, per_trial)
        return Baseline(
            device=self._config.device,
            device_name=name,
            total_mb=total,
            free_mb=free,
            params=params,
            seq_len=self._config.seq_len,
            dtype=self._config.dtype,
            headroom=self._config.headroom,
            batches=measurements,
            max_batch=max_batch,
            per_trial_mb=per_trial,
            concurrent_trials=concurrent,
            created_at=datetime.now(UTC).isoformat(),
        )

    _last_params: int | None = None

    def _snapshot(self) -> tuple[str, float, float]:
        if self._config.device == "cuda":
            return _cuda_snapshot()
        return _cpu_snapshot()

    def _measure(self, batch: int, total: float) -> BatchMeasurement:
        started_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        try:
            params = self._run_step(batch)
        except Exception as error:  # probe must record an OOM and keep going
            return BatchMeasurement(
                batch_size=batch,
                ok=False,
                error=f"{type(error).__name__}: {error}",
                peak_rss_mb=started_rss,
            )
        self._last_params = params
        if self._config.device == "cuda":
            import torch

            return BatchMeasurement(
                batch_size=batch,
                ok=True,
                peak_alloc_mb=torch.cuda.max_memory_allocated() / _MIB,
                peak_reserved_mb=torch.cuda.max_memory_reserved() / _MIB,
                free_mb=torch.cuda.mem_get_info(0)[0] / _MIB,
                total_mb=total,
            )
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        return BatchMeasurement(
            batch_size=batch, ok=True, peak_rss_mb=peak, total_mb=total
        )

    def _run_step(self, batch: int) -> int | None:
        if self._config.device == "cuda":
            import torch

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return self._step(batch, self._config.seq_len, self._config.device, self._config.dtype)

    def _max_batch(self, measurements: list[BatchMeasurement], total: float) -> int | None:
        ok = [m for m in measurements if m.ok]
        if not ok:
            return None
        limit = total * self._config.headroom
        fitting = [
            m
            for m in ok
            if (m.peak_reserved_mb or m.peak_rss_mb or 0.0) <= limit
        ]
        if not fitting:
            return None
        return max(m.batch_size for m in fitting)

    def _per_trial(self, measurements: list[BatchMeasurement], max_batch: int | None) -> float | None:
        if max_batch is None:
            return None
        for measurement in measurements:
            if measurement.batch_size == max_batch and measurement.ok:
                return measurement.peak_reserved_mb or measurement.peak_rss_mb
        return None

    def _concurrency(self, free: float, per_trial: float | None) -> int | None:
        if self._config.device != "cuda" or not per_trial:
            return None
        return max(1, math.floor((free * self._config.headroom) / per_trial))


def _resolve_step(spec: str) -> StepFn:
    module_name, _, function_name = spec.partition(":")
    if not module_name or not function_name:
        raise ValueError(f"step must be 'module:function', got {spec!r}")
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"step {spec!r} is not callable")
    return function


def _write(baseline: Baseline, output: str) -> Path:
    path = Path(output)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(baseline.model_dump_json(indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure a training step's memory footprint.")
    parser.add_argument("--config", type=Path, default=Path("config/memory_probe.yaml"))
    parser.add_argument("--step", default=None, help="override module:function")
    parser.add_argument("--device", default=None, help="override cuda|cpu")
    parser.add_argument("--batches", default=None, help="override comma-separated batches")
    parser.add_argument("--output", default=None, help="override output path")
    arguments = parser.parse_args(argv)

    config = ProbeConfig.from_yaml(arguments.config)
    if arguments.step or arguments.device or arguments.batches or arguments.output:
        updates: dict[str, Any] = {}
        if arguments.step:
            updates["step"] = arguments.step
        if arguments.device:
            updates["device"] = arguments.device
        if arguments.batches:
            updates["batches"] = [int(v) for v in arguments.batches.split(",")]
        if arguments.output:
            updates["output"] = arguments.output
        # model_copy(update=...) does not validate; re-validate at the CLI boundary
        config = ProbeConfig.model_validate({**config.model_dump(), **updates})

    baseline = MemoryProbe.from_config(config).run()
    path = _write(baseline, config.output)
    print(json.dumps(baseline.model_dump(), indent=2))
    print(f"baseline written to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
