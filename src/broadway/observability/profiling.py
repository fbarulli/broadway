"""Profiling seam (PROFILING.md §1-§4).

The single place in ``src/`` that times anything. Disabled by default; enabled
through config or ``BROADWAY_PROFILE=1``. The profile gate is planned, not
wired — this module records the machine artifact, it never blocks a run.
"""

from __future__ import annotations

import logging
import os
import resource
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from broadway.config.loader import config_path
from broadway.observability.context import current_run_id
from broadway.paths import repo_root

logger = logging.getLogger(__name__)

PROFILE_CONFIG = "profiling.yaml"
PROFILE_ENV = "BROADWAY_PROFILE"
_KIB_PER_MIB = 1024.0


class ProfileConfig(BaseModel):
    """Boundary-validated shape of ``configs/profiling.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    output_dir: str
    runs: int = Field(ge=1)
    noise_band_pct: float = Field(ge=0.0)
    budgets: dict[str, float]

    @classmethod
    def from_yaml(cls, path: Path) -> ProfileConfig:
        """Load and validate the profiling config at ``path``."""
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"profiling config must be a mapping: {path}")
        return cls(**data)


class ProfileRecord(BaseModel):
    """One measured step invocation (``artifacts/profiles/<step>.json``)."""

    model_config = ConfigDict(extra="forbid")

    step: str
    run_id: str
    wall_p50: float
    wall_p95: float
    cpu_seconds: float
    peak_rss_mb: float
    rows: int | None
    throughput_per_1k: float | None
    runs: int
    dataset: str | None
    created_at: str


def _percentiles(values: list[float]) -> tuple[float, float]:
    """Return (p50, p95); a single run is its own p50 and p95."""
    if len(values) == 1:
        return values[0], values[0]
    cuts = statistics.quantiles(values, n=100, method="inclusive")
    return cuts[49], cuts[94]


def _peak_rss_mb() -> float:
    """Process-wide peak RSS, not this step's: ``ru_maxrss`` is Linux KiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _KIB_PER_MIB


class StepProfiler:
    """Measures a callable and writes/does not write its profile record."""

    def __init__(self, config: ProfileConfig) -> None:
        self._config = config

    @classmethod
    def from_default_config(cls) -> StepProfiler:
        """Build from ``configs/profiling.yaml`` (overlay-aware)."""
        return cls(ProfileConfig.from_yaml(config_path(PROFILE_CONFIG)))

    @property
    def enabled(self) -> bool:
        """Config flag or the one-off env override; never a code edit."""
        return self._config.enabled or os.getenv(PROFILE_ENV) == "1"

    def run(self, step: str, dataset: str | None, fn: Callable[[], None]) -> None:
        """Execute ``fn`` and, when enabled, record and write its profile.

        Disabled means no timing and no file, never a skipped ``fn``: the
        pipeline seam passes the step's own ``run`` here.
        """
        if not self.enabled:
            fn()
            return
        walls, cpus = self._measure(fn)
        self._write(self._record(step, dataset, walls, cpus))

    def _measure(self, fn: Callable[[], None]) -> tuple[list[float], list[float]]:
        walls: list[float] = []
        cpus: list[float] = []
        for _ in range(self._config.runs):
            cpu_start = time.process_time()
            wall_start = time.perf_counter()
            fn()
            walls.append(time.perf_counter() - wall_start)
            cpus.append(time.process_time() - cpu_start)
        return walls, cpus

    def _record(
        self, step: str, dataset: str | None, walls: list[float], cpus: list[float]
    ) -> ProfileRecord:
        wall_p50, wall_p95 = _percentiles(walls)
        return ProfileRecord(
            step=step,
            run_id=current_run_id(),
            wall_p50=wall_p50,
            wall_p95=wall_p95,
            cpu_seconds=statistics.fmean(cpus),
            peak_rss_mb=_peak_rss_mb(),
            # The step API exposes no row count at the seam: None is honest,
            # a derived number would be invented (PROFILING.md §3, §6).
            rows=None,
            throughput_per_1k=None,
            runs=self._config.runs,
            dataset=dataset,
            created_at=datetime.now(UTC).isoformat(),
        )

    def _write(self, record: ProfileRecord) -> None:
        output_dir = Path(self._config.output_dir)
        if not output_dir.is_absolute():
            output_dir = repo_root() / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{record.step}.json"
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"profile written: {path}")
