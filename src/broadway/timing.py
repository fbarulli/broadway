"""Central wall-clock profiling for training and experiment runs.

One home for named phase timings so steps (and the NLP benchmark) report where
time goes from a single API instead of scattering time.perf_counter() calls
across modules. The report is plain data — safe to dump to CSV/JSON or log as
MLflow metrics.
"""

from __future__ import annotations

import time
from contextlib import contextmanager


class TimingReport:
    """Accumulates named wall-clock durations (sum + call count + last).

    ``record(name)`` is a context manager that times the enclosing with-block;
    repeated entries under the same name accumulate. ``elapsed(name)`` reads
    the running total, and ``as_dict()`` produces a JSON-safe snapshot.
    """

    def __init__(self) -> None:
        self._sum: dict[str, float] = {}
        self._count: dict[str, int] = {}
        self._last: dict[str, float] = {}

    def add(self, name: str, seconds: float) -> None:
        self._sum[name] = self._sum.get(name, 0.0) + seconds
        self._count[name] = self._count.get(name, 0) + 1
        self._last[name] = seconds

    @contextmanager
    def record(self, name: str):
        """Time the enclosing with-block and add its duration under ``name``."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def elapsed(self, name: str) -> float:
        """Total recorded seconds for ``name`` (0.0 if never recorded)."""
        return self._sum.get(name, 0.0)

    def as_dict(self) -> dict[str, dict[str, float]]:
        """Plain-data snapshot: {name: {seconds, calls, last}} sorted by name."""
        return {
            name: {
                "seconds": round(self._sum[name], 3),
                "calls": self._count[name],
                "last": round(self._last[name], 3),
            }
            for name in sorted(self._sum)
        }
