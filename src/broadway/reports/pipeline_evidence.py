"""Pipeline steps as a dashboard series.

Pipeline steps (``etl``, ``eda``, ...) are not script series: they publish their
decision record and evidence under
``<experiments root>/results/pipeline/<step>/``. The dashboard exposes them through
the normal series surface — one ``experiment | question | status | artifacts`` row per
step, opening a step page — instead of a bespoke section. This module owns the data:
the step list, the validated ``Decision`` records, the files a step produced, the
static URLs and the producing module source. Rendering stays in the dashboard.

Every ``*_decision.json`` is validated through the canonical ``Decision`` model; a
malformed record is skipped (logged once per path with its full traceback) instead of
failing the page.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from pydantic import ValidationError

from broadway.config.loader import STEP_MODULES
from broadway.reports.decision import Decision

logger = logging.getLogger(__name__)

#: The series id under which ``results/pipeline`` is served.
PIPELINE_FOCUS = "pipeline"

_RECORD_SUFFIX = "_decision.json"
#: ``src/broadway`` — the owner of every step module named by ``STEP_MODULES``.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# A permanently malformed record is skipped on every request; log its full
# traceback once so the contract's traceback rule does not become log bloat.
_LOGGED_UNREADABLE: set[Path] = set()


@dataclass(frozen=True)
class PipelineRecord:
    """One validated decision record for a pipeline step."""

    step: str
    record: Path
    decision: Decision

    @property
    def stem(self) -> str:
        """The record's filename stem (``evaluate/champion_promotion`` -> ``champion_promotion``)."""
        return self.record.name[: -len(_RECORD_SUFFIX)]

    @property
    def sidecar(self) -> Path:
        return self.record.parent / f"{self.stem}_feature_decisions.json"

    @classmethod
    def from_path(cls, record: Path) -> PipelineRecord | None:
        """Validate one record through ``Decision``, or log why it is skipped."""
        try:
            decision = Decision.model_validate_json(record.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValidationError):
            if record not in _LOGGED_UNREADABLE:
                _LOGGED_UNREADABLE.add(record)
                logger.exception("pipeline evidence: unreadable decision record at %s", record)
            return None
        return cls(step=record.parent.name, record=record, decision=decision)


class PipelineSeries:
    """The ``results/pipeline`` tree read as one dashboard series (one row per step)."""

    def __init__(self, results_root: Path) -> None:
        self._results_root = Path(results_root)
        self._pipeline_root = self._results_root / PIPELINE_FOCUS

    @classmethod
    def from_experiments_root(cls, experiments_root: Path) -> PipelineSeries:
        """Build the series for the dashboard's configured experiments root."""
        return cls(Path(experiments_root) / "results")

    @property
    def exists(self) -> bool:
        """Whether the checkout published any pipeline step."""
        return self._pipeline_root.is_dir()

    def steps(self) -> list[str]:
        """Step ids: the directories under ``results/pipeline``, sorted."""
        if not self.exists:
            return []
        return sorted(path.name for path in self._pipeline_root.iterdir() if path.is_dir())

    def step_dir(self, step: str) -> Path | None:
        """The step's directory, or None when the step is unknown."""
        return self._pipeline_root / step if step in self.steps() else None

    def artifacts(self, step: str) -> list[Path]:
        """Every file the step produced, sorted (records, figures, tables, reports)."""
        directory = self.step_dir(step)
        if directory is None:
            return []
        return sorted(path for path in directory.iterdir() if path.is_file())

    def records(self, step: str) -> list[PipelineRecord]:
        """The step's validated decision records, in filename order."""
        directory = self.step_dir(step)
        if directory is None:
            return []
        validated = (
            PipelineRecord.from_path(path) for path in sorted(directory.glob(f"*{_RECORD_SUFFIX}"))
        )
        return [record for record in validated if record is not None]

    def primary(self, step: str) -> PipelineRecord | None:
        """The record a step is summarised by: ``<step>_decision.json``, else the first."""
        records = self.records(step)
        return next(
            (record for record in records if record.stem == step), records[0] if records else None
        )

    def census_rows(self) -> list[dict[str, str | int]]:
        """One dashboard row per step: experiment | question | status | artifacts."""
        rows: list[dict[str, str | int]] = []
        for step in self.steps():
            record = self.primary(step)
            rows.append(
                {
                    "experiment": step,
                    "question": record.decision.question if record is not None else "",
                    "status": record.decision.verdict if record is not None else "no record",
                    "artifacts": len(self.artifacts(step)),
                }
            )
        return rows

    def url(self, path: Path) -> str:
        """The static URL serving one file of this series."""
        return "/results/" + quote(path.relative_to(self._results_root).as_posix(), safe="/")

    def module_source(self, step: str) -> Path | None:
        """The step's producing module: ``STEP_MODULES`` (the SSOT), else ``<step>/module.py``."""
        reference = STEP_MODULES.get(step)
        relative = (
            Path(*reference.split(".")[1:]).with_suffix(".py")
            if reference is not None
            else Path(step) / "module.py"
        )
        candidate = _PACKAGE_ROOT / relative
        return candidate if candidate.is_file() else None
