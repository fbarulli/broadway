"""Pipeline step evidence on the experiment dashboard.

Pipeline steps (``etl``, ``eda``, ...) are not experiment series: they publish
their decision record and evidence under
``<experiments root>/results/pipeline/<step>/``. The dashboard only discovers
numbered series scripts, so those records were unreachable from a fresh
checkout. This section scans the pipeline result folders for
``*_decision.json``, validates each one through the canonical ``Decision``
model, and renders one compact, fully escaped row per record, linking the
record, its per-feature evidence and the figures it was read from.
"""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from pydantic import ValidationError

from broadway.reports.decision import Decision

logger = logging.getLogger(__name__)

_RECORD_SUFFIX = "_decision.json"
_FIGURE_SUFFIX = ".png"
_DECIMALS = 4
# A permanently malformed record is skipped on every request; log its full
# traceback once so the contract's traceback rule does not become log bloat.
_LOGGED_UNREADABLE: set[Path] = set()


@dataclass(frozen=True)
class PipelineRecord:
    """One validated decision record plus the files that carry its evidence."""

    step: str
    record: Path
    decision: Decision
    figures: tuple[Path, ...]

    @property
    def stem(self) -> str:
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
                logger.exception(
                    "pipeline evidence: unreadable decision record at %s", record
                )
            return None
        return cls(
            step=record.parent.name,
            record=record,
            decision=decision,
            figures=cls._figures(record.parent, record.name[: -len(_RECORD_SUFFIX)]),
        )

    @staticmethod
    def _figures(directory: Path, stem: str) -> tuple[Path, ...]:
        """The step's figure(s): ``<stem>.png`` first, then ``<stem>_*.png``."""
        found: list[Path] = []
        candidates = (
            directory / f"{stem}{_FIGURE_SUFFIX}",
            *sorted(directory.glob(f"{stem}_*{_FIGURE_SUFFIX}")),
        )
        for candidate in candidates:
            if candidate.is_file() and candidate not in found:
                found.append(candidate)
        return tuple(found)


class PipelineEvidenceSection:
    """Render every pipeline step's decision record as one HTML table section."""

    def __init__(self, results_root: Path) -> None:
        self._results_root = Path(results_root)
        self._pipeline_root = self._results_root / "pipeline"

    @classmethod
    def from_experiments_root(cls, experiments_root: Path) -> PipelineEvidenceSection:
        """Build the section for the dashboard's configured experiments root."""
        return cls(Path(experiments_root) / "results")

    def render(self) -> str:
        """Return the section, or nothing when no pipeline record is published."""
        records = [PipelineRecord.from_path(path) for path in self._record_paths()]
        rows = [self._row(record) for record in records if record is not None]
        if not rows:
            return ""
        return (
            '<section class="pipeline-evidence">'
            "<h2>pipeline evidence</h2>"
            "<table><thead><tr>"
            "<th>step</th><th>metric</th><th>value [95%]</th><th>rule</th>"
            "<th>null band</th><th>n</th><th>verdict</th><th>evidence</th>"
            "</tr></thead><tbody>"
            f"{''.join(rows)}"
            "</tbody></table></section>"
        )

    def _record_paths(self) -> list[Path]:
        if not self._pipeline_root.is_dir():
            return []
        return [
            record
            for step_dir in sorted(self._pipeline_root.iterdir())
            if step_dir.is_dir()
            for record in sorted(step_dir.glob(f"*{_RECORD_SUFFIX}"))
        ]

    def _row(self, record: PipelineRecord) -> str:
        decision = record.decision
        null_band = (
            "—" if decision.null_threshold is None else f"{decision.null_threshold:,.{_DECIMALS}f}"
        )
        comparator = ">" if decision.direction == "max" else "<"
        dataset = decision.evidence.provenance.get("dataset", "")
        dataset_label = (
            f'<br><span class="dataset">{html.escape(dataset)}</span>' if dataset else ""
        )
        return (
            "<tr>"
            f"<td>{html.escape(record.step)}{dataset_label}</td>"
            f"<td>{html.escape(decision.metric)}</td>"
            f"<td>{decision.value:,.{_DECIMALS}f} "
            f"[{decision.interval_low:,.{_DECIMALS}f}, "
            f"{decision.interval_high:,.{_DECIMALS}f}]</td>"
            f"<td>{comparator} {decision.threshold:,.{_DECIMALS}f}</td>"
            f"<td>{null_band}</td>"
            f"<td>{decision.sample_size:,}</td>"
            f'<td class="verdict">{html.escape(decision.verdict)}</td>'
            f"<td>{self._links(record)}</td>"
            "</tr>"
        )

    def _links(self, record: PipelineRecord) -> str:
        links = []
        for figure in record.figures:
            label = (
                "figure"
                if figure.stem == record.stem
                else figure.stem[len(record.stem) + 1 :].replace("_", " ")
            )
            links.append(
                f'<a href="{html.escape(self._url(figure))}">{html.escape(label)}</a>'
            )
        if record.sidecar.is_file():
            links.append(
                f'<a href="{html.escape(self._url(record.sidecar))}">feature decisions</a>'
            )
        links.append(f'<a href="{html.escape(self._url(record.record))}">record</a>')
        return " · ".join(links)

    def _url(self, path: Path) -> str:
        return "/results/" + quote(path.relative_to(self._results_root).as_posix(), safe="/")
