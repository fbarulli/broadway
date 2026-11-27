"""Audit experiment steps against the output contract (agents/contracts/EXPERIMENT_OUTPUT.md).

A step's output is written for a human: a figure that answers the step's own
question, plus the aggregate evidence. This checker is the ratchet: series
listed in ``ENFORCED`` must comply; the rest are counted and reported only.
"""

from __future__ import annotations

import ast
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from broadway.reports.decision import Decision

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS = ROOT / "project" / "experiments"
# Pipeline steps whose evidence lives under the project tree. They are only
# enforced when the project layout (and its results root) is present, so a
# data-agnostic checkout stays green.
PROJECT_EVIDENCE_STEPS: tuple[str, ...] = ("eda",)

# Ratchet: a series joins this list once every step complies. Never remove one.
ENFORCED: tuple[str, ...] = ("amazon_survey", "fare_prediction", "timeseries")

# The decision half of the contract, per step (a step publishes a computed
# verdict). A step joins this set when it builds a StepOutput with a Decision.
# Never remove.
DECISION_ENFORCED: tuple[tuple[str, str], ...] = (
    ("fare_prediction", "01_generate_sample"),
    ("fare_prediction", "02_filtered_profile"),
    ("fare_prediction", "03_short_trips"),
    ("fare_prediction", "04_percentile_extremes_audit"),
    ("fare_prediction", "05_temporal_features"),
    ("fare_prediction", "06_prepare_model_data"),
    ("fare_prediction", "07_specification_diagnostics"),
    ("fare_prediction", "08_lightgbm_baseline"),
    ("timeseries", "01_series_profile"),
    ("timeseries", "02_seasonality"),
    ("timeseries", "03_stationarity_acf"),
    ("timeseries", "04_forecast_models"),
    ("amazon_survey", "01_profile"),
)

# Pipeline modules that must publish a decision record (the same contract, one
# level up from the experiment scripts). Grows as each step is wired; never shrinks.
PIPELINE_STEPS: tuple[str, ...] = (
    "etl",
    "eda",
    "stats",
    "features",
    "training",
    "evaluate",
    "baseline",
    "causal",
    # tracking has no step module yet; it joins the decision gate with the
    # serve feed that makes a train-vs-serve drift verdict measurable.
    "tracking",
)
# Steps that now publish a computed Decision/StepOutput. An entry is added
# once it complies; never removed.
PIPELINE_DECISION_ENFORCED: tuple[str, ...] = (
    "etl",
    "eda",
    "features",
    "training",
    "evaluate",
    "baseline",
    "causal",
)


@dataclass(frozen=True)
class ProjectLayout:
    """The project roots the gate may audit, or why they are unavailable."""

    experiments: Path | None
    results: Path | None
    reason: str = ""

    @property
    def pipeline_results(self) -> Path | None:
        return None if self.results is None else self.results / "pipeline"


def resolve_project_layout() -> ProjectLayout:
    """Import the project layer lazily so a platform-only checkout stays runnable."""
    try:
        from project.paths import load_project_paths

        paths = load_project_paths()
    except (ImportError, FileNotFoundError, ValueError) as error:
        return ProjectLayout(None, None, f"project layout unavailable ({error})")
    return ProjectLayout(paths.experiments, paths.results)


@dataclass(frozen=True)
class StepAudit:
    """One experiment step's compliance with the output contract."""

    series: str
    stem: str
    has_question: bool
    has_figure: bool
    has_decision: bool

    @property
    def compliant(self) -> bool:
        return self.has_question and self.has_figure

    def reason(self) -> str:
        missing = []
        if not self.has_question:
            missing.append("no module docstring (the question)")
        if not self.has_figure:
            missing.append("no figure (savefig)")
        return ", ".join(missing)

    def decision_reason(self) -> str:
        if not self.has_decision:
            return "no StepOutput decision record (metric + interval + verdict)"
        return "no valid decision record at the pair's results path"


@dataclass
class SeriesAudit:
    """The audited steps of one series."""

    series: str
    steps: list[StepAudit] = field(default_factory=list)

    @property
    def violations(self) -> list[StepAudit]:
        return [step for step in self.steps if not step.compliant]


class ExperimentOutputAudit:
    """Walk the experiment tree and classify each step script.

    ``DECISION_ENFORCED`` pairs must publish a record that validates through the
    canonical ``Decision`` — the script carrying a ``StepOutput`` call is not
    evidence that the record exists (a deleted record would otherwise pass).
    """

    def __init__(
        self,
        root: Path | None = DEFAULT_EXPERIMENTS,
        enforced: tuple[str, ...] = ENFORCED,
        results_root: Path | None = None,
        decision_enforced: tuple[tuple[str, str], ...] = DECISION_ENFORCED,
    ) -> None:
        self._root = root
        self._enforced = enforced
        self._decision_enforced = decision_enforced
        self._results_root = (
            results_root
            if results_root is not None
            else (Path(root) / "results" if root is not None else None)
        )

    def series(self) -> list[SeriesAudit]:
        if self._root is None or not self._root.is_dir():
            return []
        audits = []
        for directory in sorted(path for path in self._root.iterdir() if path.is_dir()):
            steps = [
                self._step(directory.name, script)
                for script in sorted(directory.glob("[0-9][0-9]*_*.py"))
            ]
            if steps:
                audits.append(SeriesAudit(directory.name, steps))
        return audits

    def _step(self, series: str, script: Path) -> StepAudit:
        tree = ast.parse(script.read_text(encoding="utf-8"))
        has_question = ast.get_docstring(tree) not in (None, "")
        has_figure = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"savefig", "save_figure"}
            for node in ast.walk(tree)
        )
        has_decision = any(
            isinstance(node, ast.Call)
            and (
                getattr(node.func, "id", None) == "StepOutput"
                or getattr(node.func, "attr", None) == "StepOutput"
            )
            for node in ast.walk(tree)
        )
        return StepAudit(
            series=series,
            stem=script.stem,
            has_question=has_question,
            has_figure=has_figure,
            has_decision=has_decision,
        )

    def violations(self) -> list[StepAudit]:
        """Violations in the enforced series/pairs only (the ratchet's teeth)."""
        steps = [step for audit in self.series() for step in audit.steps]
        base = [step for step in steps if step.series in self._enforced and not step.compliant]
        insight = [
            step
            for step in steps
            if (step.series, step.stem) in self._decision_enforced
            and not self._publishes_decision(step)
        ]
        return base + insight

    def _publishes_decision(self, step: StepAudit) -> bool:
        """A pair passes only with a published record that validates as a Decision."""
        if not step.has_decision or self._results_root is None:
            return False
        record = self._results_root / step.series / f"{step.stem}_decision.json"
        try:
            Decision.model_validate_json(record.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValidationError):
            logger.warning(
                "experiment audit: no valid decision record for %s/%s at %s",
                step.series,
                step.stem,
                record,
                exc_info=True,
            )
            return False
        return True

    def advisory(self) -> list[StepAudit]:
        """Non-compliant steps outside the enforced series (reported, not gated)."""
        return [
            step
            for audit in self.series()
            if audit.series not in self._enforced
            for step in audit.violations
        ]


class PipelineDecisionAudit:
    """Which pipeline steps publish a Decision/StepOutput today."""

    def __init__(
        self,
        root: Path = ROOT / "src" / "broadway",
        results_root: Path | None = None,
    ) -> None:
        self._root = root
        self._results_root = results_root

    def status(self) -> dict[str, bool]:
        status = {}
        for step in PIPELINE_STEPS:
            module = self._root / step / "module.py"
            if not module.is_file():
                continue
            status[step] = self._publishes_in_code(module) or self._publishes_record(step)
        return status

    @staticmethod
    def _publishes_in_code(module: Path) -> bool:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        return any(
            isinstance(node, ast.Call)
            and (
                getattr(node.func, "id", None) == "StepOutput"
                or getattr(node.func, "attr", None) == "StepOutput"
                or (
                    getattr(node.func, "attr", None) == "evaluate"
                    and getattr(getattr(node.func, "value", None), "id", None) == "Decision"
                )
            )
            for node in ast.walk(tree)
        )

    def _publishes_record(self, step: str) -> bool:
        """A published record counts only when the canonical Decision validates."""
        if self._results_root is None:
            return False
        for record in sorted((self._results_root / step).glob("*_decision.json")):
            try:
                Decision.model_validate_json(record.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValidationError):
                logger.warning(
                    "pipeline audit: unreadable decision record at %s", record, exc_info=True
                )
                continue
            return True
        return False


def _enforced_pipeline_steps(layout: ProjectLayout) -> tuple[str, ...]:
    """Enforce project evidence whenever this is a project checkout.

    The evidence directory itself is not the signal: a project checkout whose
    results tree vanished must fail the missing record, not skip it.
    """
    if layout.experiments is not None and layout.experiments.is_dir():
        return PIPELINE_DECISION_ENFORCED
    return tuple(
        step for step in PIPELINE_DECISION_ENFORCED if step not in PROJECT_EVIDENCE_STEPS
    )


def main() -> int:
    layout = resolve_project_layout()
    if layout.reason:
        print(f"note project experiments skipped: {layout.reason}")
    audit = ExperimentOutputAudit(layout.experiments, results_root=layout.results)
    violations = audit.violations()
    advisory = audit.advisory()
    for step in violations:
        print(f"FAIL {step.series}/{step.stem}.py: {step.reason() or step.decision_reason()}")
    for step in advisory:
        print(
            f"note {step.series}/{step.stem}.py: {step.reason() or step.decision_reason()} "
            "(not yet enforced)"
        )
    enforced_pipeline = _enforced_pipeline_steps(layout)
    skipped_pipeline = tuple(
        step
        for step in PROJECT_EVIDENCE_STEPS
        if step in PIPELINE_DECISION_ENFORCED and step not in enforced_pipeline
    )
    pipeline = PipelineDecisionAudit(results_root=layout.pipeline_results).status()
    for pipeline_step, publishes in sorted(pipeline.items()):
        if publishes:
            continue
        if pipeline_step in skipped_pipeline:
            print(
                f"note pipeline/{pipeline_step}: decision record not enforced "
                "(project results root unavailable)"
            )
            continue
        enforced = pipeline_step in enforced_pipeline
        print(
            f"{'FAIL' if enforced else 'note'} pipeline/{pipeline_step}: "
            f"{'missing' if enforced else 'no'} decision record"
            + ("" if enforced else " (not yet enforced)")
        )
    if violations or any(
        pipeline_step in enforced_pipeline
        for pipeline_step in pipeline
        if not pipeline[pipeline_step]
    ):
        print(
            "EXPERIMENT OUTPUTS RED — every enforced step needs a question (docstring) "
            "and a figure answering it (savefig). See agents/contracts/EXPERIMENT_OUTPUT.md"
        )
        return 1
    enforced_steps = sum(
        len(audit_.steps) for audit_ in audit.series() if audit_.series in audit._enforced
    )
    print(f"EXPERIMENT OUTPUTS OK — {enforced_steps} enforced steps, {len(advisory)} advisory gaps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
