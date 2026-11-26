"""Audit experiment steps against the output contract (agents/contracts/EXPERIMENT_OUTPUT.md).

A step's output is written for a human: a figure that answers the step's own
question, plus the aggregate evidence. This checker is the ratchet: series
listed in ``ENFORCED`` must comply; the rest are counted and reported only.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "project" / "experiments"

# Ratchet: a series joins this list once every step complies. Never remove one.
ENFORCED: tuple[str, ...] = ("fare_prediction", "timeseries")

# The decision half of the contract, per step (a step publishes a computed
# verdict). A step joins this set when it builds a StepOutput with a Decision.
# Never remove.
DECISION_ENFORCED: tuple[tuple[str, str], ...] = (("fare_prediction", "01_generate_sample"),)


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
        return "" if self.has_decision else "no StepOutput decision record (metric + interval + verdict)"


@dataclass
class SeriesAudit:
    """The audited steps of one series."""

    series: str
    steps: list[StepAudit] = field(default_factory=list)

    @property
    def violations(self) -> list[StepAudit]:
        return [step for step in self.steps if not step.compliant]


class ExperimentOutputAudit:
    """Walk the experiment tree and classify each step script."""

    def __init__(self, root: Path = EXPERIMENTS, enforced: tuple[str, ...] = ENFORCED) -> None:
        self._root = root
        self._enforced = enforced

    def series(self) -> list[SeriesAudit]:
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
            if (step.series, step.stem) in DECISION_ENFORCED and not step.has_decision
        ]
        return base + insight

    def advisory(self) -> list[StepAudit]:
        """Non-compliant steps outside the enforced series (reported, not gated)."""
        return [
            step
            for audit in self.series()
            if audit.series not in self._enforced
            for step in audit.violations
        ]


def main() -> int:
    audit = ExperimentOutputAudit()
    violations = audit.violations()
    advisory = audit.advisory()
    for step in violations:
        print(f"FAIL {step.series}/{step.stem}.py: {step.reason()}")
    for step in advisory:
        print(f"note {step.series}/{step.stem}.py: {step.reason()} (not yet enforced)")
    if violations:
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
