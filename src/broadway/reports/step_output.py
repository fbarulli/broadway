"""One output writer per experiment step: question, decision, figure, evidence.

The contract (``agents/contracts/EXPERIMENT_OUTPUT.md``) says a step's output is
for a human making a decision. A figure alone is not evidence: the step must
state one metric, its 95% interval, the config-declared threshold, and the
computed verdict (``act`` / ``investigate`` / ``no_action``) with the owner's
action. The verdict comes from :class:`~broadway.reports.decision.Decision`,
never hand-written.

Every step builds a :class:`StepOutput`, attaches one figure, optional evidence
tables, and writes the report plus ``<stem>_decision.json``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from broadway.reports.decision import Decision
from broadway.reports.decision_log import DecisionLog


class StepOutput:
    """Write one step's report + figure + decision record under a results dir."""

    def __init__(
        self,
        *,
        results_dir: Path,
        stem: str,
        decision: Decision,
        log: DecisionLog | None = None,
    ) -> None:
        self._results_dir = Path(results_dir)
        self._stem = stem
        self._decision = decision
        self._log = log if log is not None else DecisionLog.from_config()
        self._tables: list[tuple[str, Path]] = []
        self._figure_path = self._results_dir / f"{stem}.png"
        self._report_path = self._results_dir / f"{stem}.md"
        self._decision_path = self._results_dir / f"{stem}_decision.json"

    @property
    def figure_path(self) -> Path:
        return self._figure_path

    @property
    def report_path(self) -> Path:
        return self._report_path

    @property
    def decision_path(self) -> Path:
        return self._decision_path

    def save_figure(self, figure: object) -> Path:
        """Persist the matplotlib figure under the step's canonical name."""
        self._results_dir.mkdir(parents=True, exist_ok=True)
        savefig = getattr(figure, "savefig", None)
        if savefig is None:
            raise TypeError("StepOutput.save_figure needs a matplotlib Figure")
        savefig(self._figure_path, dpi=150)
        return self._figure_path

    def add_table(self, name: str, frame: pd.DataFrame) -> Path:
        """Persist an evidence table (aggregates only) under the step's prefix."""
        self._results_dir.mkdir(parents=True, exist_ok=True)
        path = self._results_dir / f"{self._stem}_{name}.csv"
        frame.to_csv(path, index=False)
        self._tables.append((name, path))
        return path

    def _markdown(self) -> str:
        decision = self._decision
        comparison = (
            f"- reference: {decision.reference:,.4f} (delta {decision.delta:+,.4f})\n"
            if decision.reference is not None
            else ""
        )
        tables = "\n".join(
            f"- [{name}]({path.name})" for name, path in self._tables
        ) or "- none"
        evidence = self._decision.evidence
        segments = "\n".join(
            f"| {item.segment} | {item.value:,.4f} | {item.sample_size:,} |"
            for item in evidence.segments
        )
        sensitivity = "\n".join(
            f"- {name}: {value:,.4f}" for name, value in evidence.sensitivity.items()
        )
        provenance = "\n".join(
            f"- {name}: {value}" for name, value in evidence.provenance.items()
        )
        delta = (
            f"\n- delta vs baseline: {evidence.delta:+,.4f}"
            if evidence.delta is not None
            else ""
        )
        null = (
            f" (permutation null {decision.null_threshold:,.4f})"
            if decision.null_threshold is not None
            else ""
        )
        return (
            f"# {self._stem}\n\n"
            f"## question\n\n{decision.question}\n\n"
            f"## decision — {decision.verdict}\n\n"
            f"- metric: {decision.metric} = {decision.value:,.4f} "
            f"[{decision.interval_low:,.4f}, {decision.interval_high:,.4f}] 95% "
            f"(n={decision.sample_size:,})\n"
            f"- rule: {decision.direction} {decision.threshold:,.4f}{null}\n"
            f"{comparison}"
            f"- action ({decision.owner}): {decision.action}\n\n"
            f"![figure]({self._figure_path.name})\n\n"
            f"## evidence\n\n"
            f"- baseline: {evidence.baseline}{delta}\n\n"
            f"### segments\n\n| segment | value | n |\n| --- | ---: | ---: |\n{segments}\n\n"
            f"### sensitivity\n\n{sensitivity}\n\n"
            f"### falsifier\n\n{evidence.falsifier}\n\n"
            f"### provenance\n\n{provenance}\n\n"
            f"## tables\n\n{tables}\n"
        )

    def write(self) -> tuple[Path, Path, Path]:
        """Write report + decision record; return ``(report, figure, decision)``."""
        self._results_dir.mkdir(parents=True, exist_ok=True)
        if not self._figure_path.exists():
            raise FileNotFoundError(
                f"step {self._stem}: no figure saved at {self._figure_path} — "
                "an output without a figure violates EXPERIMENT_OUTPUT.md"
            )
        self._report_path.write_text(self._markdown(), encoding="utf-8")
        self._decision_path.write_text(
            self._decision.model_dump_json(indent=2), encoding="utf-8"
        )
        self._log.log(
            stem=self._stem,
            decision=self._decision,
            figure=self._figure_path,
            tables=self._tables,
            params={},
            record_path=self._decision_path,
        )
        return self._report_path, self._figure_path, self._decision_path
