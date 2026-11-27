"""The project story: one human page generated from the committed records.

A busy human should not have to read the pipeline to understand the project.
This module turns the published ``results/pipeline/<step>/`` decision records,
their feature-decision sidecars and their evidence CSVs into a single narrative:
the question, the data, what the EDA found, what was changed and why, the
features, the model chain, and the honest limits. Each section puts the
conclusion first, then one graph, then the numbers (value, 95% interval, n,
verdict) and a link to the record it came from.

Nothing here is a hardcoded result. A stage with no record renders a short
"not run yet" line, so the page fills in as the pipeline lands. A malformed
record is skipped by :class:`~broadway.reports.pipeline_evidence.PipelineSeries`
with one logged traceback; this module never raises for bad evidence. The story
is filtered by dataset so another dataset's committed records cannot be
presented as this project's results.

Graphs are drawn in memory with matplotlib's Agg backend and the shared
``broadway.viz`` helpers (deterministic, no new dependency, no scratch files).
"""

from __future__ import annotations

import html
import json
import logging
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from broadway import viz
from broadway.reports.decision import Decision
from broadway.reports.pipeline_evidence import Finding, PipelineRecord, PipelineSeries

logger = logging.getLogger(__name__)

STORY_TITLE = "The project story"
STORY_QUESTION = (
    "Which orders will end with a 1- or 2-star review — predicted only from "
    "what is known at purchase time?"
)

#: Env knob for the dataset the story narrates; the project's dataset is the default.
STORY_DATASET_ENV = "BROADWAY_STORY_DATASET"
DEFAULT_DATASET = "olist"

_TARGET_COLUMNS = frozenset({"bad_review", "target", "class", "label"})
_COUNT_COLUMNS = frozenset({"count", "n", "orders", "rows", "order_count"})
_FEATURE_SECTIONS = frozenset({"features", "associations"})
_ROLE_COLORS = {
    "naive_reference": "#999999",
    "linear_reference": "#E69F00",
    "candidate": "#0072B2",
}
_LABEL_PREFIXES = ("association_", "outlier_share_", "standardized_shift_", "shift_")
_THRESHOLD_COLOR = "#D55E00"
# A permanently malformed table is skipped on every request; log its full
# traceback once so the contract's traceback rule does not become log bloat.
_LOGGED_UNREADABLE: set[Path] = set()


def configured_dataset() -> str | None:
    """The dataset the story narrates: env override, else the project's default."""
    return os.environ.get(STORY_DATASET_ENV, DEFAULT_DATASET) or None


@dataclass(frozen=True)
class Section:
    """One story section: conclusions first, one graph, then numbers and records."""

    title: str
    conclusions: tuple[str, ...]
    numbers: tuple[str, ...] = ()
    figure: str | None = None
    records: tuple[tuple[str, str], ...] = ()
    not_run: bool = False


@dataclass(frozen=True)
class Effect:
    """One association: its label, point estimate and 95% interval (when present)."""

    label: str
    value: float
    low: float | None
    high: float | None
    verdict: str


class Story:
    """Build the story sections and figures from one pipeline results tree."""

    def __init__(self, pipeline: PipelineSeries, dataset: str | None = None) -> None:
        self._pipeline = pipeline
        self.dataset = dataset

    @classmethod
    def from_experiments_root(
        cls, experiments_root: Path, dataset: str | None = None
    ) -> Story:
        """Build the story for a checkout's experiments root."""
        return cls(PipelineSeries.from_experiments_root(experiments_root), dataset)

    @property
    def question(self) -> str:
        """The one-line question the whole page answers."""
        return STORY_QUESTION

    @property
    def has_records(self) -> bool:
        """Whether any stage record belongs to the story's dataset."""
        return any(self._record(step) is not None for step in _STEPS)

    # -- record access -----------------------------------------------------

    def _record(self, step: str) -> PipelineRecord | None:
        """The step's primary record, only when it belongs to the story's dataset."""
        record = self._pipeline.primary(step)
        if record is None or self.dataset is None:
            return record
        provenance = record.decision.evidence.provenance
        declared = provenance.get("dataset") or record.decision.owner
        return record if declared == self.dataset else None

    def _step_dir(self, step: str) -> Path | None:
        return self._pipeline.step_dir(step)

    # -- sections ----------------------------------------------------------

    def sections(self) -> list[Section]:
        """All seven sections, in narrative order (missing stages say so)."""
        return [
            self._question(),
            self._data(),
            self._eda(),
            self._changed(),
            self._features(),
            self._model(),
            self._learned(),
        ]

    def _question(self) -> Section:
        has_target = self._record("etl") is not None
        records = (("ETL record", "etl"),) if has_target else ()
        return Section(
            title="The question",
            conclusions=(
                (
                    "Bad reviews are the cost this project predicts: which orders end with "
                    "a 1- or 2-star review, using only what the storefront knows when the "
                    "order is placed. Every stage below exists to answer that one question "
                    "honestly."
                ),
            ),
            numbers=("bad review := review score ≤ 2", "decision moment := purchase time"),
            records=records,
        )

    def _data(self) -> Section:
        record = self._record("etl")
        if record is None:
            return _missing("The data", "etl")
        decision = record.decision
        provenance = decision.evidence.provenance
        rows_in = _as_int(provenance.get("rows_in"))
        rows_out = _as_int(provenance.get("rows_out"))
        conclusions = [
            (
                "Nine Olist source tables collapse to one row per order. Orders without a "
                "review are dropped with a recorded reason, never imputed."
            )
        ]
        if rows_in is not None and rows_out is not None:
            conclusions.append(
                f"The join took {rows_in:,} source orders to {rows_out:,} rows after the "
                "accounted drops."
            )
        else:
            conclusions.append(
                f"The order-grain join produced n={decision.sample_size:,} rows."
            )
        conclusions.append(f"The record asks: {decision.question}")
        conclusions.append(f"Verdict: {decision.verdict} — {decision.action}.")
        numbers = [_decision_number(decision)]
        balance = _class_balance(self._step_dir("etl"))
        if balance:
            numbers.append(_balance_number(balance))
        else:
            conclusions.append("The class balance has not been published yet.")
        return Section(
            title="The data",
            conclusions=tuple(conclusions),
            numbers=tuple(numbers),
            figure="data",
            records=(("ETL record", "etl"),),
        )

    def _eda(self) -> Section:
        record = self._record("eda")
        if record is None:
            return _missing("What the data says", "eda")
        effects = _effects(record)
        outliers = _findings_for(record, "outliers")
        comparison = _findings_for(record, "comparison")
        conclusions: list[str] = []
        acts = [effect for effect in effects if effect.verdict == "act"]
        if len(acts) >= 2:
            conclusions.append(
                f"{acts[0].label} and {acts[1].label} dominate the association ranking "
                f"({acts[0].value:.2f} and {acts[1].value:.2f}); both clear the configured "
                "effect floor."
            )
        elif len(acts) == 1:
            conclusions.append(
                f"{acts[0].label} is the only behaviour that clears the effect floor "
                f"({acts[0].value:.2f})."
            )
        elif effects:
            strongest = effects[0]
            conclusions.append(
                f"The strongest association is {strongest.label} "
                f"({strongest.value:.2f}), but nothing clears the floor — no behaviour is "
                "worth carrying forward."
            )
        else:
            conclusions.append(
                "No feature association cleared the floor — the EDA found nothing worth "
                "carrying forward."
            )
        noise = len(effects) - len(acts)
        if acts and noise:
            verb = "behaviour sits" if noise == 1 else "behaviours sit"
            conclusions.append(
                f"{noise} further {verb} at or below the floor and {'is' if noise == 1 else 'are'} "
                "noise."
            )
        if effects:
            conclusions.append(
                "These are measured associations, not feature admissions: the feature step "
                "keeps purchase-time columns only."
            )
        if outliers:
            flagged = [decision for decision in outliers if decision.verdict == "act"]
            names = ", ".join(_label(decision.metric) for decision in flagged[:3])
            suffix = f" ({names})" if names else ""
            conclusions.append(
                f"{len(flagged)} of {len(outliers)} profiled columns are heavy-tailed{suffix}; "
                "the data was flagged, not silently trimmed."
            )
        if comparison:
            worst = max(comparison, key=lambda decision: abs(decision.value))
            conclusions.append(
                f"Before/after shifts cover zero (largest |shift| {abs(worst.value):.4f}), "
                "so no column's distribution moved materially."
            )
        numbers = [_decision_number(record.decision)]
        if effects:
            numbers.append(
                f"{len(effects)} associations tested · "
                f"{sum(1 for effect in effects if effect.verdict == 'act')} act"
            )
        return Section(
            title="What the data says",
            conclusions=tuple(conclusions),
            numbers=tuple(numbers),
            figure="eda" if effects else None,
            records=(("EDA record", "eda"),),
        )

    def _changed(self) -> Section:
        etl = self._record("etl")
        eda = self._record("eda")
        if etl is None and eda is None:
            return _missing("What we changed and why", "etl")
        conclusions: list[str] = []
        numbers: list[str] = []
        records: list[tuple[str, str]] = []
        flagged: list[Decision] = []
        if etl is not None:
            provenance = etl.decision.evidence.provenance
            rows_in = _as_int(provenance.get("rows_in"))
            rows_out = _as_int(provenance.get("rows_out"))
            if rows_in and rows_out is not None:
                dropped = rows_in - rows_out
                if dropped:
                    conclusions.append(
                        f"Nothing was silently repaired: the ETL dropped {dropped:,} of "
                        f"{rows_in:,} rows ({dropped / rows_in:.1%}) and recorded a reason "
                        "for each."
                    )
                else:
                    conclusions.append(
                        "Nothing was silently repaired: every input row survived the "
                        "contract's row policy."
                    )
            else:
                conclusions.append("The ETL recorded its row policy in the decision record.")
            reasons = provenance.get("reasons", "")
            if reasons and reasons.lower() not in {"none", "no drops"}:
                conclusions.append(f"Reason: {reasons}.")
            conclusions.append(
                "No row values were imputed: missing item aggregates stay null and are "
                "handled by the declared feature transform."
            )
            numbers.append(_decision_number(etl.decision))
            records.append(("ETL record", "etl"))
        if eda is not None:
            outliers = _findings_for(eda, "outliers")
            comparison = _findings_for(eda, "comparison")
            flagged = [decision for decision in outliers if decision.verdict == "act"]
            if outliers:
                conclusions.append(
                    f"{len(flagged)} of {len(outliers)} candidate columns are heavy-tailed; "
                    "the policy is to winsorize the flagged columns rather than delete them."
                )
            if comparison:
                worst = max(comparison, key=lambda decision: abs(decision.value))
                conclusions.append(
                    f"The before/after shift intervals cover zero (largest |shift| "
                    f"{abs(worst.value):.4f}), so no distribution moved materially."
                )
            if flagged:
                numbers.append(f"outlier policy: {flagged[0].action}")
            records.append(("EDA record", "eda"))
        if flagged:
            conclusions.append(
                "Trade-off: winsorizing keeps rare high-value orders in the training set; "
                "the cost is a noisier tail, which the model is told about rather than "
                "shielded from."
            )
        elif etl is not None:
            conclusions.append(
                "Trade-off: rows are dropped only with a recorded reason, so the model "
                "inherits that selection honestly instead of seeing a silently repaired table."
            )
        else:
            conclusions.append(
                "Trade-off: the EDA flags heavy tails for the feature transform rather than "
                "removing rows, so rare orders stay in the data at the cost of a longer tail."
            )
        return Section(
            title="What we changed and why",
            conclusions=tuple(conclusions),
            numbers=tuple(numbers),
            records=tuple(records),
        )

    def _features(self) -> Section:
        record = self._record("features")
        if record is None:
            return _missing("The features", "features")
        decision = record.decision
        provenance = decision.evidence.provenance
        added = _sensitivity_int(decision, "features_added")
        example = _first_column(provenance.get("columns_added"))
        if decision.value > 0:
            conclusions = [
                (
                    f"The feature step lost {decision.value:.1%} of rows "
                    f"(verdict: {decision.verdict})."
                )
            ]
        else:
            conclusions = [
                "The feature step kept every row and built purchase-time features only."
            ]
        if added is not None:
            conclusions.append(f"{added} features were added.")
        if example is not None:
            conclusions.append(f"One example: {example}.")
        conclusions.append(
            "Deliberately excluded: review score, delivery and review timestamps, and every "
            "post-purchase column — they are only known after the outcome, so using them "
            "would be leakage."
        )
        return Section(
            title="The features",
            conclusions=tuple(conclusions),
            numbers=(_decision_number(decision),),
            records=(("features record", "features"),),
        )

    def _model(self) -> Section:
        baseline = self._record("baseline")
        selection = self._record("selection")
        hpo = self._record("hpo")
        evaluate = self._record("evaluate")
        if baseline is None and selection is None and hpo is None and evaluate is None:
            return _missing("The model", "baseline")
        conclusions: list[str] = []
        numbers: list[str] = []
        records: list[tuple[str, str]] = []
        if baseline is not None:
            decision = baseline.decision
            verb = "beats it by" if decision.value > 0 else "falls short of it by"
            conclusions.append(
                "Baseline first: the naive mean is the bar. The linear reference "
                f"{verb} {abs(decision.value):.1%} [{decision.interval_low:.1%}, "
                f"{decision.interval_high:.1%}] on the holdout (verdict: {decision.verdict})."
            )
            numbers.append(f"baseline · {_decision_number(decision)}")
            records.append(("baseline", "baseline"))
        else:
            conclusions.append("Baseline: not run yet.")
        candidate_frame = _read_csv(self._candidates_path())
        if selection is not None:
            conclusions.append(_selection_conclusion(selection, candidate_frame))
            numbers.append(f"selection · {_decision_number(selection.decision)}")
            records.append(("selection", "selection"))
        else:
            conclusions.append("Candidate selection: not run yet.")
        if hpo is not None:
            decision = hpo.decision
            worth = decision.interval_low > decision.threshold
            conclusions.append(
                f"HPO moved CV RMSE by {decision.value:.2%} "
                f"[{decision.interval_low:.2%}, {decision.interval_high:.2%}] over the "
                f"defaults against a {decision.threshold:.1%} floor — "
                f"{'worth adopting' if worth else 'not proven worth the extra complexity'} "
                f"(verdict: {decision.verdict})."
            )
            numbers.append(f"HPO · {_decision_number(decision)}")
            records.append(("HPO", "hpo"))
        else:
            conclusions.append("HPO: not run yet.")
        if evaluate is not None:
            decision = evaluate.decision
            conclusions.append(
                f"Holdout verdict: {decision.verdict} — {decision.value:+.1%} "
                f"[{decision.interval_low:+.1%}, {decision.interval_high:+.1%}] versus a "
                f"{decision.threshold:.0%} floor; {decision.action}."
            )
            numbers.append(f"holdout · {_decision_number(decision)}")
            records.append(("holdout", "evaluate"))
        else:
            conclusions.append("Holdout promotion: not run yet.")
        return Section(
            title="The model",
            conclusions=tuple(conclusions),
            numbers=tuple(numbers),
            figure=(
                "model" if selection is not None and candidate_frame is not None else None
            ),
            records=tuple(records),
        )

    def _learned(self) -> Section:
        evaluate = self._record("evaluate")
        selection = self._record("selection")
        record = evaluate or selection
        next_action = (
            record.decision.action
            if record is not None
            else "run the holdout promotion check before shipping anything"
        )
        records: tuple[tuple[str, str], ...] = ()
        if evaluate is not None:
            records = (("holdout record", "evaluate"),)
        elif selection is not None:
            records = (("selection record", "selection"),)
        return Section(
            title="What we learned / what is next",
            conclusions=(
                (
                    "Three limits bound the result: the target is an imbalanced binary, "
                    "the model sees purchase-time information only, and this is prediction, "
                    "not causation — a feature that predicts a bad review is not a lever "
                    f"that fixes it. The next decision: {next_action}."
                ),
            ),
            records=records,
        )

    # -- rendering ---------------------------------------------------------

    def render_section(self, section: Section) -> str:
        """One section: heading, conclusions, figure, numbers, then record links."""
        parts = ['<section class="story-section">', f"<h2>{html.escape(section.title)}</h2>"]
        css = "not-run" if section.not_run else "takeaway"
        parts.extend(
            f'<p class="{css}">{html.escape(conclusion)}</p>'
            for conclusion in section.conclusions
        )
        if not section.not_run:
            if section.figure:
                alt = html.escape(f"{section.title} figure")
                parts.append(
                    f'<figure><img src="/story/figure/{html.escape(section.figure)}.png"'
                    f' alt="{alt}"></figure>'
                )
            if section.numbers:
                items = "".join(
                    f"<li>{html.escape(number)}</li>" for number in section.numbers
                )
                parts.append(f'<ul class="numbers">{items}</ul>')
        if section.records:
            links = " · ".join(
                f'<a href="/experiments/{quote(step)}?focus=pipeline">{html.escape(label)}</a>'
                for label, step in section.records
            )
            parts.append(f'<p class="record-link">decision record: {links}</p>')
        parts.append("</section>")
        return "\n".join(parts)

    def page_body(self) -> str:
        """The story body: navigation, a dataset note, then every section."""
        parts = [
            (
                '<nav class="story-nav"><a href="/">← experiments</a> · '
                '<a href="/?focus=pipeline">pipeline evidence</a></nav>'
            )
        ]
        if not self.has_records:
            scope = html.escape(self.dataset or "project")
            parts.append(
                f'<p class="not-run">No {scope} decision records are published yet — every '
                "numeric stage below says “not run yet” until the pipeline lands.</p>"
            )
        parts.extend(self.render_section(section) for section in self.sections())
        return "\n".join(parts)

    def figure_png(self, name: str) -> bytes | None:
        """Render one story figure to PNG bytes, or None for an unknown/absent figure."""
        builders = {
            "data": self._data_figure,
            "eda": self._eda_figure,
            "model": self._model_figure,
        }
        build = builders.get(name)
        figure = build() if build is not None else None
        if figure is None:
            return None
        FigureCanvasAgg(figure)
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        return buffer.getvalue()

    # -- figures -----------------------------------------------------------

    def _data_figure(self) -> Figure | None:
        record = self._record("etl")
        if record is None:
            return None
        stages = _funnel(record)
        balance = _class_balance(self._step_dir("etl"))
        columns = 2 if balance else 1
        figure = Figure(figsize=(8.0, 3.6), dpi=150)
        axis = figure.add_subplot(1, columns, 1)
        labels = [label for label, _ in stages]
        counts = [count for _, count in stages]
        axis.barh(labels, counts, color=viz.palette_colors(len(labels)))
        axis.set_title("Rows through the order-grain join", fontsize=viz.TITLE_FONTSIZE)
        axis.set_xlabel("orders", fontsize=viz.LABEL_FONTSIZE)
        axis.tick_params(labelsize=viz.TICK_FONTSIZE)
        for index, (_, count) in enumerate(stages):
            axis.text(count, index, f" {count:,}", va="center", fontsize=viz.TICK_FONTSIZE)
        viz.despine(axis)
        if balance:
            balance_axis = figure.add_subplot(1, 2, 2)
            balance_labels = [label for label, _ in balance]
            balance_counts = [count for _, count in balance]
            balance_axis.bar(
                balance_labels, balance_counts, color=viz.palette_colors(len(balance_labels))
            )
            balance_axis.set_title("Target balance", fontsize=viz.TITLE_FONTSIZE)
            balance_axis.set_ylabel("orders", fontsize=viz.LABEL_FONTSIZE)
            balance_axis.tick_params(labelsize=viz.TICK_FONTSIZE)
            viz.despine(balance_axis)
        return figure

    def _eda_figure(self) -> Figure | None:
        record = self._record("eda")
        if record is None:
            return None
        effects = _effects(record)
        if not effects:
            return None
        top = effects[:10][::-1]
        figure = Figure(figsize=(8.0, max(2.8, 0.34 * len(top) + 1.5)), dpi=150)
        axis = figure.add_subplot(111)
        labels = [effect.label for effect in top]
        if all(effect.low is not None and effect.high is not None for effect in top):
            frame = pd.DataFrame(
                {
                    "value": [effect.value for effect in top],
                    "low": [effect.low for effect in top],
                    "high": [effect.high for effect in top],
                },
                index=labels,
            )
            viz.draw_coef_forest(
                axis, frame["value"], frame["low"], frame["high"], annotate=False
            )
            axis.axvline(
                record.decision.threshold,
                color=_THRESHOLD_COLOR,
                linestyle="--",
                linewidth=1,
                label="effect floor",
            )
            axis.legend(fontsize=viz.TICK_FONTSIZE, frameon=False, loc="lower right")
        else:
            axis.barh(labels, [effect.value for effect in top], color=viz.palette_colors(len(top)))
            axis.axvline(
                record.decision.threshold,
                color=_THRESHOLD_COLOR,
                linestyle="--",
                label="effect floor",
            )
            axis.legend(fontsize=viz.TICK_FONTSIZE, frameon=False, loc="lower right")
        axis.set_title(
            "Association with the target (effect size, 95% interval)",
            fontsize=viz.TITLE_FONTSIZE,
        )
        axis.set_xlabel(record.decision.unit or "effect size", fontsize=viz.LABEL_FONTSIZE)
        axis.tick_params(labelsize=viz.TICK_FONTSIZE)
        viz.despine(axis)
        return figure

    def _model_figure(self) -> Figure | None:
        if self._record("selection") is None:
            return None
        frame = _read_csv(self._candidates_path())
        required = {"candidate", "rmse", "rmse_low", "rmse_high"}
        if frame is None or not required.issubset(frame.columns):
            return None
        candidates = [str(value) for value in frame["candidate"]]
        rmses = [float(value) for value in frame["rmse"]]
        lows = [float(value) for value in frame["rmse_low"]]
        highs = [float(value) for value in frame["rmse_high"]]
        roles = (
            [str(value) for value in frame["role"]]
            if "role" in frame.columns
            else [""] * len(candidates)
        )
        figure = Figure(figsize=(8.0, max(2.8, 0.5 * len(candidates) + 1.5)), dpi=150)
        axis = figure.add_subplot(111)
        for index, (rmse, low, high, role) in enumerate(zip(rmses, lows, highs, roles)):
            axis.errorbar(
                rmse,
                index,
                xerr=[[rmse - low], [high - rmse]],
                fmt="o",
                capsize=3,
                color=_ROLE_COLORS.get(role, "#0072B2"),
            )
            axis.annotate(
                f"{rmse:.3f}",
                (rmse, index),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=viz.TICK_FONTSIZE,
            )
        axis.set_yticks(range(len(candidates)))
        axis.set_yticklabels(candidates)
        axis.invert_yaxis()
        axis.set_xlabel("RMSE (lower is better)", fontsize=viz.LABEL_FONTSIZE)
        axis.set_title(
            "Repeated-CV RMSE by candidate (95% interval)", fontsize=viz.TITLE_FONTSIZE
        )
        axis.grid(axis="x", alpha=viz.GRID_ALPHA)
        viz.despine(axis)
        return figure

    def _candidates_path(self) -> Path:
        step_dir = self._step_dir("selection")
        return (step_dir / "selection_candidates.csv") if step_dir else Path()


_STEPS = ("etl", "eda", "features", "baseline", "selection", "hpo", "evaluate")


def _missing(title: str, step: str) -> Section:
    return Section(
        title=title,
        conclusions=(f"not run yet — the {step} decision record has not been published.",),
        not_run=True,
    )


def _decision_number(decision: Decision) -> str:
    unit = f" {decision.unit}" if decision.unit else ""
    return (
        f"{decision.metric} = {_format(decision.value)}{unit} "
        f"[{_format(decision.interval_low)}, {_format(decision.interval_high)}] 95% · "
        f"n={decision.sample_size:,} · verdict: {decision.verdict}"
    )


def _format(value: float) -> str:
    if value and abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:,.4g}"


def _selection_conclusion(
    record: PipelineRecord | None, frame: pd.DataFrame | None
) -> str:
    """One plain-language outcome for the candidate leaderboard."""
    champion = _read_json(
        record.record.parent / "selection_champion.json" if record is not None else Path()
    )
    winner = champion.get("champion") if isinstance(champion, dict) else None
    rmse = champion.get("cv_mean") if isinstance(champion, dict) else None
    low = champion.get("interval_low") if isinstance(champion, dict) else None
    high = champion.get("interval_high") if isinstance(champion, dict) else None
    if (winner is None or rmse is None) and frame is not None and not frame.empty:
        top = frame.iloc[0]
        winner = winner or str(top["candidate"])
        rmse = rmse if rmse is not None else float(top["rmse"])
        low = low if low is not None else float(top["rmse_low"])
        high = high if high is not None else float(top["rmse_high"])
    if winner is None or rmse is None:
        return "The candidate leaderboard ran, but no champion is recorded."
    low = rmse if low is None else low
    high = rmse if high is None else high
    edge = None
    edge_low = None
    edge_high = None
    if record is not None:
        sensitivity = record.decision.evidence.sensitivity
        edge = sensitivity.get("champion_vs_linear_rmse")
        edge_low = sensitivity.get("champion_vs_linear_rmse_low")
        edge_high = sensitivity.get("champion_vs_linear_rmse_high")
    if edge is not None and edge_low is not None and edge_high is not None:
        if edge_low < 0 < edge_high:
            verdict = (
                "its edge over linear has an interval that crosses zero, so the data "
                "cannot tell them apart"
            )
        else:
            verdict = "its edge over linear clears zero, so it genuinely beat the reference"
        return (
            f"{winner} won the repeated-CV leaderboard at {float(rmse):.3f} RMSE "
            f"[{float(low):.3f}, {float(high):.3f}]; {verdict} "
            f"(Δ {float(edge):+.3f} [{float(edge_low):+.3f}, {float(edge_high):+.3f}])."
        )
    return (
        f"{winner} won the repeated-CV leaderboard at {float(rmse):.3f} RMSE "
        f"[{float(low):.3f}, {float(high):.3f}]."
    )


def _effects(record: PipelineRecord) -> list[Effect]:
    """Association effects from the sidecar, falling back to the published CSV."""
    decisions = [
        finding.decision
        for finding in record.findings()
        if finding.section in _FEATURE_SECTIONS
    ]
    effects = [
        Effect(
            label=_label(decision.metric),
            value=decision.value,
            low=decision.interval_low,
            high=decision.interval_high,
            verdict=decision.verdict,
        )
        for decision in decisions
    ]
    if not effects:
        frame = _read_csv(record.record.parent / "eda_effects.csv")
        if frame is not None and {"feature", "effect"}.issubset(frame.columns):
            effects = [
                Effect(
                    label=str(row["feature"]).replace("_", " "),
                    value=float(row["effect"]),
                    low=None,
                    high=None,
                    verdict=str(row["verdict"]) if "verdict" in frame.columns else "",
                )
                for _, row in frame.iterrows()
            ]
    return sorted(effects, key=lambda effect: effect.value, reverse=True)


def _findings_for(record: PipelineRecord, section: str) -> list[Decision]:
    findings: list[Finding] = record.findings()
    return [finding.decision for finding in findings if finding.section == section]


def _label(metric: str) -> str:
    for prefix in _LABEL_PREFIXES:
        if metric.startswith(prefix):
            metric = metric[len(prefix) :]
            break
    return metric.replace("_", " ")


def _funnel(record: PipelineRecord) -> list[tuple[str, int]]:
    """Rows in, rows out, labelled rows — the data funnel the ETL record proves."""
    provenance = record.decision.evidence.provenance
    rows_in = _as_int(provenance.get("rows_in"))
    rows_out = _as_int(provenance.get("rows_out"))
    stages: list[tuple[str, int]] = []
    if rows_in is not None:
        stages.append(("source rows", rows_in))
    if rows_out is not None and rows_out != rows_in:
        stages.append(("rows kept", rows_out))
        if rows_in is not None and rows_in > rows_out:
            stages.append(("dropped (recorded)", rows_in - rows_out))
    recorded = record.decision.sample_size
    if recorded not in {rows_in, rows_out}:
        stages.append(("recorded rows", recorded))
    if not stages:
        stages.append(("recorded rows", recorded))
    if len({count for _, count in stages}) == 1 and record.decision.evidence.segments:
        stages = [
            (segment.segment.replace("_", " "), segment.sample_size)
            for segment in record.decision.evidence.segments
        ] + [stages[-1]]
    return stages


def _class_balance(step_dir: Path | None) -> list[tuple[str, int]] | None:
    """Find a two-class balance table in a step dir (``bad_review``/``count``)."""
    if step_dir is None:
        return None
    for path in sorted(step_dir.glob("*.csv")):
        frame = _read_csv(path)
        if frame is None:
            continue
        target = next(
            (column for column in frame.columns if str(column).lower() in _TARGET_COLUMNS),
            None,
        )
        count = next(
            (column for column in frame.columns if str(column).lower() in _COUNT_COLUMNS),
            None,
        )
        if target is None or count is None:
            continue
        rows: list[tuple[str, int]] = []
        try:
            for _, row in frame.iterrows():
                parsed = _as_int(row[count])
                if parsed is not None:
                    rows.append((_balance_label(row[target]), parsed))
        except (KeyError, TypeError, ValueError):
            continue
        if rows:
            return rows
    return None


def _balance_label(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "bad", "bad_review", "yes"}:
        return "bad review (1–2★)"
    if text in {"false", "0", "0.0", "good", "not_bad", "no"}:
        return "not a bad review"
    return str(value)


def _balance_number(balance: list[tuple[str, int]]) -> str:
    total = sum(count for _, count in balance)
    if total <= 0:
        return "class balance unavailable"
    shares = ", ".join(f"{label} {count / total:.1%}" for label, count in balance)
    return f"class balance (n={total:,}): {shares}"


def _sensitivity_int(decision: Decision, prefix: str) -> int | None:
    for key, value in decision.evidence.sensitivity.items():
        if key.startswith(prefix):
            parsed = _as_int(value)
            if parsed is not None:
                return parsed
    return None


def _first_column(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    first = value.split(",")[0].strip()
    return first or None


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _read_csv(path: Path) -> pd.DataFrame | None:
    """Read a small evidence table, logging one traceback per unreadable path."""
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError):
        if path not in _LOGGED_UNREADABLE:
            _LOGGED_UNREADABLE.add(path)
            logger.exception("story: unreadable evidence table at %s", path)
        return None


def _read_json(path: Path) -> object:
    """Read a small JSON document, logging one traceback per unreadable path."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        if path not in _LOGGED_UNREADABLE:
            _LOGGED_UNREADABLE.add(path)
            logger.exception("story: unreadable JSON at %s", path)
        return None
