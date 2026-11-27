"""The project story: one human page generated from the committed records.

A busy human should not have to read the pipeline to understand the project.
This module turns the published ``results/pipeline/<step>/`` decision records,
their feature-decision sidecars and their provenance-named evidence tables into
a single narrative: the question, the data, what the EDA found, what was changed
and why, the features, the model chain, and the honest limits. Each section puts
the conclusion first, then one graph, then the numbers (value, 95% interval, n,
verdict) and a link to the record it came from.

Nothing here is a hardcoded result. A stage with no record renders a short
"not run yet" line, so the page fills in as the pipeline lands. A malformed
record is skipped by :class:`~broadway.reports.pipeline_evidence.PipelineSeries`
with one logged traceback; this module never raises for bad evidence. Evidence
sidecars are accepted only when their declared dataset matches the story's
dataset, and evidence tables are read only from a path the record's provenance
names — a stale table from another dataset can never leak into the story.

The dataset name and the project framing (question, target, source table count,
exclusions, caveats) belong to the project layer: ``project/dashboard.py`` passes
a :class:`StoryConfig` through :func:`configure_story`. The platform fallback is
neutral, so ``src/broadway`` stays data-agnostic.

Every section that makes a **data claim** carries a figure: the question's
decision boundary, the data funnel, the EDA effect sizes, the feature surface,
the cleaning funnel, and the model intervals. "What we learned / what is next"
is a next-steps section: it carries the numbers and record links, deliberately
without a decorative chart. That split is the acceptance criterion, and the
tests pin it.

Graphs are drawn in memory with matplotlib's Agg backend and the shared
``broadway.viz`` helpers (deterministic, no new dependency, no scratch files);
the narrative-to-HTML frame and the routes live in
:mod:`broadway.reports.story_page`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field

from broadway.reports import story_figures as figures
from broadway.reports.decision import Decision
from broadway.reports.pipeline_evidence import (
    PipelineRecord,
    PipelineSeries,
    findings_from_sidecar,
)

logger = logging.getLogger(__name__)

STORY_TITLE = "The project story"

#: Optional env override of the configured dataset (an empty value defers to config).
STORY_DATASET_ENV = "BROADWAY_STORY_DATASET"

#: Neutral aliases for a target column; the project's own name comes from config.
_TARGET_COLUMN_ALIASES = frozenset({"target", "class", "label"})
_COUNT_COLUMNS = frozenset({"count", "n", "orders", "rows", "order_count"})
_FEATURE_SECTIONS = ("features", "associations")
_VERDICT_LABELS = (
    ("act", "act"),
    ("investigate", "investigate"),
    ("no_action", "no action"),
    ("insufficient", "insufficient"),
)
_LABEL_PREFIXES = ("association_", "outlier_share_", "standardized_shift_", "shift_")
_STEPS = ("etl", "eda", "features", "baseline", "selection", "hpo", "evaluate")
# A permanently malformed file is skipped on every request; log its full
# traceback once so the contract's traceback rule does not become log bloat.
_LOGGED_UNREADABLE: set[Path] = set()


class StoryConfig(BaseModel):
    """The project's framing of the story (the platform fallback is neutral)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str | None = None
    question: str = Field(min_length=1)
    why: str = Field(min_length=1)
    target_label: str = Field(min_length=1)
    target_rule: str = Field(min_length=1)
    target_column: str = Field(default="target", min_length=1)
    decision_moment: str = Field(min_length=1)
    source_name: str = ""
    source_tables: int | None = Field(default=None, gt=0)
    post_purchase_note: str = Field(min_length=1)
    excluded_note: str = Field(min_length=1)
    caveats: tuple[str, ...] = ()

    @classmethod
    def fallback(cls) -> StoryConfig:
        """A neutral framing for a checkout with no project configuration."""
        return cls(
            dataset=None,
            question="Which rows carry the outcome the project predicts?",
            why=(
                "The project predicts an outcome from the information available at "
                "decision time."
            ),
            target_label="outcome",
            target_rule="the configured target",
            target_column="target",
            decision_moment="decision time",
            source_name="",
            source_tables=None,
            post_purchase_note=(
                "Columns known only after the outcome are context only; the feature step "
                "keeps decision-time information only."
            ),
            excluded_note=(
                "Columns known only after the outcome are excluded from the features; using "
                "them would be leakage."
            ),
            caveats=(),
        )


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


@dataclass(frozen=True)
class Candidate:
    """One leaderboard row: name, point RMSE and its 95% interval."""

    name: str
    rmse: float
    low: float
    high: float
    role: str


class Story:
    """Build the story sections and figures from one pipeline results tree."""

    def __init__(
        self, pipeline: PipelineSeries, config: StoryConfig | None = None
    ) -> None:
        self._pipeline = pipeline
        self._config = config if config is not None else StoryConfig.fallback()
        self.dataset = os.environ.get(STORY_DATASET_ENV) or self._config.dataset

    @classmethod
    def from_experiments_root(
        cls, experiments_root: Path, config: StoryConfig | None = None
    ) -> Story:
        """Build the story for a checkout's experiments root."""
        return cls(PipelineSeries.from_experiments_root(experiments_root), config)

    @property
    def question(self) -> str:
        """The one-line question the whole page answers."""
        return self._config.question

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

    def _findings(self, record: PipelineRecord, section: str) -> list[Decision]:
        """Sidecar conclusions for one section, only from this dataset's sidecar."""
        payload = _read_json(record.sidecar)
        if not isinstance(payload, dict):
            return []
        declared = payload.get("dataset")
        if self.dataset is not None and declared != self.dataset:
            return []
        return [
            finding.decision
            for finding in findings_from_sidecar(record.sidecar)
            if finding.section == section
        ]

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
        config = self._config
        records = (("ETL record", "etl"),) if self._record("etl") is not None else ()
        return Section(
            title="The question",
            conclusions=(config.why,),
            numbers=(
                f"{config.target_label} := {config.target_rule}",
                f"decision moment := {config.decision_moment}",
            ),
            figure="question",
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
        name = f" {self._config.source_name}" if self._config.source_name else ""
        source = (
            f"{self._config.source_tables}{name} source tables"
            if self._config.source_tables
            else "The source tables"
        )
        conclusions = [
            (
                f"{source} collapse to one row per order. Orders without a target are "
                "dropped with a recorded reason, never imputed."
            )
        ]
        if rows_in is not None and rows_out is not None:
            dropped = rows_in - rows_out
            detail = f" ({dropped:,} dropped)" if dropped else ""
            source_suffix = " source orders" if decision.sample_size == rows_in else ""
            n_phrase = f"{decision.sample_size:,}{source_suffix}"
            conclusions.append(
                f"The record's n is {n_phrase}; the join kept {rows_out:,} rows after the "
                f"accounted drops{detail}."
            )
        else:
            conclusions.append(
                f"The order-grain join produced n={decision.sample_size:,} rows."
            )
        conclusions.append(f"The record asks: {decision.question}")
        conclusions.append(f"Verdict: {decision.verdict} — {decision.action}.")
        numbers = [_decision_number(decision)]
        balance = self._balance(record)
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
        decision = record.decision
        provenance = decision.evidence.provenance
        effects = self._effects(record)
        outliers = self._findings(record, "outliers")
        comparison = self._findings(record, "comparison")
        conclusions = [
            (
                f"The EDA record reports {decision.verdict} on {decision.metric} "
                f"({_format(decision.value)})."
            )
        ]
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
                "The per-feature evidence table has not been published yet, so the "
                "association ranking is not narrated."
            )
        noise = len(effects) - len(acts)
        if acts and noise:
            verb = "behaviour sits" if noise == 1 else "behaviours sit"
            is_are = "is" if noise == 1 else "are"
            conclusions.append(f"{noise} further {verb} at or below the floor and {is_are} noise.")
        if effects:
            conclusions.append(self._config.post_purchase_note)
        if outliers:
            flagged = [item for item in outliers if item.verdict == "act"]
            names = ", ".join(_label(item.metric) for item in flagged[:3])
            suffix = f" ({names})" if names else ""
            conclusions.append(
                f"{len(flagged)} of {len(outliers)} profiled columns are heavy-tailed{suffix}; "
                "the data was flagged, not silently trimmed."
            )
        if comparison:
            conclusions.append(_shift_note(max(comparison, key=lambda item: abs(item.value))))
        tested = _as_int(provenance.get("family_size"))
        decided = len(effects) or _decided_from_provenance(provenance)
        verdict_counts = _verdict_counts(effects, provenance)
        if tested is not None and decided and tested != decided:
            conclusions.append(
                f"The BH correction family is {tested} features; only {decided} crossed "
                "the reporting threshold."
            )
        numbers = [_decision_number(decision)]
        if decided:
            numbers.append(_association_count(tested, decided, verdict_counts))
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
            conclusions.append(self._config.post_purchase_note)
            numbers.append(_decision_number(etl.decision))
            records.append(("ETL record", "etl"))
        if eda is not None:
            outliers = self._findings(eda, "outliers")
            comparison = self._findings(eda, "comparison")
            flagged = [item for item in outliers if item.verdict == "act"]
            if flagged:
                conclusions.append(
                    f"{len(flagged)} of {len(outliers)} candidate columns are heavy-tailed. "
                    f"Outlier decision: {flagged[0].action}."
                )
            elif outliers:
                conclusions.append(
                    f"{len(outliers)} candidate columns were profiled; none cleared the "
                    "outlier threshold."
                )
            if comparison:
                conclusions.append(
                    _shift_note(max(comparison, key=lambda item: abs(item.value)))
                )
            if flagged:
                numbers.append(f"outlier action: {flagged[0].action}")
            records.append(("EDA record", "eda"))
        if etl is not None:
            conclusions.append(
                "Trade-off: rows are dropped only with a recorded reason, so the model "
                "inherits that selection honestly instead of seeing a silently repaired table."
            )
        return Section(
            title="What we changed and why",
            conclusions=tuple(conclusions),
            numbers=tuple(numbers),
            figure="change" if etl is not None else None,
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
                "The feature step kept every row and built decision-time features only."
            ]
        if added is not None:
            conclusions.append(f"{added} features were added.")
        if example is not None:
            conclusions.append(f"One example: {example}.")
        conclusions.append(self._config.excluded_note)
        stages = _feature_stages(decision)
        if stages:
            conclusions.append(
                "Feature surface: "
                + ", ".join(f"{label} {count:,}" for label, count in stages)
                + "."
            )
        return Section(
            title="The features",
            conclusions=tuple(conclusions),
            numbers=(_decision_number(decision),),
            figure="features" if stages else None,
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
        candidates = _candidate_rows(candidate_frame, self._candidates_path())
        if selection is not None:
            conclusions.append(_selection_conclusion(selection, candidates))
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
            figure="model" if selection is not None and candidates else None,
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
        caveats = self._config.caveats or (
            "the target is observational",
            "the features are decision-time only",
            "this is prediction, not causation",
        )
        limits = "; ".join(caveats)
        if record is not None:
            conclusion = (
                f"The result is bounded by: {limits}. The next decision: {next_action}."
            )
        else:
            conclusion = (
                f"No model has run yet, so there is no result to bound. The design limits: "
                f"{limits}. The next decision: {next_action}."
            )
        numbers: list[str] = []
        records: tuple[tuple[str, str], ...] = ()
        if evaluate is not None:
            numbers.append(f"holdout · {_decision_number(evaluate.decision)}")
            records = (("holdout record", "evaluate"),)
        elif selection is not None:
            numbers.append(f"selection · {_decision_number(selection.decision)}")
            records = (("selection record", "selection"),)
        return Section(
            title="What we learned / what is next",
            conclusions=(conclusion,),
            numbers=tuple(numbers),
            records=records,
        )

    # -- figures -----------------------------------------------------------

    def figure_png(self, name: str) -> bytes | None:
        """Render one story figure to PNG bytes, or None for an unknown/absent figure."""
        builders = {
            "question": self._question_figure,
            "data": self._data_figure,
            "eda": self._eda_figure,
            "features": self._features_figure,
            "change": self._change_figure,
            "model": self._model_figure,
        }
        build = builders.get(name)
        figure = build() if build is not None else None
        return None if figure is None else figures.to_png(figure)

    # -- figures -----------------------------------------------------------

    def _question_figure(self) -> Figure:
        return figures.question_figure(
            self._config.decision_moment, self._config.target_label
        )

    def _data_figure(self) -> Figure | None:
        record = self._record("etl")
        if record is None:
            return None
        return figures.join_figure(
            _funnel(record), "Rows through the order-grain join", self._balance(record)
        )

    def _change_figure(self) -> Figure | None:
        record = self._record("etl")
        if record is None:
            return None
        return figures.join_figure(
            _funnel(record), "Rows affected by the cleaning policy", None
        )

    def _eda_figure(self) -> Figure | None:
        record = self._record("eda")
        if record is None:
            return None
        effects = self._effects(record)
        rows = [
            (effect.label, effect.value, effect.low, effect.high) for effect in effects
        ]
        return figures.eda_figure(record.decision.threshold, record.decision.unit, rows)

    def _features_figure(self) -> Figure | None:
        record = self._record("features")
        if record is None:
            return None
        stages = _feature_stages(record.decision)
        return figures.features_figure(stages, "Feature surface") if stages else None

    def _model_figure(self) -> Figure | None:
        if self._record("selection") is None:
            return None
        rows = _candidate_rows(_read_csv(self._candidates_path()), self._candidates_path())
        if not rows:
            return None
        return figures.model_figure(
            [row.name for row in rows],
            [row.rmse for row in rows],
            [row.low for row in rows],
            [row.high for row in rows],
            [row.role for row in rows],
        )

    def _effects(self, record: PipelineRecord) -> list[Effect]:
        """Association effects from the dataset-tagged sidecar, ranked."""
        decisions = [
            decision
            for section in _FEATURE_SECTIONS
            for decision in self._findings(record, section)
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
        return sorted(effects, key=lambda effect: effect.value, reverse=True)

    def _candidates_path(self) -> Path:
        step_dir = self._step_dir("selection")
        return (step_dir / "selection_candidates.csv") if step_dir else Path()

    def _balance(self, record: PipelineRecord) -> list[tuple[str, int]] | None:
        """A balance table only when the record's provenance names the CSV."""
        for value in record.decision.evidence.provenance.values():
            if not isinstance(value, str) or not value.lower().endswith(".csv"):
                continue
            frame = _read_csv(record.record.parent / Path(value).name)
            if frame is None:
                continue
            rows = self._long_balance(frame)
            if rows is None:
                rows = self._wide_balance(frame)
            if rows:
                return rows
        return None

    def _long_balance(self, frame: pd.DataFrame) -> list[tuple[str, int]] | None:
        """Long form: one row per class, with a target label and a count column."""
        target = self._target_column(frame)
        count = next(
            (column for column in frame.columns if str(column).lower() in _COUNT_COLUMNS),
            None,
        )
        if target is None or count is None:
            return None
        rows: list[tuple[str, int]] = []
        try:
            for _, row in frame.iterrows():
                parsed = _as_int(row[count])
                if parsed is not None:
                    rows.append((self._balance_label(row[target]), parsed))
        except (KeyError, TypeError, ValueError):
            return None
        return rows or None

    def _wide_balance(self, frame: pd.DataFrame) -> list[tuple[str, int]]:
        """Wide form: one row whose columns are the class labels and counts."""
        if len(frame) != 1:
            return []
        rows: list[tuple[str, int]] = []
        for column in frame.columns:
            parsed = _as_int(frame.iloc[0][column])
            if parsed is not None:
                rows.append((self._balance_label(column), parsed))
        return rows

    def _target_column(self, frame: pd.DataFrame) -> str | None:
        if self._config.target_column in frame.columns:
            return self._config.target_column
        return next(
            (
                column
                for column in frame.columns
                if str(column).lower() in _TARGET_COLUMN_ALIASES
            ),
            None,
        )

    def _balance_label(self, value: object) -> str:
        text = str(value).strip().lower()
        if text in {"true", "1", "1.0", "yes"}:
            return self._config.target_label
        if text in {"false", "0", "0.0", "no"}:
            return f"not {self._config.target_label}"
        label = str(value).replace("_", " ").strip()
        return label or str(value)


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


def _shift_note(decision: Decision) -> str:
    """One honest before/after sentence; never call a one-sided shift zero."""
    label = _label(decision.metric)
    if decision.interval_low <= 0 <= decision.interval_high:
        return (
            f"Before/after shift for {label} is consistent with no change: "
            f"{_format(decision.value)} [{_format(decision.interval_low)}, "
            f"{_format(decision.interval_high)}] covers zero."
        )
    return (
        f"Before/after shift for {label} is one-sided: {_format(decision.value)} "
        f"[{_format(decision.interval_low)}, {_format(decision.interval_high)}] does not "
        "cover zero, so the shift is reported rather than called zero."
    )


def _candidate_rows(frame: pd.DataFrame | None, path: Path) -> list[Candidate]:
    """Parse leaderboard rows, skipping malformed ones with one traceback per file."""
    if frame is None:
        return []
    required = {"candidate", "rmse", "rmse_low", "rmse_high"}
    if not required.issubset(frame.columns):
        return []
    rows: list[Candidate] = []
    for _, row in frame.iterrows():
        try:
            rows.append(
                Candidate(
                    name=str(row["candidate"]),
                    rmse=float(row["rmse"]),
                    low=float(row["rmse_low"]),
                    high=float(row["rmse_high"]),
                    role=str(row["role"]) if "role" in frame.columns else "",
                )
            )
        except (TypeError, ValueError):
            if path not in _LOGGED_UNREADABLE:
                _LOGGED_UNREADABLE.add(path)
                logger.exception("story: malformed candidate row in %s", path)
    return rows


def _selection_conclusion(
    record: PipelineRecord | None, candidates: list[Candidate]
) -> str:
    """One plain-language outcome for the candidate leaderboard."""
    champion = _read_json(
        record.record.parent / "selection_champion.json" if record is not None else Path()
    )
    winner = champion.get("champion") if isinstance(champion, dict) else None
    rmse = champion.get("cv_mean") if isinstance(champion, dict) else None
    low = champion.get("interval_low") if isinstance(champion, dict) else None
    high = champion.get("interval_high") if isinstance(champion, dict) else None
    if (winner is None or rmse is None) and candidates:
        top = candidates[0]
        winner = winner or top.name
        rmse = rmse if rmse is not None else top.rmse
        low = low if low is not None else top.low
        high = high if high is not None else top.high
    if winner is None or rmse is None:
        return (
            "The candidate evidence table has not been published, so the leaderboard is "
            "not narrated."
        )
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


def _label(metric: str) -> str:
    for prefix in _LABEL_PREFIXES:
        if metric.startswith(prefix):
            metric = metric[len(prefix) :]
            break
    return metric.replace("_", " ")


def _funnel(record: PipelineRecord) -> list[tuple[str, int]]:
    """Rows in, rows out, rows dropped — the data funnel the ETL record proves."""
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


def _balance_number(balance: list[tuple[str, int]]) -> str:
    total = sum(count for _, count in balance)
    if total <= 0:
        return "class balance unavailable"
    counts = ", ".join(f"{label} {count:,}" for label, count in balance)
    return f"class balance (n={total:,}): {counts}"


def _decided_from_provenance(provenance: dict[str, str]) -> int:
    """Features the record decided, from its acted + investigated counts."""
    acted = _as_int(provenance.get("acted")) or 0
    investigated = _as_int(provenance.get("investigated")) or 0
    return acted + investigated


def _verdict_counts(effects: list[Effect], provenance: dict[str, str]) -> dict[str, int]:
    """Verdict tally from the evidence rows, or the record's counts when it has none."""
    counts: dict[str, int] = {}
    for effect in effects:
        counts[effect.verdict] = counts.get(effect.verdict, 0) + 1
    if effects:
        return counts
    acted = _as_int(provenance.get("acted"))
    investigated = _as_int(provenance.get("investigated"))
    if acted is not None:
        counts["act"] = acted
    if investigated is not None:
        counts["investigate"] = investigated
    return counts


def _association_count(
    tested: int | None, decided: int, counts: dict[str, int]
) -> str:
    """``21 tested · 12 decided (11 act, 1 investigate)`` — never the row count as tested."""
    parts = ", ".join(
        f"{counts[key]} {label}" for key, label in _VERDICT_LABELS if counts.get(key)
    )
    prefix = f"{tested} tested · " if tested is not None else ""
    suffix = f" ({parts})" if parts else ""
    return f"{prefix}{decided} decided{suffix}"


def _feature_stages(decision: Decision) -> list[tuple[str, int]]:
    """Feature-surface counts the record publishes, in narrative order."""
    stages: list[tuple[str, int]] = []
    for label, prefix in (
        ("added", "features_added"),
        ("removed", "features_removed"),
        ("plan columns", "plan_columns"),
    ):
        value = _sensitivity_int(decision, prefix)
        if value is not None:
            stages.append((label, value))
    return stages


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
