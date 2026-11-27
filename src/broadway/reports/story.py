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
without a decorative chart. "The step map" is a status section (done / not run
yet), not a data claim, so it carries no chart either. That split is the
acceptance criterion, and the tests pin it.

Graphs are drawn in memory with matplotlib's Agg backend and the shared
``broadway.viz`` helpers (deterministic, no new dependency, no scratch files);
the narrative-to-HTML frame and the routes live in
:mod:`broadway.reports.story_page`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from broadway.reports import story_figures as figures
from broadway.reports.decision import Decision
from broadway.reports.pipeline_evidence import (
    PipelineRecord,
    PipelineSeries,
    findings_from_sidecar,
)
from broadway.reports.story_models import (
    _VERDICT_LABELS,
    Candidate,
    Effect,
    Section,
    StoryConfig,
    _as_int,
    _association_count,
    _balance_number,
    _candidate_rows,
    _decided_from_provenance,
    _decision_number,
    _effect_label,
    _effect_stats,
    _feature_key,
    _feature_stages,
    _first_column,
    _format,
    _funnel,
    _label,
    _leaderboard_metric,
    _lower_is_better,
    _missing,
    _read_csv,
    _read_json,
    _selection_conclusion,
    _sensitivity_int,
    _shift_note,
    _significant_count,
    _verdict_counts,
)

STORY_TITLE = "The project story"

#: Optional env override of the configured dataset (an empty value defers to config).
STORY_DATASET_ENV = "BROADWAY_STORY_DATASET"

#: Neutral aliases for a target column; the project's own name comes from config.
_TARGET_COLUMN_ALIASES = frozenset({"target", "class", "label"})
_COUNT_COLUMNS = frozenset({"count", "n", "orders", "rows", "order_count"})
#: Balance-table buckets that are not target classes (pre-ETL drops, not classes).
_MISSING_LABELS = frozenset(
    {"missing", "missing_target", "unlabeled", "unlabelled", "unknown", "null", "none", "na"}
)
_FEATURE_SECTIONS = ("features", "associations")
_STEPS = ("etl", "eda", "features", "baseline", "selection", "hpo", "evaluate")

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
        resolved = config if config is not None else StoryConfig.fallback()
        pipeline = PipelineSeries.from_experiments_root(
            experiments_root,
            configured=resolved.steps.configured,
            planned=resolved.steps.planned,
        )
        return cls(pipeline, resolved)

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
        """All eight sections, in narrative order (missing stages say so)."""
        return [
            self._question(),
            self._step_map(),
            self._data(),
            self._eda(),
            self._changed(),
            self._features(),
            self._model(),
            self._learned(),
        ]

    def _step_map(self) -> Section:
        """The project's configured steps and planned lanes, with per-step status."""
        configured = self._config.steps.configured
        planned = self._config.steps.planned
        extras = [
            step
            for step in self._pipeline.steps()
            if step not in configured and step not in planned
        ]
        if not configured and not planned and not extras:
            return Section(
                title="The step map",
                conclusions=("No pipeline steps are published yet.",),
                not_run=True,
            )
        done = [step for step in configured if self._record(step) is not None]
        planned_pending = [step for step in planned if self._record(step) is None]
        conclusions = [
            (
                f"{len(done)} of {len(configured)} configured steps done; "
                f"{len(planned_pending)} planned lanes not configured."
            )
        ]
        def label(step: str, is_planned: bool) -> str:
            has_record = self._record(step) is not None
            status = self._pipeline.status(step, has_record=has_record)
            if status.kind == "record":
                return "done (planned lane)" if is_planned else "done"
            return status.text

        numbers: list[str] = []
        for step in configured:
            numbers.append(f"{step} — {label(step, False)}")
        for step in planned:
            numbers.append(f"{step} — {label(step, True)}")
        for step in extras:
            numbers.append(f"{step} — {label(step, False)}")
        return Section(title="The step map", conclusions=tuple(conclusions), numbers=tuple(numbers))

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
            numbers.append(_balance_number(balance, rows_in))
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
        context = [effect for effect in effects if effect.verdict != "act"]
        if len(acts) >= 2 and len(context) >= 2 and context[0].value > acts[0].value:
            conclusions.append(
                f"In the purchase-time feature ranking, {acts[0].label} and {acts[1].label} "
                f"lead ({acts[0].value:.2f} and {acts[1].value:.2f}); {context[0].label} and "
                f"{context[1].label} are the strongest context associations "
                f"({context[0].value:.2f} and {context[1].value:.2f})."
            )
        elif len(acts) >= 2:
            conclusions.append(
                f"{acts[0].label} and {acts[1].label} lead the association ranking "
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
        verdicts = _verdict_counts(effects, provenance)
        biased = [effect for effect in effects if effect.bootstrap_biased]
        biased_investigate = sum(1 for effect in biased if effect.verdict == "investigate")
        parts = []
        for key, label in _VERDICT_LABELS:
            if verdicts.get(key):
                suffix = (
                    f" ({biased_investigate} bootstrap-biased)"
                    if key == "investigate" and biased_investigate
                    else ""
                )
                parts.append(f"{verdicts[key]} {label}{suffix}")
        if parts:
            conclusions.append("Verdicts: " + ", ".join(parts) + ".")
        if biased:
            names = ", ".join(effect.label for effect in biased)
            conclusions.append(
                f"{names}: bootstrap-biased, interval not centred on the point; kept in "
                "the decision record, not the figure."
            )
        if effects:
            conclusions.append(self._config.post_purchase_note)
        if outliers:
            flagged = [item for item in outliers if item.verdict == "act"]
            names = ", ".join(_label(item.metric) for item in flagged[:3])
            suffix = f" ({names})" if names else ""
            conclusions.append(
                f"{len(flagged)} of {len(outliers)} numeric feature columns are "
                f"heavy-tailed{suffix}; the data was flagged, not silently trimmed."
            )
        if comparison:
            conclusions.append(_shift_note(max(comparison, key=lambda item: abs(item.value))))
        tested = _as_int(provenance.get("family_size"))
        significant = _as_int(provenance.get("significant"))
        if significant is None:
            significant = _significant_count(self._effects_table(record), tested)
        decided = len(effects) or _decided_from_provenance(provenance)
        verdict_counts = _verdict_counts(effects, provenance)
        if tested is not None and decided and tested != decided:
            if significant is not None:
                conclusions.append(
                    f"The BH correction family is {tested} features; {significant} are "
                    f"significant and the top {decided} by effect are ranked."
                )
            else:
                conclusions.append(
                    f"The BH correction family is {tested} features; the top {decided} by "
                    "effect are ranked."
                )
        numbers = [_decision_number(decision)]
        if decided:
            numbers.append(
                _association_count(tested, decided, verdict_counts, significant)
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
            conclusions.append(self._config.post_purchase_note)
            numbers.append(_decision_number(etl.decision))
            records.append(("ETL record", "etl"))
        if eda is not None:
            outliers = self._findings(eda, "outliers")
            comparison = self._findings(eda, "comparison")
            flagged = [item for item in outliers if item.verdict == "act"]
            if flagged:
                conclusions.append(
                    f"{len(flagged)} of {len(outliers)} numeric feature columns are "
                    "heavy-tailed. "
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
        candidates: list[Candidate] = []
        if selection is not None:
            metric = _leaderboard_metric(selection.decision)
            candidates = _candidate_rows(
                _read_csv(self._candidates_path()), self._candidates_path(), metric
            )
            conclusions.append(_selection_conclusion(selection, candidates, metric))
            numbers.append(f"selection · {_decision_number(selection.decision)}")
            records.append(("selection", "selection"))
        else:
            conclusions.append("Candidate selection: not run yet.")
        if hpo is not None:
            decision = hpo.decision
            worth = decision.interval_low > decision.threshold
            conclusions.append(
                f"HPO moved CV {_leaderboard_metric(decision)} by {decision.value:.2%} "
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
        plot_effects = [effect for effect in effects if not effect.bootstrap_biased]
        if not plot_effects:
            return None
        calibrated = all(
            effect.null_threshold is not None
            and effect.low is not None
            and effect.high is not None
            for effect in plot_effects
        )
        if calibrated:
            rows: list[tuple[str, float, float | None, float | None]] = []
            for effect in plot_effects:
                assert effect.null_threshold is not None
                assert effect.low is not None and effect.high is not None
                null = effect.null_threshold
                rows.append(
                    (
                        _effect_label(effect),
                        effect.value - null,
                        effect.low - null,
                        effect.high - null,
                    )
                )
            return figures.eda_figure(
                rows,
                0.0,
                "effect minus its own shuffle null",
                "own-null baseline",
                "Association with the target (calibrated margin)",
            )
        rows = [
            (_effect_label(effect), effect.value, effect.low, effect.high)
            for effect in plot_effects
        ]
        return figures.eda_figure(
            rows,
            record.decision.threshold,
            record.decision.unit or "effect size",
            "effect floor",
            "Association with the target (effect size, 95% interval)",
        )

    def _features_figure(self) -> Figure | None:
        record = self._record("features")
        if record is None:
            return None
        stages = _feature_stages(record.decision)
        return figures.features_figure(stages, "Feature surface") if stages else None

    def _model_figure(self) -> Figure | None:
        record = self._record("selection")
        if record is None:
            return None
        metric = _leaderboard_metric(record.decision)
        rows = _candidate_rows(
            _read_csv(self._candidates_path()), self._candidates_path(), metric
        )
        if not rows:
            return None
        return figures.model_figure(
            [row.name for row in rows],
            [row.value for row in rows],
            [row.low for row in rows],
            [row.high for row in rows],
            [row.role for row in rows],
            metric,
            _lower_is_better(metric),
        )

    def _effects(self, record: PipelineRecord) -> list[Effect]:
        """Association effects from the dataset-tagged sidecar, ranked."""
        stats = _effect_stats(self._effects_table(record))
        decisions = [
            decision
            for section in _FEATURE_SECTIONS
            for decision in self._findings(record, section)
        ]
        effects = []
        for decision in decisions:
            statistic, null, table_biased = stats.get(
                _feature_key(decision.metric), (None, None, False)
            )
            value = decision.value
            low = decision.interval_low
            high = decision.interval_high
            biased = table_biased or value < low or value > high
            effects.append(
                Effect(
                    label=_label(decision.metric),
                    value=value,
                    low=low,
                    high=high,
                    verdict=decision.verdict,
                    statistic=statistic,
                    null_threshold=(
                        null if null is not None else decision.null_threshold
                    ),
                    bootstrap_biased=biased,
                )
            )
        return sorted(effects, key=lambda effect: effect.value, reverse=True)

    def _effects_table(self, record: PipelineRecord) -> pd.DataFrame | None:
        """The EDA effects table, only when it matches the record's tested family."""
        table = _read_csv(record.record.parent / "eda_effects.csv")
        if table is None:
            return None
        family = _as_int(record.decision.evidence.provenance.get("family_size"))
        return table if family is None or len(table) == family else None

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
                if str(row[target]).strip().lower() in _MISSING_LABELS:
                    continue
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
            if str(column).strip().lower() in _MISSING_LABELS:
                continue
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


