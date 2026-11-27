"""Story models and evidence helpers: config, section/candidate shapes, parsing.

These are the pure shapes and evidence helpers shared by the narrative and the
page. Nothing here renders HTML or draws a figure; the story module and the
page/figures modules own those concerns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from broadway.reports.decision import Decision
from broadway.reports.pipeline_evidence import PipelineRecord

logger = logging.getLogger(__name__)

_VERDICT_LABELS = (
    ("act", "act"),
    ("investigate", "investigate"),
    ("no_action", "no action"),
    ("insufficient", "insufficient"),
)
_LABEL_PREFIXES = ("association_", "outlier_share_", "standardized_shift_", "shift_")
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
    """One association: its own statistic, point estimate, interval and shuffle null."""

    label: str
    value: float
    low: float | None
    high: float | None
    verdict: str
    statistic: str | None = None
    null_threshold: float | None = None
    bootstrap_biased: bool = False


@dataclass(frozen=True)
class Candidate:
    """One leaderboard row: name, metric, point value and its 95% interval."""

    name: str
    metric: str
    value: float
    low: float
    high: float
    role: str


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
    """One honest before/after sentence: a shift is real only when the record confirms it."""
    label = _label(decision.metric)
    null = decision.null_threshold
    confirmed = decision.verdict == "act" or (
        null is not None and decision.interval_low > null
    )
    interval = (
        f"{_format(decision.value)} [{_format(decision.interval_low)}, "
        f"{_format(decision.interval_high)}]"
    )
    if not confirmed:
        return (
            f"Before/after shift for {label} is consistent with no material change "
            f"({interval}; verdict: {decision.verdict})."
        )
    return (
        f"Before/after shift for {label} clears its null band: {interval} "
        f"(verdict: {decision.verdict})."
    )


def _leaderboard_metric(decision: Decision) -> str:
    """The candidate metric, read from ``improvement_over_naive_<metric>``."""
    metric = decision.metric
    for prefix in ("improvement_over_naive_", "improvement_over_default_"):
        metric = metric.removeprefix(prefix)
    return _metric_label(metric)


def _metric_label(metric: str) -> str:
    return metric.replace("_", " ").strip().upper()


def _lower_is_better(metric: str) -> bool:
    return metric.lower().replace(" ", "_") in {
        "rmse",
        "mae",
        "mape",
        "mse",
        "log_loss",
        "logloss",
        "error",
    }


def _value_columns(
    frame: pd.DataFrame | None, metric: str
) -> tuple[str, str, str] | None:
    """The value/low/high column trio: generic, metric-named, then the rmse legacy."""
    if frame is None:
        return None
    key = metric.lower().replace(" ", "_")
    for base in ("value", key, "rmse"):
        if base and {base, f"{base}_low", f"{base}_high"} <= set(frame.columns):
            return base, f"{base}_low", f"{base}_high"
    return None


def _candidate_rows(
    frame: pd.DataFrame | None, path: Path, metric: str
) -> list[Candidate]:
    """Parse leaderboard rows, skipping malformed ones with one traceback per file."""
    columns = _value_columns(frame, metric)
    if frame is None or columns is None:
        return []
    value_column, low_column, high_column = columns
    rows: list[Candidate] = []
    for _, row in frame.iterrows():
        try:
            value = _as_float(row[value_column])
            low = _as_float(row[low_column])
            high = _as_float(row[high_column])
            if value is None or low is None or high is None:
                raise ValueError("candidate value is not numeric")
            row_metric = str(row["metric"]) if "metric" in frame.columns else metric
            rows.append(
                Candidate(
                    name=str(row["candidate"]),
                    metric=_metric_label(row_metric),
                    value=value,
                    low=low,
                    high=high,
                    role=str(row["role"]) if "role" in frame.columns else "",
                )
            )
        except (KeyError, TypeError, ValueError):
            if path not in _LOGGED_UNREADABLE:
                _LOGGED_UNREADABLE.add(path)
                logger.exception("story: malformed candidate row in %s", path)
    return rows


def _selection_conclusion(
    record: PipelineRecord | None, candidates: list[Candidate], metric: str
) -> str:
    """One plain-language outcome for the candidate leaderboard."""
    champion = _read_json(
        record.record.parent / "selection_champion.json" if record is not None else Path()
    )
    winner = champion.get("champion") if isinstance(champion, dict) else None
    value = _as_float(champion.get("cv_mean")) if isinstance(champion, dict) else None
    low = _as_float(champion.get("interval_low")) if isinstance(champion, dict) else None
    high = _as_float(champion.get("interval_high")) if isinstance(champion, dict) else None
    if (winner is None or value is None) and candidates:
        top = candidates[0]
        winner = winner or top.name
        value = value if value is not None else top.value
        low = low if low is not None else top.low
        high = high if high is not None else top.high
    if winner is None or value is None:
        return (
            "The candidate evidence table has not been published, so the leaderboard is "
            "not narrated."
        )
    low = value if low is None else low
    high = value if high is None else high
    edge = None
    edge_low = None
    edge_high = None
    if record is not None:
        sensitivity = record.decision.evidence.sensitivity
        key = next(
            (
                name
                for name in sensitivity
                if name.startswith("champion_vs_linear_")
                and not name.endswith(("_low", "_high"))
            ),
            None,
        )
        if key is not None:
            edge = _as_float(sensitivity.get(key))
            edge_low = _as_float(sensitivity.get(f"{key}_low"))
            edge_high = _as_float(sensitivity.get(f"{key}_high"))
    if edge is not None and edge_low is not None and edge_high is not None:
        if edge_low < 0 < edge_high:
            verdict = (
                "its edge over linear has an interval that crosses zero, so the data "
                "cannot tell them apart"
            )
        else:
            verdict = "its edge over linear clears zero, so it genuinely beat the reference"
        return (
            f"{winner} won the repeated-CV leaderboard at {value:.3f} {metric} "
            f"[{low:.3f}, {high:.3f}]; {verdict} "
            f"(Δ {edge:+.3f} [{edge_low:+.3f}, {edge_high:+.3f}])."
        )
    return (
        f"{winner} won the repeated-CV leaderboard at {value:.3f} {metric} "
        f"[{low:.3f}, {high:.3f}]."
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


def _balance_number(
    balance: list[tuple[str, int]], rows_in: int | None
) -> str:
    kept = sum(count for _, count in balance)
    if kept <= 0:
        return "class balance unavailable"
    counts = " / ".join(f"{label} {count:,}" for label, count in balance)
    text = f"{counts} (n={kept:,})"
    dropped = (rows_in - kept) if rows_in is not None and rows_in > kept else 0
    if dropped:
        text += f"; {dropped:,} unlabeled rows were dropped"
    return text


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
    tested: int | None,
    decided: int,
    counts: dict[str, int],
    significant: int | None,
) -> str:
    """``21 tested, top 12 by effect decided (…); 19 significant`` — never the row count."""
    parts = ", ".join(
        f"{counts[key]} {label}" for key, label in _VERDICT_LABELS if counts.get(key)
    )
    if tested is None:
        prefix = f"{decided} decided"
    elif tested == decided:
        prefix = f"{tested} tested, {decided} decided"
    else:
        prefix = f"{tested} tested, top {decided} by effect decided"
    suffix = f" ({parts})" if parts else ""
    tail = f"; {significant} significant" if significant is not None else ""
    return f"{prefix}{suffix}{tail}"


def _significant_count(frame: pd.DataFrame | None, tested: int | None) -> int | None:
    """Significant rows in the record's own effects table, only for its tested family."""
    if frame is None or "significant" not in frame.columns:
        return None
    if tested is not None and len(frame) != tested:
        return None
    return sum(1 for value in frame["significant"] if _is_true(value))


def _is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y", "t"}


def _feature_key(metric: str) -> str:
    return metric.removeprefix("association_")


def _effect_stats(
    frame: pd.DataFrame | None,
) -> dict[str, tuple[str | None, float | None, bool]]:
    """Per-feature statistic, shuffle null and bootstrap-bias flag, when present."""
    if frame is None or "feature" not in frame.columns:
        return {}
    stats: dict[str, tuple[str | None, float | None, bool]] = {}
    for _, row in frame.iterrows():
        statistic = str(row["effect_name"]) if "effect_name" in frame.columns else None
        null = _as_float(row["null_threshold"]) if "null_threshold" in frame.columns else None
        biased = (
            _is_true(row["bootstrap_biased"])
            if "bootstrap_biased" in frame.columns
            else False
        )
        stats[str(row["feature"])] = (statistic, null, biased)
    return stats


def _effect_label(effect: Effect) -> str:
    """The effect label with its own statistic name, so one axis never implies one unit."""
    return f"{effect.label} · {effect.statistic}" if effect.statistic else effect.label


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


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(str(value))
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
