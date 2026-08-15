"""``ds-pipeline audit`` — render persisted typed evidence into human-readable Markdown.

Pure rendering: reads JSON artifacts, deserializes them via typed Pydantic
models, renders Markdown, and writes files under ``reports/audit/``. It never
re-runs ingest/etl/stats/profile and never reads parquet.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from broadway.cleaning.models import StructuralCleanResult
from broadway.config.loader import load_config
from broadway.config.viz import load_viz_config
from broadway.data.join_audit import JoinAuditReport
from broadway.data.lookup_value_audit import LookupValueAuditReport
from broadway.discover.profile import DatasetProfile
from broadway.discover.qq import QqOverview
from broadway.reports.paths import AUDIT_DIR

PASS = "PASS"
WARNING = "WARNING"
INCOMPLETE = "INCOMPLETE"


def _identifier_threshold() -> float:
    return float(os.getenv("BROADWAY_IDENTIFIER_THRESHOLD", "0.95"))


def _fmt(value: str | None) -> str:
    return "-" if value is None else str(value)


def _sig3(x: float | None) -> str:
    return "-" if x is None else f"{x:.3g}"


def _incomplete_answer(source: str) -> str:
    return f"{INCOMPLETE} — artifact not found: {source or 'unknown'}"


def _render_page(
    title: str,
    answer: str,
    evidence_sections: list[tuple[str, str]],
    why: str,
    not_decided: str,
    source: str = "",
    extra_sections: list[tuple[str, str]] | None = None,
) -> str:
    lines = [f"# {title}", "", "## Answer", "", answer, "", "## Key evidence", ""]
    for subheading, body in evidence_sections:
        lines.append(f"### {subheading}")
        lines.append("")
        lines.append(body)
        lines.append("")
    for heading, body in extra_sections or []:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(body)
        lines.append("")
    lines.append("## Why this may matter")
    lines.append("")
    lines.append(why or "Nothing material to flag.")
    lines.append("")
    lines.append("## What Broadway did not decide")
    lines.append("")
    lines.append(not_decided)
    lines.append("")
    lines.append("## Technical details")
    lines.append("")
    lines.append(source or "-")
    lines.append("")
    return "\n".join(lines)


def _render_figure_block(
    caption: str,
    figures: list[str],
    by_figure: dict[str, list[str]],
) -> list[str]:
    lines: list[str] = []
    n = len(figures)
    for i, figure in enumerate(figures, start=1):
        names = ", ".join(by_figure.get(figure, []))
        lines.append(f"![{caption} — figure {i} of {n}](../{figure})")
        lines.append("")
        lines.append(
            f"In this figure: {names}. Chunk {i} of {n}; the trailing `_{i}` in the "
            "filename is the chunk number."
        )
        lines.append("")
    return lines


def _render_profile_evidence(qq: QqOverview | None) -> list[tuple[str, str]]:
    if qq is None:
        return []
    qq_by_figure: dict[str, list[str]] = {}
    dist_by_figure: dict[str, list[str]] = {}
    for feature in qq.features:
        if feature.figure:
            qq_by_figure.setdefault(feature.figure, []).append(feature.feature)
        if feature.dist_figure:
            dist_by_figure.setdefault(feature.dist_figure, []).append(feature.feature)
    diag_names = [
        f.feature
        for f in qq.features
        if f.status in ("plotted", "discrete")
        and f.skew is not None
        and f.kurtosis is not None
        and f.zero_rate is not None
    ]
    diag_by_figure = {figure: diag_names for figure in qq.diagnostics_figures}

    lines: list[str] = []
    if qq.figures:
        lines.append(f"Traces are {qq.standardization}.")
        if qq.sample_size is not None:
            lines.append(f"Sample size: n = {qq.sample_size:,}")
        lines.append("")
        lines.extend(_render_figure_block("Per-feature Q-Q plots", qq.figures, qq_by_figure))
        zones = load_viz_config().qq_zones
        mid = int((zones.central_quantiles[1] - zones.central_quantiles[0]) * 100)
        lines.append(
            "How to read (Q-Q): points should follow the fitted reference line; "
            "S-curves indicate tail behavior; curvature indicates skew. "
            f"Shaded bands mark the middle {mid}% (centre) and the ±{zones.tail_threshold}σ tails; "
            "the red dashed horizontal line is the zero-mass shelf (a flat clump of dots = a spike of exact zeros)."
        )
        lines.append("")
    if qq.dist_figures:
        lines.append("Histograms are in raw units.")
        lines.append("")
        lines.extend(_render_figure_block("Per-feature distributions", qq.dist_figures, dist_by_figure))
        lines.append(
            "How to read (distribution): actual spread and skew in original units; "
            "look for heavy tails, multimodality, and gaps."
        )
        lines.append("")

    diag_features = [f for f in qq.features if f.status in ("plotted", "discrete")]
    if diag_features:
        lines.append("### Distribution diagnostics")
        lines.append("")
        lines.append("| Variable | n | mean | std | skew | excess_kurtosis | zero_rate | p99/median | max/median | log_skew |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for f in diag_features:
            zero_rate = "-" if f.zero_rate is None else f"{f.zero_rate:.3f}"
            p99_ratio = (
                _sig3(f.p99 / f.median)
                if f.median is not None and f.p99 is not None and f.median > 0
                else "-"
            )
            max_ratio = (
                _sig3(f.max / f.median)
                if f.median is not None and f.max is not None and f.median > 0
                else "-"
            )
            lines.append(
                f"| {f.feature} | {f.n_valid} | {_sig3(f.mean)} | {_sig3(f.std)} | "
                f"{_sig3(f.skew)} | {_sig3(f.kurtosis)} | {zero_rate} | "
                f"{p99_ratio} | {max_ratio} | {_sig3(f.log_skew)} |"
            )
        lines.append("")

    if qq.diagnostics_figures:
        lines.extend(_render_figure_block(
            "Per-feature distribution diagnostics", qq.diagnostics_figures, diag_by_figure,
        ))
        lines.append(
            "How to read (diagnostics): colors are per-column z-scores; "
            "cell text is the raw value."
        )
        lines.append("")

    lines.append("### Decision flags")
    lines.append("")
    flagged = [(f.feature, f.flags) for f in qq.features if f.flags]
    if flagged:
        for name, fls in flagged:
            for fl in fls:
                lines.append(f"- {name}: {fl}")
    else:
        lines.append("none")
    lines.append("")

    notes: list[str] = []
    for feature in qq.features:
        if feature.status == "excluded":
            notes.append(f"{feature.feature}: {feature.reason}")
    for feature in qq.features:
        if feature.status == "discrete":
            notes.append(
                f"{feature.feature}: {feature.reason} (excluded from Q-Q, kept as a "
                "bar chart in the distribution grid)"
            )
    for name in qq.flagged_id_columns:
        notes.append(
            f"{name}: name suggests an identifier; declare in exclude_from_profiling to exclude"
        )
    for name in qq.non_numeric_columns:
        notes.append(f"not profiled (non-numeric): {name}")

    if notes:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in notes)
    return [("Profile evidence", "\n".join(lines))]


def render_profile(profile: DatasetProfile | None, source: str = "", qq: QqOverview | None = None) -> str:
    if profile is None:
        return _render_page(
            "Dataset Profile",
            _incomplete_answer(source),
            [],
            "",
            "Broadway did not exclude or transform any column based on this profile.",
            source,
        )

    answer = f"The dataset has {profile.row_count} rows and {len(profile.columns)} columns."

    header = "| Variable | Type | Missing | Unique | Min | Max |"
    sep = "| --- | --- | --- | --- | --- | --- |"
    rows = [header, sep]
    for name, col in profile.columns.items():
        is_dt = col.dtype.startswith("datetime")
        mn = _fmt(col.datetime_min if is_dt else col.min)
        mx = _fmt(col.datetime_max if is_dt else col.max)
        rows.append(f"| {name} | {col.dtype} | {col.null_count} | {col.cardinality} | {mn} | {mx} |")
    variables = "\n".join(rows)

    threshold = _identifier_threshold()
    observations: list[str] = []
    for name, col in profile.columns.items():
        if col.identifier_score >= threshold:
            observations.append(
                f"{name} has high cardinality (identifier_score={col.identifier_score}) and behaves "
                "like an identifier; it may not be a meaningful grouping variable."
            )
        if col.null_count > 0:
            observations.append(f"{name} has {col.null_count} missing values.")
    obs_body = "\n".join(f"- {o}" for o in observations) if observations else "none"

    evidence = [("Variables", variables), ("Potentially important observations", obs_body)]
    why = (
        "High-cardinality/identifier-like columns can inflate group counts or leak "
        "identity if used as grouping features."
    )
    not_decided = "Broadway did not exclude or transform any column based on this profile."
    return _render_page(
        "Dataset Profile",
        answer,
        evidence,
        why,
        not_decided,
        source,
        extra_sections=_render_profile_evidence(qq),
    )


def render_transform(result: StructuralCleanResult | None, source: str = "") -> str:
    not_decided = "Broadway recorded these changes but did not make any analytical judgment about them."
    if result is None:
        return _render_page(
            "Structural Transform", _incomplete_answer(source), [], "", not_decided, source
        )

    audit = result.audit
    answer = (
        f"Structural canonicalization removed {audit.rows_dropped_total} of {audit.rows_in} rows; "
        f"{audit.rows_out} rows remain. Rows were removed only for deterministic structural reasons. "
        "No domain or outlier cleaning was performed here."
    )

    row_transitions = (
        f"- rows_in: {audit.rows_in}\n"
        f"- rows_out: {audit.rows_out}\n"
        f"- rows_dropped_total: {audit.rows_dropped_total}\n"
        f"- rows_dropped_unexplained: {audit.rows_dropped_unexplained}"
    )
    reasons = "\n".join(f"- {r}" for r in audit.reasons) if audit.reasons else "none"
    added = ", ".join(audit.columns_added) if audit.columns_added else "none"
    removed = ", ".join(audit.columns_removed) if audit.columns_removed else "none"

    if result.parse_failures:
        pf_lines = []
        for pf in result.parse_failures:
            examples = ", ".join(pf.examples) if pf.examples else "-"
            pf_lines.append(
                f"- {pf.column}: {pf.count} value(s) failed to parse as {pf.target_dtype} "
                f"(examples: {examples})"
            )
        parse_failures = "\n".join(pf_lines)
    else:
        parse_failures = "none"

    evidence = [
        ("Row transitions", row_transitions),
        ("Reasons", reasons),
        ("Columns added", added),
        ("Columns removed", removed),
        ("Parse failures", parse_failures),
    ]

    why_parts = []
    if result.parse_failures:
        why_parts.append(
            "Some values could not be parsed to their target dtype and became null; "
            "analyses on those columns may reflect fewer valid observations."
        )
    if audit.rows_dropped_unexplained > 0:
        why_parts.append("Some row loss is not accounted for by recorded reasons; review the transform.")
    why = " ".join(why_parts)

    return _render_page("Structural Transform", answer, evidence, why, not_decided, source)


def _join_counts(report: JoinAuditReport) -> tuple[int, int, int, int]:
    if not report.joins:
        return (0, 0, 0, 0)
    row_counts = {join.rows_attempted for join in report.joins}
    if len(row_counts) != 1:
        raise ValueError(
            "JoinAuditReport contains joins evaluated over different row counts; "
            "cannot produce a single rows_evaluated summary."
        )
    rows_evaluated = next(iter(row_counts))
    n_joins = len(report.joins)
    matched_events = sum(join.matched for join in report.joins)
    unmatched_events = sum(join.unmatched for join in report.joins)
    return rows_evaluated, n_joins, matched_events, unmatched_events


def render_join(report: JoinAuditReport | None, source: str = "") -> str:
    not_decided = "Broadway did not drop or impute unmatched rows."
    if report is None:
        return _render_page("Join Audit", _incomplete_answer(source), [], "", not_decided, source)

    rows_evaluated, n_joins, matched_events, unmatched_events = _join_counts(report)
    answer = "\n".join(
        [
            f"Rows evaluated: {rows_evaluated}",
            f"Lookup joins checked: {n_joins}",
            f"Matched join-key events: {matched_events}",
            f"Unmatched join-key events: {unmatched_events}",
            f"Join completeness: {_join_state(report)}",
        ]
    )
    answer += (
        "\n\nThis describes key matching only. It does NOT mean the matched values were usable — "
        "see the Lookup Value Audit."
    )

    rows = [
        "| lookup | rows_evaluated | matched | unmatched | null_keys | unmatched_rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for j in report.joins:
        rows.append(
            f"| {j.lookup} | {j.rows_attempted} | {j.matched} | {j.unmatched} | "
            f"{j.null_keys} | {j.unmatched_rate} |"
        )
    table = "\n".join(rows)

    why = (
        "Unmatched keys mean some observations did not receive enrichment values."
        if any(j.unmatched > 0 for j in report.joins)
        else ""
    )
    return _render_page("Join Audit", answer, [("Joins", table)], why, not_decided, source)


def render_lookup_values(report: LookupValueAuditReport | None, source: str = "") -> str:
    not_decided = "Broadway did not exclude these rows or redefine the population."
    if report is None:
        return _render_page(
            "Lookup Value Audit", _incomplete_answer(source), [], "", not_decided, source
        )

    affected = [
        (lookup, col)
        for lookup in report.lookups
        for col in lookup.columns
        if col.affected_rows > 0
    ]
    answer = (
        "Some matched enrichment values were missing or sentinel."
        if affected
        else "Matched enrichment values were all usable."
    )

    sections: list[tuple[str, str]] = []
    for lookup in report.lookups:
        rows = [
            "| column | nulls | sentinels | affected_rows | affected_rate | affected_keys |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for col in lookup.columns:
            sentinels = (
                ", ".join(f"{k}: {v}" for k, v in col.sentinel_counts.items())
                if col.sentinel_counts
                else "-"
            )
            keys = ", ".join(col.affected_lookup_keys) if col.affected_lookup_keys else "-"
            rows.append(
                f"| {col.column} | {col.null_count} | {sentinels} | {col.affected_rows} | "
                f"{col.affected_rate} | {keys} |"
            )
        sections.append((lookup.lookup, "\n".join(rows)))

    why_parts = []
    for _, col in affected:
        why_parts.append(
            f"{col.column} has deficient values; analyses that group or filter by this column may "
            "use fewer or differently classified observations depending on the analytical "
            "population definition."
        )
    why = "\n".join(f"- {w}" for w in why_parts)

    return _render_page("Lookup Value Audit", answer, sections, why, not_decided, source)


def _transform_state(result: StructuralCleanResult | None) -> str:
    if result is None:
        return INCOMPLETE
    if result.parse_failures or result.audit.rows_dropped_unexplained > 0:
        return WARNING
    return PASS


def _join_state(report: JoinAuditReport | None) -> str:
    if report is None:
        return INCOMPLETE
    if any(j.unmatched > 0 for j in report.joins):
        return WARNING
    return PASS


def _lookup_state(report: LookupValueAuditReport | None) -> str:
    if report is None:
        return INCOMPLETE
    if any(col.affected_rows > 0 for l in report.lookups for col in l.columns):
        return WARNING
    return PASS


def _profile_state(profile: DatasetProfile | None) -> str:
    if profile is None:
        return INCOMPLETE
    if any(col.null_count > 0 for col in profile.columns.values()):
        return WARNING
    return PASS


def _dataset_status(
    result: StructuralCleanResult | None,
    join_report: JoinAuditReport | None,
    lookup_report: LookupValueAuditReport | None,
    profile: DatasetProfile | None,
) -> str:
    states = [
        _transform_state(result),
        _join_state(join_report),
        _lookup_state(lookup_report),
        _profile_state(profile),
    ]
    if any(s == INCOMPLETE for s in states):
        return INCOMPLETE
    if any(s == WARNING for s in states):
        return "READY WITH WARNINGS"
    return "READY"


def _lookup_affected_total(report: LookupValueAuditReport | None) -> int:
    if report is None:
        return 0
    return sum(col.affected_rows for l in report.lookups for col in l.columns)


def _borough_deficient(report: LookupValueAuditReport | None) -> bool:
    if report is None:
        return False
    return any(
        "borough" in col.column.lower() and col.affected_rows > 0
        for lookup in report.lookups
        for col in lookup.columns
    )


def _join_summary(report: JoinAuditReport | None) -> str:
    if report is None:
        return "no join audit available"
    rows_evaluated, n_joins, matched_events, unmatched_events = _join_counts(report)
    if unmatched_events == 0:
        return (
            f"{rows_evaluated} rows evaluated across {n_joins} join(s); "
            f"{matched_events} matched key events, 0 unmatched"
        )
    return (
        f"{rows_evaluated} rows evaluated across {n_joins} join(s); "
        f"{unmatched_events} unmatched key event(s)"
    )


def _lookup_summary(report: LookupValueAuditReport | None) -> str:
    if report is None:
        return "no lookup value audit available"
    total = _lookup_affected_total(report)
    if total == 0:
        return "all matched enrichment values usable"
    return f"{total} matched rows had missing or sentinel enrichment values"


def _changed_items(result: StructuralCleanResult | None) -> list[str]:
    if result is None:
        return []
    items: list[str] = []
    for reason in result.audit.reasons:
        match = re.search(r"^(.*?):\s*-(\d+) rows$", reason)
        if not match:
            continue
        kind, count = match.group(1).strip(), int(match.group(2))
        if count == 0:
            continue
        label = {
            "duplicates": "exact-duplicate rows dropped",
            "null target": "target-missing rows dropped",
            "CI sampling": "rows removed by CI sampling",
        }.get(kind, kind)
        items.append(f"{label}: {count}")
    for pf in result.parse_failures:
        if pf.count > 0:
            kind = "datetime" if "datetime" in pf.target_dtype else "numeric"
            items.append(f"{kind} parse failures in {pf.column}: {pf.count}")
    return items


def _considerations(
    result: StructuralCleanResult | None,
    join_report: JoinAuditReport | None,
    lookup_report: LookupValueAuditReport | None,
    profile: DatasetProfile | None,
) -> list[str]:
    consider: list[str] = []
    if profile is not None:
        threshold = _identifier_threshold()
        if any(c.identifier_score >= threshold for c in profile.columns.values()):
            consider.append(
                "High-cardinality/identifier-like columns can inflate group counts or leak "
                "identity if used as grouping features."
            )
        if any(c.null_count > 0 for c in profile.columns.values()):
            consider.append("Columns with missing values may reflect fewer valid observations in analyses.")
    if result is not None:
        if result.parse_failures:
            consider.append(
                "Some values could not be parsed to their target dtype and became null; "
                "analyses on those columns may reflect fewer valid observations."
            )
        if result.audit.rows_dropped_unexplained > 0:
            consider.append("Some row loss is not accounted for by recorded reasons; review the transform.")
    if join_report is not None and any(j.unmatched > 0 for j in join_report.joins):
        consider.append("Unmatched keys mean some observations did not receive enrichment values.")
    if lookup_report is not None:
        for lookup in lookup_report.lookups:
            for col in lookup.columns:
                if col.affected_rows > 0:
                    consider.append(
                        f"{col.column} has deficient values; analyses that group or filter by this "
                        "column may use fewer or differently classified observations depending on "
                        "the analytical population definition."
                    )
    consider.append("No decision has been made automatically.")
    if _borough_deficient(lookup_report):
        consider.append("If your analysis compares NYC boroughs, define the analytical population explicitly.")
    return consider


def render_index(
    dataset: str,
    analysis: str | None,
    result: StructuralCleanResult | None,
    join_report: JoinAuditReport | None,
    lookup_report: LookupValueAuditReport | None,
    profile: DatasetProfile | None,
) -> str:
    title = f"Data Audit — {dataset}"
    if analysis:
        title += f" — supporting {analysis}"

    if result is None:
        data_used = f"{INCOMPLETE} — clean artifact not found"
    else:
        audit = result.audit
        data_used = (
            f"- rows_in: {audit.rows_in}\n"
            f"- rows_out (canonical): {audit.rows_out}\n"
            f"- rows_dropped_total: {audit.rows_dropped_total}"
        )

    status = _dataset_status(result, join_report, lookup_report, profile)
    status_line = f"Dataset status: {status}"
    if status == "READY WITH WARNINGS":
        status_line += (
            "\nAll required audit artifacts are present, but one or more deterministic "
            "evidence checks reported non-zero deficiencies."
        )

    changed = _changed_items(result)
    if changed:
        changed_body = "\n".join(f"- {c}" for c in changed)
    elif result is None:
        changed_body = f"{INCOMPLETE} — clean artifact not found"
    else:
        changed_body = "none"

    enrichment = (
        f"- Join completeness: {_join_state(join_report)} — {_join_summary(join_report)}\n"
        f"- Lookup value quality: {_lookup_state(lookup_report)} — {_lookup_summary(lookup_report)}"
    )

    consider_body = "\n".join(f"- {c}" for c in _considerations(result, join_report, lookup_report, profile))

    details = "\n".join(
        [
            "- [profile](profile.md)",
            "- [transform](transform.md)",
            "- [join](join.md)",
            "- [lookup_values](lookup_values.md)",
            "- [lineage graph](../lineage/graph.md)",
        ]
    )

    lines = [
        f"# {title}",
        "",
        "## Data used",
        "",
        data_used,
        "",
        status_line,
        "",
        "## What changed",
        "",
        changed_body,
        "",
        "## Enrichment quality",
        "",
        enrichment,
        "",
        "## Things to consider before inference",
        "",
        consider_body,
        "",
        "## Details",
        "",
        details,
        "",
    ]
    return "\n".join(lines)


def _evidence_paths(cfg: Any) -> dict[str, Path]:
    processed = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    return {
        "clean": processed / f"{cfg.dataset.name}_clean.json",
        "join": processed / f"{cfg.dataset.name}_join_audit.json",
        "lookup": processed / f"{cfg.dataset.name}_lookup_value_audit.json",
        "profile": Path(os.getenv("BROADWAY_ARTIFACTS_DIR", "artifacts")) / "discover" / "profile.json",
        "qq": Path(os.getenv("BROADWAY_ARTIFACTS_DIR", "artifacts")) / "discover" / "qq_overview.json",
    }


def _load(path: Path, model: type) -> Any:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def run(dataset: str, analysis: str | None, environment: str) -> None:
    cfg = load_config("etl", dataset=dataset, environment=environment)
    paths = _evidence_paths(cfg)

    result = _load(paths["clean"], StructuralCleanResult)
    join_report = _load(paths["join"], JoinAuditReport)
    lookup_report = _load(paths["lookup"], LookupValueAuditReport)
    profile = _load(paths["profile"], DatasetProfile)
    qq = _load(paths.get("qq", Path("artifacts/discover/qq_overview.json")), QqOverview)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "profile.md").write_text(
        render_profile(profile, str(paths["profile"]), qq), encoding="utf-8"
    )
    (AUDIT_DIR / "transform.md").write_text(render_transform(result, str(paths["clean"])), encoding="utf-8")
    (AUDIT_DIR / "join.md").write_text(render_join(join_report, str(paths["join"])), encoding="utf-8")
    (AUDIT_DIR / "lookup_values.md").write_text(
        render_lookup_values(lookup_report, str(paths["lookup"])), encoding="utf-8"
    )
    (AUDIT_DIR / "index.md").write_text(
        render_index(dataset, analysis, result, join_report, lookup_report, profile),
        encoding="utf-8",
    )

    status = _dataset_status(result, join_report, lookup_report, profile)
    print(f"dataset status: {status}")

    if join_report is None:
        print("join unmatched rate: no join audit")
    else:
        rows_evaluated, n_joins, matched_events, unmatched_events = _join_counts(join_report)
        total_events = matched_events + unmatched_events
        rate = round(unmatched_events / total_events * 100, 4) if total_events else 0.0
        print(f"join unmatched rate: {rate}% across {total_events} key events")

    if lookup_report is None:
        print("lookup affected rows total: no lookup audit")
    else:
        print(f"lookup affected rows total: {_lookup_affected_total(lookup_report)}")

    if result is None:
        print("canonical rows: no clean audit")
    else:
        print(f"canonical rows: {result.audit.rows_out}")
