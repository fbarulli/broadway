"""Compose one self-contained HTML report from the collected EDA results."""

from __future__ import annotations

from html import escape

from broadway.eda.models import (
    ColumnSummary,
    CompareResult,
    EdaResult,
    MissingResult,
    QualityResult,
    SummaryResult,
)

_DECIMALS = 4
_PERCENT = 100.0
_KIB = 1024.0
_MIB = _KIB * 1024.0

_STYLE = """
body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 2rem auto;
       max-width: 1180px; color: #1f2933; background: #ffffff; }
h1 { border-bottom: 3px solid #334e68; padding-bottom: .4rem; }
h2 { margin-top: 2rem; color: #243b53; border-bottom: 1px solid #d9e2ec; }
table { border-collapse: collapse; width: 100%; margin: .8rem 0 1.2rem; font-size: .86rem; }
th, td { border: 1px solid #d9e2ec; padding: .35rem .5rem; text-align: left; }
th { background: #f0f4f8; }
tr:nth-child(even) td { background: #fbfcfe; }
.problem { color: #9b1c1c; font-weight: 600; }
.ok { color: #17643a; font-weight: 600; }
.note { color: #52606d; font-size: .9rem; }
figure { margin: 1rem 0 1.6rem; }
img { max-width: 100%; height: auto; border: 1px solid #d9e2ec; }
code { background: #f0f4f8; padding: .1rem .25rem; }
"""


class EdaReportBuilder:
    """Renders the typed results into a portable, dependency-free HTML document."""

    def build(self, result: EdaResult) -> str:
        body = [
            self._header(result),
            self._findings(result),
            self._summary_section(result.summary, result.missing),
            self._missing_section(result.missing),
            self._quality_section(result.quality),
            self._compare_section(result.compare),
            self._figures_section(result),
        ]
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en"><head><meta charset="utf-8">'
            f"<title>EDA — {escape(result.dataset)}</title>"
            f"<style>{_STYLE}</style></head>\n"
            f"<body>{''.join(body)}</body></html>\n"
        )

    def _header(self, result: EdaResult) -> str:
        experiment = result.experiment or "—"
        sampling = (
            f"{result.sampled_rows} of {result.source_rows} rows (sampled)"
            if result.sampled
            else f"{result.source_rows} rows (all rows)"
        )
        return (
            "<h1>Exploratory Data Analysis</h1>"
            f"<p><strong>Dataset:</strong> {escape(result.dataset)} &nbsp; "
            f"<strong>Experiment:</strong> {escape(experiment)} &nbsp; "
            f"<strong>Task:</strong> {escape(result.task)}</p>"
            f"<p class='note'>Profiled frame: {escape(sampling)}.</p>"
        )

    def _findings(self, result: EdaResult) -> str:
        summary = result.summary
        missing = result.missing
        quality = result.quality
        items = [
            (
                f"{summary.rows} profiled rows × {summary.columns} columns "
                f"({summary.numeric_columns} numeric, {summary.categorical_columns} "
                f"categorical, {summary.temporal_columns} temporal)."
            ),
            (
                f"Missing cells: {missing.missing_cells} of {missing.total_cells} "
                f"({self._fmt_pct(missing.missing_rate)}); "
                f"{len(missing.columns)} column(s) carry missing values."
            ),
            (
                f"Duplicate rows: {quality.duplicate_rows} "
                f"({self._fmt_pct(quality.duplicate_rate)})."
            ),
            (
                f"Constant columns: {len(quality.constant_columns)}"
                + (
                    f" — {escape(', '.join(quality.constant_columns))}."
                    if quality.constant_columns
                    else "."
                )
            ),
            (
                f"Outlier screening covered {len(quality.outliers)} numeric column(s) "
                f"(IQR multiplier / z threshold applied)."
            ),
        ]
        imbalance = quality.imbalance
        if imbalance.applicable:
            items.append(
                f"Class imbalance: minority share {self._fmt_pct(imbalance.minority_share)}"
                + (
                    f", ratio {self._fmt(imbalance.imbalance_ratio)}"
                    if imbalance.imbalance_ratio
                    else ""
                )
                + (" — flagged." if imbalance.flagged else " — within threshold.")
            )
        else:
            items.append(f"Class imbalance: not applicable — {escape(imbalance.reason or '')}.")
        compare = result.compare
        if compare.available:
            items.append(
                f"Before/after ETL comparison available: {compare.row_delta:+d} rows, "
                f"{len(compare.added_columns)} added / {len(compare.removed_columns)} "
                f"removed column(s)."
            )
        else:
            items.append(f"Before/after ETL comparison skipped: {escape(compare.skip_reason or '')}.")
        return "<h2>Findings</h2><ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"

    def _summary_section(self, summary: SummaryResult, missing: MissingResult) -> str:
        missing_rate = {item.column: item.missing_rate for item in missing.columns}
        rows = []
        for item in summary.detail:
            rows.append(
                [
                    escape(item.name),
                    escape(item.dtype),
                    escape(item.role),
                    escape(item.kind),
                    self._fmt_pct(missing_rate.get(item.name, 0.0)),
                    self._stats(item),
                    self._memory(item.memory_bytes),
                ]
            )
        return (
            "<h2>Summary</h2>"
            f"<p class='note'>Total memory: {self._memory(summary.total_memory_bytes)}.</p>"
            + self._table(
                ["column", "dtype", "role", "kind", "missing", "distribution", "memory"],
                rows,
            )
        )

    def _stats(self, item: ColumnSummary) -> str:
        if item.numeric is not None:
            numeric = item.numeric
            return (
                f"n={numeric.count}, mean={self._fmt(numeric.mean)}, "
                f"median={self._fmt(numeric.median)}, std={self._fmt(numeric.std)}, "
                f"skew={self._fmt(numeric.skew)}, kurt={self._fmt(numeric.kurtosis)}"
            )
        if item.categorical is not None:
            categorical = item.categorical
            top = ", ".join(
                f"{escape(value.value)} ({value.count})"
                for value in categorical.top_values[:5]
            )
            return f"unique={categorical.unique}; top: {top or '—'}"
        if item.temporal is not None:
            temporal = item.temporal
            return (
                f"unique={temporal.unique}; "
                f"{escape(str(temporal.minimum))} → {escape(str(temporal.maximum))}"
            )
        return "—"

    def _missing_section(self, missing: MissingResult) -> str:
        rows = [
            [escape(item.column), str(item.missing_count), self._fmt_pct(item.missing_rate)]
            for item in missing.columns
        ]
        patterns = [
            [
                escape(", ".join(pattern.columns)) or "(none)",
                str(pattern.row_count),
                self._fmt_pct(pattern.row_rate),
            ]
            for pattern in missing.patterns
        ]
        complete = (
            f"<p class='note'>Complete columns: {escape(', '.join(missing.complete_columns))}.</p>"
            if missing.complete_columns
            else ""
        )
        observed = (
            "<p class='note'>Observation only — this report does not impute or drop rows.</p>"
        )
        return (
            "<h2>Missingness</h2>"
            f"<p>{missing.missing_cells} missing cells of {missing.total_cells} "
            f"({self._fmt_pct(missing.missing_rate)}); "
            f"{missing.pattern_count} distinct missingness pattern(s).</p>"
            + observed
            + complete
            + self._table(["column", "missing", "rate"], rows)
            + ("<h3>Top missingness patterns</h3>" + self._table(["columns", "rows", "rate"], patterns)
               if patterns else "")
        )

    def _quality_section(self, quality: QualityResult) -> str:
        constants = (
            f"<p class='problem'>Constant columns: {escape(', '.join(quality.constant_columns))}</p>"
            if quality.constant_columns
            else "<p class='ok'>No constant columns.</p>"
        )
        imbalance = quality.imbalance
        if imbalance.applicable:
            counts = ", ".join(
                f"{escape(item.value)} ({item.count}, {self._fmt_pct(item.share)})"
                for item in imbalance.value_counts
            )
            verdict = "problem" if imbalance.flagged else "ok"
            imbalance_html = (
                f"<p class='{verdict}'>{escape(imbalance.target)}: {counts}; "
                f"minority share {self._fmt_pct(imbalance.minority_share)}, "
                f"ratio {self._fmt(imbalance.imbalance_ratio)}"
                f"{' — flagged' if imbalance.flagged else ' — within threshold'}.</p>"
            )
        else:
            imbalance_html = (
                f"<p class='note'>Class imbalance not applicable: "
                f"{escape(imbalance.reason or '')}.</p>"
            )
        outlier_rows = [
            [
                escape(item.column),
                f"{item.iqr_count} ({self._fmt_pct(item.iqr_rate)})",
                f"{self._fmt(item.iqr_lower)} … {self._fmt(item.iqr_upper)}",
                f"{item.z_count} ({self._fmt_pct(item.z_rate)})",
            ]
            for item in quality.outliers
        ]
        return (
            "<h2>Quality</h2>"
            f"<p>Duplicate rows: {quality.duplicate_rows} "
            f"({self._fmt_pct(quality.duplicate_rate)}).</p>"
            + constants
            + imbalance_html
            + self._table(
                ["column", "IQR outliers", "IQR fence", "z-score outliers"],
                outlier_rows,
            )
        )

    def _compare_section(self, compare: CompareResult) -> str:
        if not compare.available:
            return (
                "<h2>Before / after ETL</h2>"
                f"<p class='note'>Skipped: {escape(compare.skip_reason or '')}</p>"
            )
        rows = [
            [
                escape(item.column),
                self._fmt(item.before_mean),
                self._fmt(item.after_mean),
                self._fmt(item.mean_delta),
                self._fmt(item.before_median),
                self._fmt(item.after_median),
                self._fmt(item.median_delta),
            ]
            for item in compare.numeric_shifts
        ]
        dtype_rows = [
            [escape(name), escape(change)]
            for name, change in sorted(compare.dtype_changes.items())
        ]
        notes = "".join(f"<li>{escape(note)}</li>" for note in compare.notes)
        return (
            "<h2>Before / after ETL</h2>"
            f"<p>{escape(compare.before_label)}: {compare.before_rows} rows → "
            f"{escape(compare.after_label)}: {compare.after_rows} rows "
            f"(delta {compare.row_delta:+d}).</p>"
            f"<p>Added columns: {escape(', '.join(compare.added_columns)) or '—'}; "
            f"removed columns: {escape(', '.join(compare.removed_columns)) or '—'}.</p>"
            + (f"<ul>{notes}</ul>" if notes else "")
            + ("<h3>Dtype changes</h3>" + self._table(["column", "change"], dtype_rows)
               if dtype_rows else "")
            + "<h3>Numeric distribution shift</h3>"
            + self._table(
                ["column", "before mean", "after mean", "mean delta",
                 "before median", "after median", "median delta"],
                rows,
            )
        )

    def _figures_section(self, result: EdaResult) -> str:
        if not result.figures:
            return "<h2>Figures</h2><p class='note'>No figures were produced.</p>"
        blocks = [
            f"<figure><figcaption><strong>{escape(figure.title)}</strong></figcaption>"
            f'<img alt="{escape(figure.title)}" '
            f'src="data:{figure.mime_type};base64,{figure.data_base64}"></figure>'
            for figure in result.figures
        ]
        return "<h2>Figures</h2>" + "".join(blocks)

    def _table(self, headers: list[str], rows: list[list[str]]) -> str:
        head = "".join(f"<th>{escape(header)}</th>" for header in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
        )
        if not body:
            body = f"<tr><td colspan='{len(headers)}'>none</td></tr>"
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    @staticmethod
    def _fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.{_DECIMALS}f}"

    @staticmethod
    def _fmt_pct(value: float | None) -> str:
        return "—" if value is None else f"{value * _PERCENT:.2f}%"

    @staticmethod
    def _memory(value: int) -> str:
        if value >= _MIB:
            return f"{value / _MIB:.2f} MiB"
        return f"{value / _KIB:.2f} KiB"
