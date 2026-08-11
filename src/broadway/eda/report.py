"""Compose self-contained HTML report from all submodule outputs."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go


def build_report(summary_data: dict, quality_data: dict, missing_data: dict, figures: list[go.Figure], output_path: Path) -> None:
    plotly_divs = "\n".join(fig.to_html(full_html=False) for fig in figures)
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>EDA Report</title></head>
<body>
<h1>EDA Report</h1>
<h2>Summary</h2>
<pre>{json.dumps(summary_data, indent=2)}</pre>
<h2>Quality</h2>
<pre>{json.dumps(quality_data, indent=2)}</pre>
<h2>Missingness</h2>
<pre>{json.dumps(missing_data, indent=2)}</pre>
<h2>Visualizations</h2>
{plotly_divs}
</body>
</html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
