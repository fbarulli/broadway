"""Bare-bones FastAPI dashboard + per-experiment pages for the univariate
``fare_amount_trip_distance`` experiment series.

Scope: only the 30 step scripts under ``experiments/univariate/
fare_amount_trip_distance/`` (``*.py`` excluding ``_*.py``) are censused,
graphed, and observed. Other experiment series (``multivariate``,
``mlflow``) are intentionally out of scope for now — nothing walks them.

Each step page renders a bare-bones HTML pipeline graph (upstream -> this
step -> downstream consumers of the datasets it writes), the step's result
artifacts, and a form for a human observation plus a required verdict,
persisted to ``artifacts/experiments/observations/<stem>.json``. No
templates, no JS, no external assets.
"""

import ast
import html
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

EXPERIMENT_DIR = (
    Path(__file__).resolve().parent / "experiments" / "univariate" / "fare_amount_trip_distance"
)
RESULTS_DIR = (
    Path(__file__).resolve().parent / "experiments" / "results" / "univariate" / "fare_amount_trip_distance"
)
OBSERVATIONS_DIR = Path(__file__).resolve().parent / "artifacts" / "experiments" / "observations"

_PARQUET_CONSTANTS = ("CLEAN_PARQUET", "FULL_PARQUET", "RATECODE1_PARQUET")
_EXTERNAL_RE = re.compile(r"read_training_sample|load_metered|load_working")
_NUMBER_PREFIX = re.compile(r"^\d+:\s*")
_VERDICTS = ("supported", "refuted", "inconclusive", "partial")
_UPSTREAM_LABEL = "upstream (project data / working dataset)"

app = FastAPI()
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


@dataclass
class ScriptProfile:
    """How one step script relates to the series' shared parquet constants."""

    produced: set[str] = field(default_factory=set)
    consumed: set[str] = field(default_factory=set)
    external: bool = False


def series_scripts() -> list[Path]:
    """Return the series' step scripts (``*.py``, excluding ``_*.py``), sorted."""
    return sorted(p for p in EXPERIMENT_DIR.glob("*.py") if not p.name.startswith("_"))


def script_profile(script: Path) -> ScriptProfile:
    """Classify a script: parquet constants it produces/consumes + external source."""
    text = script.read_text(encoding="utf-8")
    lines = text.splitlines()
    produced = {
        const
        for const in _PARQUET_CONSTANTS
        if any(const in line and "to_parquet" in line for line in lines)
    }
    consumed = {const for const in _PARQUET_CONSTANTS if const in text} - produced
    return ScriptProfile(
        produced=produced, consumed=consumed, external=bool(_EXTERNAL_RE.search(text))
    )


def series_profiles() -> dict[str, ScriptProfile]:
    """Return ``{stem: profile}`` for every step script in the series."""
    return {script.stem: script_profile(script) for script in series_scripts()}


def census_rows() -> list[dict[str, str | int]]:
    """Build one dashboard row per series script (experiment | question | status | artifacts)."""
    rows: list[dict[str, str | int]] = []
    for script in series_scripts():
        stem = script.stem
        results = _result_files(stem)
        rows.append(
            {
                "experiment": stem,
                "question": _question_from_docstring(script),
                "status": "ran" if results else "no outputs",
                "artifacts": len(results),
            }
        )
    return rows


def load_observations(stem: str) -> dict[str, str]:
    """Load saved observations for ``stem``, or an empty dict."""
    path = OBSERVATIONS_DIR / f"{stem}.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _question_from_docstring(script: Path) -> str:
    """Return the docstring's first line with its ``NN:`` prefix stripped."""
    try:
        tree = ast.parse(script.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return ""
    doc = ast.get_docstring(tree, clean=False)
    if not doc:
        return ""
    first_line = doc.strip().splitlines()[0]
    return _NUMBER_PREFIX.sub("", first_line, count=1).strip()


def _result_files(stem: str) -> list[Path]:
    """Return result files for ``stem`` under RESULTS_DIR, sorted."""
    if not RESULTS_DIR.is_dir():
        return []
    return sorted(RESULTS_DIR.glob(f"{stem}.*"))


def _has_results(stem: str) -> bool:
    """Whether any result files exist for ``stem`` under RESULTS_DIR."""
    return any(RESULTS_DIR.glob(f"{stem}.*"))


def _box_class(stem: str) -> str:
    """CSS class for a node box: green when the step ran, gray otherwise."""
    return "box green" if _has_results(stem) else "box gray"


def _producers_of(constant: str, profiles: dict[str, ScriptProfile]) -> list[str]:
    """Return stems of scripts that produce ``constant``, sorted."""
    return sorted(stem for stem, profile in profiles.items() if constant in profile.produced)


def _consumers_of(constant: str, stem: str, profiles: dict[str, ScriptProfile]) -> list[str]:
    """Return stems of other scripts that consume ``constant``, sorted."""
    return sorted(
        other for other, profile in profiles.items() if other != stem and constant in profile.consumed
    )


def _graph_html(stem: str, profiles: dict[str, ScriptProfile]) -> str:
    """Render the bare-bones pipeline graph as escaped HTML boxes + arrows."""
    profile = profiles[stem]
    nodes: list[tuple[str, str]] = []
    if profile.external:
        nodes.append((_UPSTREAM_LABEL, "box gray"))
    else:
        for constant in sorted(profile.consumed):
            for producer in _producers_of(constant, profiles):
                nodes.append((producer, _box_class(producer)))
    nodes.append((stem, "box strong " + _box_class(stem)))
    for constant in sorted(profile.produced):
        for consumer in _consumers_of(constant, stem, profiles):
            nodes.append((consumer, _box_class(consumer)))
    boxes = " <span class=\"arrow\">→</span> ".join(
        f'<span class="{cls}">{html.escape(label)}</span>' for label, cls in nodes
    )
    return f'<div class="graph">{boxes}</div>'


def _render_table(rows: list[dict[str, str | int]]) -> str:
    """Render the dashboard table body: linked experiment, question, status, artifacts."""
    cells = []
    for row in rows:
        stem = html.escape(str(row["experiment"]))
        cells.append(
            "<tr>"
            f'<td><a href="/experiments/{stem}">{stem}</a></td>'
            f"<td>{html.escape(str(row['question']))}</td>"
            f"<td>{html.escape(str(row['status']))}</td>"
            f"<td>{row['artifacts']}</td>"
            "</tr>"
        )
    return "\n".join(cells)


def _render_artifacts(stem: str) -> str:
    """Render result files as links plus the embedded PNG, if present."""
    files = _result_files(stem)
    if not files:
        return "<p>No artifacts yet.</p>"
    links = "".join(
        f'<li><a href="/results/{html.escape(f.name)}">{html.escape(f.name)}</a></li>' for f in files
    )
    png = next((f for f in files if f.suffix == ".png"), None)
    image = (
        f'<p><img src="/results/{html.escape(png.name)}" alt="{html.escape(png.name)}"></p>' if png else ""
    )
    return f"<ul>{links}</ul>{image}"


def _verdict_options(saved: str) -> str:
    """Render the verdict select options, preselected to ``saved``."""
    return "".join(
        f'<option value="{verdict}"{" selected" if verdict == saved else ""}>{verdict}</option>'
        for verdict in _VERDICTS
    )


def _render_observations_form(stem: str, observations: dict[str, str]) -> str:
    """Render saved observations (if any) above the observation form."""
    saved_block = ""
    if observations:
        verdict = html.escape(str(observations.get("verdict", "")))
        text = html.escape(str(observations.get("observations", "")))
        updated = html.escape(str(observations.get("updated_at", "")))
        saved_block = (
            '<div class="saved">'
            f"<p><strong>verdict:</strong> {verdict}</p>"
            f"<p><strong>observations:</strong> {text}</p>"
            f"<p><strong>updated:</strong> {updated}</p>"
            "</div>"
        )
    selected = str(observations.get("verdict", _VERDICTS[0]))
    prefill = html.escape(str(observations.get("observations", "")))
    return (
        saved_block
        + f'<form method="post" action="/experiments/{html.escape(stem)}/observations">'
        + '<p><label for="verdict">verdict</label> '
        + f'<select name="verdict" id="verdict" required>{_verdict_options(selected)}</select></p>'
        + '<p><label for="observations">observations</label></p>'
        + f'<p><textarea name="observations" id="observations" rows="6" cols="60">{prefill}</textarea></p>'
        + '<p><button type="submit">save observation</button></p>'
        + "</form>"
    )


def _render_dashboard_page(rows: list[dict[str, str | int]]) -> str:
    """Render the full dashboard HTML page with minimal inline styling."""
    table_rows = _render_table(rows)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Broadway experiments</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; vertical-align: top; }}
  th {{ background: #f0f0f0; }}
  tr:nth-child(even) td {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>Broadway experiments</h1>
<table>
<thead>
<tr><th>experiment</th><th>question</th><th>status</th><th>artifacts</th></tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
</body>
</html>
"""


def _render_experiment_page(stem: str, question: str, profiles: dict[str, ScriptProfile]) -> str:
    """Render the full per-experiment HTML page (graph, artifacts, observation form)."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(stem)} — Broadway experiments</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  .graph {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem; margin: 1rem 0; }}
  .box {{ border: 1px solid #999; border-radius: 4px; padding: 0.4rem 0.8rem; font-size: 0.85rem; }}
  .box.green {{ background: #d9f2d9; }}
  .box.gray {{ background: #ececec; }}
  .box.strong {{ font-weight: bold; border-width: 2px; }}
  .arrow {{ color: #666; }}
  img {{ max-width: 100%; }}
</style>
</head>
<body>
<h1>{html.escape(stem)}</h1>
<p>{html.escape(question)}</p>
{_graph_html(stem, profiles)}
<h2>artifacts</h2>
{_render_artifacts(stem)}
<h2>observations</h2>
{_render_observations_form(stem, load_observations(stem))}
</body>
</html>
"""


def _script_for(name: str) -> Path | None:
    """Return the series script with stem ``name``, or None if unknown."""
    if not name or name.startswith("_"):
        return None
    candidate = EXPERIMENT_DIR / f"{name}.py"
    return candidate if candidate.is_file() else None


@app.get("/")
def index() -> HTMLResponse:
    """Serve the dashboard for the univariate series."""
    return HTMLResponse(_render_dashboard_page(census_rows()))


@app.get("/experiments/{name}", response_model=None)
def experiment_page(name: str) -> HTMLResponse | PlainTextResponse:
    """Serve the per-experiment page with graph, artifacts, and observation form."""
    script = _script_for(name)
    if script is None:
        return PlainTextResponse("unknown experiment", status_code=404)
    return HTMLResponse(
        _render_experiment_page(name, _question_from_docstring(script), series_profiles())
    )


@app.post("/experiments/{name}/observations", response_model=None)
async def save_observations(name: str, request: Request) -> RedirectResponse | PlainTextResponse:
    """Validate and persist an observation + verdict for a series script."""
    if _script_for(name) is None:
        return PlainTextResponse("unknown experiment", status_code=404)
    fields = parse_qs((await request.body()).decode("utf-8"))
    verdict = fields.get("verdict", [""])[0]
    if verdict not in _VERDICTS:
        return PlainTextResponse("invalid verdict", status_code=422)
    record = {
        "experiment": name,
        "verdict": verdict,
        "observations": fields.get("observations", [""])[0],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    OBSERVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    (OBSERVATIONS_DIR / f"{name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("saved observations for %s (verdict=%s)", name, verdict)
    return RedirectResponse(url=f"/experiments/{name}", status_code=303)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
