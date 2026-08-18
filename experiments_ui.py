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
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs

import uvicorn
from fastapi import FastAPI, Query, Request
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
_NUMBER_PREFIX = re.compile(r"^\d+[a-z]?:\s*")
_STEM_PREFIX = re.compile(r"^\d+")
_SLUG_RE = re.compile(r"[a-z0-9_]+")
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
        f'<span class="{cls}"><a href="{_node_href(label)}">{html.escape(label)}</a></span>'
        for label, cls in nodes
    )
    return f'<div class="graph">{boxes}</div>'


def _node_href(label: str) -> str:
    """Href for a graph node: the dashboard for upstream, the step page otherwise."""
    if label == _UPSTREAM_LABEL:
        return "/"
    return f"/experiments/{html.escape(label)}"


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


def _render_observations_form(
    stem: str, observations: dict[str, str], next_url: str = ""
) -> str:
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
    hidden = (
        f'<input type="hidden" name="next" value="{html.escape(next_url)}">' if next_url else ""
    )
    return (
        saved_block
        + f'<form method="post" action="/experiments/{html.escape(stem)}/observations">'
        + hidden
        + '<p><label for="verdict">verdict</label> '
        + f'<select name="verdict" id="verdict" required>{_verdict_options(selected)}</select></p>'
        + '<p><label for="observations">observations</label></p>'
        + f'<p><textarea name="observations" id="observations" rows="6" cols="60">{prefill}</textarea></p>'
        + '<p><button type="submit">save observation</button></p>'
        + "</form>"
    )


def _prev_next_hrefs(stems: list[str], current: str, href: str) -> tuple[str, str]:
    """Return (prev, next) step hrefs around ``current`` in sorted ``stems``."""
    idx = stems.index(current)
    prev = f"{href}{stems[idx - 1]}" if idx > 0 else ""
    nxt = f"{href}{stems[idx + 1]}" if idx < len(stems) - 1 else ""
    return prev, nxt


def _series_strip(
    stems: list[str],
    current: str,
    href_for: Callable[[str], str],
    prev_href: str,
    next_href: str,
) -> str:
    """Render the series strip: prev/next plus one mini-link per stem (current highlighted)."""
    items = []
    for stem in stems:
        cls = "strip-link" + (" current" if stem == current else "")
        items.append(
            f'<a class="{cls}" href="{html.escape(href_for(stem))}">{html.escape(stem)}</a>'
        )
    prev = (
        f'<a class="strip-nav" href="{html.escape(prev_href)}">« prev</a>'
        if prev_href
        else '<span class="strip-nav strip-dim">« prev</span>'
    )
    nxt = (
        f'<a class="strip-nav" href="{html.escape(next_href)}">next »</a>'
        if next_href
        else '<span class="strip-nav strip-dim">next »</span>'
    )
    return f'<nav class="strip">{prev}{"".join(items)}{nxt}</nav>'


def _next_prefix(stem: str, stems: list[str]) -> str:
    """Pick the filename prefix for a step inserted after ``stem``."""
    match = _STEM_PREFIX.match(stem)
    base = int(match.group(0)) if match else 0
    candidate = f"{base + 1:02d}"
    if not any(s.startswith(candidate) for s in stems):
        return candidate
    letter = "a"
    while any(s.startswith(f"{base:02d}{letter}") for s in stems):
        letter = chr(ord(letter) + 1)
    return f"{base:02d}{letter}"


def _scaffold_text(prefix: str, question: str) -> str:
    """Return the scaffolded step-script text for an inserted step."""
    safe = question.replace('"""', "'''")
    return (
        f'"""{prefix}: {safe}"""\n'
        "\n"
        "from pathlib import Path\n"
        "\n"
        "import pandas as pd\n"
        "\n"
        "from _common import RESULTS\n"
        "from project.working import load_metered\n"
        "\n"
        'OUT = RESULTS / f"{Path(__file__).stem}.png"\n'
        "\n"
        "\n"
        "def main() -> None:\n"
        "    RESULTS.mkdir(parents=True, exist_ok=True)\n"
        "    df = load_metered()\n"
        '    print(f"rows: {len(df)}")\n'
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
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


def _render_experiment_page(
    stem: str,
    question: str,
    stems: list[str],
    profiles: dict[str, ScriptProfile],
) -> str:
    """Render the full per-experiment HTML page (strip, graph, artifacts, observation form)."""
    prev_href, next_href = _prev_next_hrefs(stems, stem, "/experiments/")
    strip = _series_strip(stems, stem, lambda s: f"/experiments/{s}", prev_href, next_href)
    nav = (
        "<p>"
        f'<a href="/series?from={html.escape(stem)}">show with next</a>'
        f' | <a href="/experiments/{html.escape(stem)}/new">＋ new step after this</a>'
        "</p>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(stem)} — Broadway experiments</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  .strip {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; margin: 1rem 0; font-size: 0.85rem; }}
  .strip a, .strip span {{ padding: 0.15rem 0.4rem; border: 1px solid #ccc; border-radius: 3px; text-decoration: none; color: #222; }}
  .strip a.current {{ background: #d9f2d9; border-color: #2a7; }}
  .strip .strip-dim {{ color: #999; border-style: dashed; }}
  .graph {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem; margin: 1rem 0; }}
  .box {{ border: 1px solid #999; border-radius: 4px; padding: 0.4rem 0.8rem; font-size: 0.85rem; }}
  .box a {{ text-decoration: none; color: inherit; }}
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
{strip}
{nav}
{_graph_html(stem, profiles)}
<h2>artifacts</h2>
{_render_artifacts(stem)}
<h2>observations</h2>
{_render_observations_form(stem, load_observations(stem))}
</body>
</html>
"""


def _render_series_cards(
    stems: list[str], profiles: dict[str, ScriptProfile], back_url: str
) -> str:
    """Render one card per step: header link, question, figure, observation form."""
    cards = []
    for stem in stems:
        script = _script_for(stem)
        question = _question_from_docstring(script) if script else ""
        png = next((f for f in _result_files(stem) if f.suffix == ".png"), None)
        figure = (
            f'<p><img src="/results/{html.escape(png.name)}" alt="{html.escape(png.name)}"></p>'
            if png
            else "<p>No figure yet.</p>"
        )
        cards.append(
            '<section class="card">'
            f'<h2><a href="/experiments/{html.escape(stem)}">{html.escape(stem)}</a></h2>'
            f"<p>{html.escape(question)}</p>"
            f"{figure}"
            f"{_render_observations_form(stem, load_observations(stem), back_url)}"
            "</section>"
        )
    return "\n".join(cards)


def _render_series_page(
    from_stem: str,
    to_stem: str,
    stems: list[str],
    profiles: dict[str, ScriptProfile],
) -> str:
    """Render the multi-step series view: strip plus one card per step in range."""
    from_idx = stems.index(from_stem)
    to_idx = stems.index(to_stem)
    prev_href = ""
    if from_idx > 0:
        prev_href = f"/series?from={stems[from_idx - 1]}&to={stems[to_idx - 1]}"
    next_href = ""
    if to_idx < len(stems) - 1:
        next_href = f"/series?from={stems[from_idx + 1]}&to={stems[to_idx + 1]}"
    strip = _series_strip(stems, from_stem, lambda s: f"/series?from={s}", prev_href, next_href)
    back_url = f"/series?from={from_stem}&to={to_stem}"
    cards = _render_series_cards(stems[from_idx : to_idx + 1], profiles, back_url)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>series — Broadway experiments</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  .strip {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; margin: 1rem 0; font-size: 0.85rem; }}
  .strip a, .strip span {{ padding: 0.15rem 0.4rem; border: 1px solid #ccc; border-radius: 3px; text-decoration: none; color: #222; }}
  .strip a.current {{ background: #d9f2d9; border-color: #2a7; }}
  .strip .strip-dim {{ color: #999; border-style: dashed; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; }}
  .card {{ border: 1px solid #ccc; border-radius: 6px; padding: 0.8rem 1rem; flex: 1 1 22rem; }}
  .card img {{ max-width: 100%; }}
</style>
</head>
<body>
<h1>series</h1>
{strip}
<div class="cards">
{cards}
</div>
</body>
</html>
"""


def _render_new_step_page(stem: str) -> str:
    """Render the insert-a-step form page."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>new step after {html.escape(stem)} — Broadway experiments</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  label {{ display: block; margin-top: 0.6rem; }}
  input, textarea {{ margin-top: 0.2rem; }}
</style>
</head>
<body>
<h1>new step after {html.escape(stem)}</h1>
<form method="post" action="/experiments/{html.escape(stem)}/new">
<p><label for="name">name (slug: [a-z0-9_]+)</label>
<input name="name" id="name" required pattern="[a-z0-9_]+" size="40"></p>
<p><label for="question">question</label>
<textarea name="question" id="question" rows="3" cols="60" required></textarea></p>
<p><button type="submit">create step</button></p>
</form>
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
    """Serve the per-experiment page with strip, graph, artifacts, and observation form."""
    script = _script_for(name)
    if script is None:
        return PlainTextResponse("unknown experiment", status_code=404)
    stems = [s.stem for s in series_scripts()]
    return HTMLResponse(
        _render_experiment_page(name, _question_from_docstring(script), stems, series_profiles())
    )


@app.get("/series", response_model=None)
def series_page(
    from_: str = Query(default="", alias="from"),
    to: str = Query(default=""),
) -> HTMLResponse | PlainTextResponse:
    """Serve the multi-step series view: one card per step from ``from_`` to ``to``."""
    stems = [script.stem for script in series_scripts()]
    if not stems:
        return PlainTextResponse("no experiments", status_code=404)
    from_stem = from_ or stems[0]
    if from_stem not in stems or (to and to not in stems):
        return PlainTextResponse("unknown experiment", status_code=404)
    from_idx = stems.index(from_stem)
    to_idx = stems.index(to) if to else min(from_idx + 1, len(stems) - 1)
    if to_idx < from_idx:
        from_idx, to_idx = to_idx, from_idx
    return HTMLResponse(
        _render_series_page(stems[from_idx], stems[to_idx], stems, series_profiles())
    )


@app.get("/experiments/{name}/new", response_model=None)
def new_step_page(name: str) -> HTMLResponse | PlainTextResponse:
    """Serve the form to insert a new step after ``name``."""
    if _script_for(name) is None:
        return PlainTextResponse("unknown experiment", status_code=404)
    return HTMLResponse(_render_new_step_page(name))


@app.post("/experiments/{name}/new", response_model=None)
async def create_step(name: str, request: Request) -> RedirectResponse | PlainTextResponse:
    """Validate and scaffold a new step script after ``name``; 303 to its page."""
    if _script_for(name) is None:
        return PlainTextResponse("unknown experiment", status_code=404)
    fields = parse_qs((await request.body()).decode("utf-8"))
    new_name = fields.get("name", [""])[0]
    question = fields.get("question", [""])[0]
    if not _SLUG_RE.fullmatch(new_name) or not question.strip():
        return PlainTextResponse("invalid name or question", status_code=422)
    stems = [script.stem for script in series_scripts()]
    prefix = _next_prefix(name, stems)
    stem = f"{prefix}_{new_name}"
    (EXPERIMENT_DIR / f"{stem}.py").write_text(
        _scaffold_text(prefix, question), encoding="utf-8"
    )
    logger.info("created step script %s after %s", stem, name)
    return RedirectResponse(url=f"/experiments/{stem}", status_code=303)


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
    next_url = fields.get("next", [""])[0]
    if not (next_url.startswith("/") and not next_url.startswith("//")):
        next_url = ""
    return RedirectResponse(url=next_url or f"/experiments/{name}", status_code=303)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
