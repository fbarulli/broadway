"""Bare-bones FastAPI census dashboard for Broadway experiments.

Serves a single read-only HTML table listing every experiment script under
``experiments/*/*/*.py`` with its category, docstring question, run status and
artifact count. No templates, no JS, no external assets — a skeleton to be
extended later.
"""

import ast
import html
import re
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

EXPERIMENTS_ROOT = Path(__file__).resolve().parent / "experiments"
RESULTS_ROOT = EXPERIMENTS_ROOT / "results"
_NUMBER_PREFIX = re.compile(r"^\d+:\s*")

app = FastAPI()


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


def _result_files(category: str, stem: str) -> list[Path]:
    """Return result files for ``stem`` under ``results/<category>``."""
    results_dir = RESULTS_ROOT / category
    if not results_dir.is_dir():
        return []
    return sorted(results_dir.glob(f"{stem}.*"))


def _numeric_sort_key(name: str) -> tuple[bool, int, str]:
    """Sort key: numeric prefix first, then name."""
    match = re.match(r"(\d+)", name)
    if match:
        return (False, int(match.group(1)), name)
    return (True, 0, name)


def census_rows() -> list[dict[str, str | int]]:
    """Walk ``experiments/*/*/*.py`` and build one row per experiment."""
    rows: list[dict[str, str | int]] = []
    for script in sorted(EXPERIMENTS_ROOT.glob("*/*/*.py")):
        if script.name.startswith("_"):
            continue
        stem = script.stem
        category = f"{script.parent.parent.name}/{script.parent.name}"
        results = _result_files(category, stem)
        rows.append(
            {
                "experiment": stem,
                "category": category,
                "question": _question_from_docstring(script),
                "status": "ran" if results else "no outputs",
                "artifacts": len(results),
            }
        )
    rows.sort(
        key=lambda row: (_numeric_sort_key(str(row["category"])), _numeric_sort_key(str(row["experiment"])))
    )
    return rows


def _render_table(rows: list[dict[str, str | int]]) -> str:
    """Render the census table body as escaped HTML."""
    cells = []
    for row in rows:
        values = [html.escape(str(row[key])) for key in ("experiment", "category", "question", "status", "artifacts")]
        cells.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    return "\n".join(cells)


def _render_page(rows: list[dict[str, str | int]]) -> str:
    """Render the full HTML page with inline minimal styling."""
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
<tr><th>experiment</th><th>category</th><th>question</th><th>status</th><th>artifacts</th></tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
</body>
</html>
"""


@app.get("/")
def index() -> HTMLResponse:
    """Serve the census dashboard page."""
    return HTMLResponse(_render_page(census_rows()))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
