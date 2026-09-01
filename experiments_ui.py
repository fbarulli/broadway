"""Experiment UI — euromonitor only (post-composition launcher).

Serves the canonical dashboard app (`src/broadway/reports/experiments_dashboard.py`)
restricted to the euromonitor experiment series under
`project/euromonitor/`. The dashboard module is environment-configured
(`BROADWAY_EXPERIMENTS_ROOT` / `BROADWAY_DEFAULT_EXPERIMENT_SERIES` /
`BROADWAY_OBSERVATIONS_DIR`); this launcher pins those to the euromonitor
surface BEFORE the first import so the UI lists and renders only
euromonitor series, never the legacy experiment tree.

The pre-composition root `experiments_ui.py` was a full multi-series
duplicate of the dashboard module, deleted when experiments moved under
`project/` (5a028d8). This launcher reconnects the runnable entry point
for the euromonitor mission only.

Usage:
    bash scripts/uv.sh run python experiments_ui.py [--port 8000]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Lock the dashboard to the euromonitor surface BEFORE importing it, so its
# module-level constants pick up these values at first load.
os.environ["BROADWAY_EXPERIMENTS_ROOT"] = str(REPO_ROOT / "project" / "euromonitor")
os.environ["BROADWAY_DEFAULT_EXPERIMENT_SERIES"] = "."
os.environ["BROADWAY_OBSERVATIONS_DIR"] = str(
    REPO_ROOT / "project" / "euromonitor" / "observations"
)

import uvicorn  # noqa: E402

from broadway.reports import experiments_dashboard as ui  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(ui.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
