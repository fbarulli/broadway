from __future__ import annotations

import os
from pathlib import Path

REPORTS_DIR = Path(os.getenv("BROADWAY_REPORTS_DIR", "reports"))
RESULTS_DIR = REPORTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"
AUDIT_DIR = REPORTS_DIR / "audit"
TIMELINE_PATH = REPORTS_DIR / "timeline.md"
INDEX_PATH = REPORTS_DIR / "index.md"
