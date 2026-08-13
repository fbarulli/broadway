from __future__ import annotations

import os
from pathlib import Path

REPORTS_DIR = Path(os.getenv("BROADWAY_REPORTS_DIR", "reports"))
RESULTS_DIR = REPORTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"
