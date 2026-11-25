"""Feature engineering — placeholder for the tracked feature-construction step.

Owner: platform feature engineering. The EDA → model → story loop will use this
module to build and record the exact feature set (the feature manifest that
MLflow/W&B/Optuna projects). The existing ``builders``/``transformers`` remain
the live implementation; this file is the named owner until the manifest
contract lands, deliberately empty rather than a second implementation.

shortcut: blank module, replaced when the feature-manifest contract lands.
"""

from __future__ import annotations

__all__: list[str] = []
