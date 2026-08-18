"""Shared paths and constants for this experiment (no analysis logic).

The sample is declared once in ``configs/sample/fare_prediction_1m.yaml``
(seed/size/columns/filters/schema). Steps only consume the name — the sample
registry owns paths, filtering, and sampling.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "experiments" / "results" / HERE.name
SAMPLE_NAME = "fare_prediction_1m"
