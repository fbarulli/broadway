"""01: confidence intervals for feature means.

Evidence-only: computes a 95% normal-approximation CI for the mean of every
numeric column in the named fare_prediction_1m sample, excluding the target.
"""

import numpy as np
import pandas as pd

from _common import RESULTS, TARGET, load_sample, numeric_features

CONFIDENCE = 0.95
Z_95 = 1.959963984540054
CSV_OUT = RESULTS / "01_feature_mean_ci.csv"


def _clean_values(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def mean_ci(feature: str, series: pd.Series) -> dict[str, object]:
    values = _clean_values(series)
    n = len(values)

    row: dict[str, object] = {
        "feature": feature,
        "n": n,
        "missing": int(series.isna().sum()),
        "confidence": CONFIDENCE,
        "method": "normal_approximation",
    }

    if n == 0:
        row.update(
            {
                "mean": np.nan,
                "std": np.nan,
                "se_mean": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "warning": "no finite values",
            }
        )
        return row

    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0

    warning = ""
    if std == 0.0:
        warning = "zero variance"
    elif n < 100:
        warning = "small sample for normal approximation"

    row.update(
        {
            "mean": mean,
            "std": std,
            "se_mean": se,
            "ci_low": mean - Z_95 * se,
            "ci_high": mean + Z_95 * se,
            "warning": warning,
        }
    )
    return row


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = load_sample()
    features = numeric_features(df)

    rows = [mean_ci(feature, df[feature]) for feature in features]
    out = pd.DataFrame(rows).set_index("feature")
    out.to_csv(CSV_OUT)

    print(f"features: {len(out)}")
    print(out)
    print(f"wrote {CSV_OUT}")


if __name__ == "__main__":
    main()
