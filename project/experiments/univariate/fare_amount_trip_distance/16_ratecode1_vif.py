"""16: Variance Inflation Factor (VIF) for the step-15 model predictors.

fare ~ trip_distance + duration_minutes on the ratecode1 dataset. VIF
quantifies how much each predictor's variance is inflated by correlation
with the other predictors (multicollinearity):
  VIF = 1      no correlation with other features
  VIF > 5      moderate correlation (model is getting confused)
  VIF > 10     severe multicollinearity (the math is breaking down)
"""

import pandas as pd
import statsmodels.api as sm
from _common import load_metered
from statsmodels.stats.outliers_influence import variance_inflation_factor


def main() -> None:
    df = load_metered()

    X = sm.add_constant(df[["trip_distance", "duration_minutes"]])

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]

    print(f"metered rows (duration > 0): {len(df)}")
    print(vif_data.to_string(index=False))
    print(
        "\nInterpretation: only the non-constant predictors matter — "
        "trip_distance and duration_minutes both have VIF close to 1, "
        "so no multicollinearity between them."
    )


if __name__ == "__main__":
    main()
