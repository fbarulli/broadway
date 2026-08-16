"""15: fare ~ trip_distance + duration_minutes, plain OLS vs HC3 robust SEs.

Step 14 fit fare ~ trip_distance alone with HC3. This step adds trip
duration (derived from pickup/dropoff datetimes, non-positive durations
dropped) as a second predictor, and fits BOTH plain OLS and HC3 so the
robust standard errors can be compared directly. HC3 changes only
SEs/p-values/CIs — never coefficients or R^2.
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from _common import RATECODE1_PARQUET


def load_metered() -> pd.DataFrame:
    df = pd.read_parquet(RATECODE1_PARQUET)
    df["trip_duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds()
    df["duration_minutes"] = df["trip_duration"] / 60
    return df[df["trip_duration"] > 0]


def fit_both(df: pd.DataFrame) -> tuple[RegressionResultsWrapper, RegressionResultsWrapper]:
    X = sm.add_constant(df[["trip_distance", "duration_minutes"]])
    y = df["fare_amount"]
    return sm.OLS(y, X).fit(), sm.OLS(y, X).fit(cov_type="HC3")


def print_comparison(plain: RegressionResultsWrapper, hc3: RegressionResultsWrapper) -> None:
    table = pd.DataFrame(
        {
            "coef": plain.params,
            "se_plain": plain.bse,
            "se_hc3": hc3.bse,
            "se_ratio_hc3/plain": hc3.bse / plain.bse,
            "p_plain": plain.pvalues,
            "p_hc3": hc3.pvalues,
        }
    )
    print(table.round(6).to_string())


def main() -> None:
    df = load_metered()
    plain, hc3 = fit_both(df)

    print(f"metered rows (duration > 0): {len(df)}")
    print("\n=== Plain OLS ===")
    print(plain.summary())
    print("\n=== HC3 robust OLS ===")
    print(hc3.summary())
    print("\n=== Side-by-side: coefficients / SEs / p-values ===")
    print_comparison(plain, hc3)
    print(
        "\nNote: coefficients and R^2 are identical between the two; HC3 "
        "only widens the standard errors (and shifts p-values/CIs) to stay "
        "valid under the heteroskedasticity Breusch-Pagan rejects."
    )


if __name__ == "__main__":
    main()
