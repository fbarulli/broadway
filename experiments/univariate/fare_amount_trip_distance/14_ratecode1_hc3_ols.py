"""14: OLS with HC3 robust standard errors on the RatecodeID == 1 dataset.

Step 13 showed Breusch-Pagan still rejects on the ratecode1 subset
(heteroskedastic), so the plain OLS p-values/SEs are not trustworthy.
HC3 robust covariance does not change the coefficients or R^2 — it only
makes the standard errors (and hence p-values/CIs) valid despite
heteroskedasticity. Uses the same ratecode1 dataset as 11/12/13.
"""

from pathlib import Path

import polars as pl
import statsmodels.api as sm

from _common import RATECODE1_PARQUET, RESULTS
from _ols_bp import plot_resid_vs_fitted

OUT = RESULTS / f"{Path(__file__).stem}.png"


def main() -> None:
    metered_df = pl.read_parquet(RATECODE1_PARQUET).to_pandas()

    X = sm.add_constant(metered_df["trip_distance"])
    y = metered_df["fare_amount"]

    # The magic happens here: cov_type='HC3'
    model = sm.OLS(y, X).fit(cov_type="HC3")

    print(f"ratecode1 rows: {len(metered_df)}")
    print(model.summary())
    print(
        "\nNote: HC3 leaves coefficients and R^2 unchanged vs plain OLS "
        "(step 13); only std err / t / p-values change — they are now valid "
        "despite the heteroskedasticity Breusch-Pagan keeps rejecting."
    )

    plot_resid_vs_fitted(model, OUT, suptitle="RatecodeID == 1 (HC3)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
