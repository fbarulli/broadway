"""18: OLS on log1p(fare_amount) ~ trip_distance + duration_minutes, HC3 SEs.

Log-transforms the target (log1p safely handles $0 fares) to pull in the
heavy right tail seen in steps 13-15 (kurtosis ~71) and fits with HC3
robust SEs. Coefficients are on the log-fare scale. Working dataset:
ratecode1 (42,848 rows after the duration filter).
"""

from _common import load_metered
from _ols_bp import fit_log_hc3


def main() -> None:
    df = load_metered()
    model = fit_log_hc3(df)

    print(f"metered rows (duration > 0): {len(df)}")
    print(model.summary())


if __name__ == "__main__":
    main()
