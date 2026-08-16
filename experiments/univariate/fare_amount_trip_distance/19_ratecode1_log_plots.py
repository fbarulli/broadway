"""19: residuals-vs-fitted + Q-Q plots for the log-fare HC3 model (step 18)."""

from pathlib import Path

from _common import RESULTS, load_metered
from _ols_bp import fit_log_hc3, plot_log_resid_qq

OUT = RESULTS / f"{Path(__file__).stem}.png"


def main() -> None:
    df = load_metered()
    model = fit_log_hc3(df)
    plot_log_resid_qq(model, OUT, suptitle="RatecodeID == 1, log-fare (HC3)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
