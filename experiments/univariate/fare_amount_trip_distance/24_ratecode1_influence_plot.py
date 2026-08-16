"""24: influence plot — Cook's distance as circular bubbles (LR influence plot).

Leverage (hat values) on x vs externally studentized residuals on y, with
each trip drawn as a circle whose AREA is proportional to its Cook's
distance (McCulloch & Meeter 1983; car::influencePlot / statsmodels
influence_plot style). Reference boundaries: vertical lines at 2p/n and 3p/n
average leverage, horizontal lines at the studentized-residual t cutoff, and
the D > 4/n Cook's threshold separating influential (red bubbles) from
non-influential trips.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from _common import RESULTS, load_metered
from _ols_bp import fit_log_hc3

OUT = RESULTS / f"{Path(__file__).stem}.png"

ALPHA = 0.05
MAX_BUBBLE = 48  # max bubble radius points, statsmodels influence_plot default


def main() -> None:
    df = load_metered()
    model = fit_log_hc3(df)
    infl = model.get_influence()

    h = infl.hat_matrix_diag
    r = infl.resid_studentized_external
    cooks, _ = infl.cooks_distance

    n, k = len(df), int(model.params.size)
    cutoff = float(scipy_stats.t.ppf(1 - ALPHA / 2, model.df_resid))
    cooks_threshold = 4.0 / n
    infl_mask = cooks > cooks_threshold
    n_infl = int(infl_mask.sum())

    # bubble area proportional to Cook's distance (statsmodels scaling)
    ptp_cooks = float(np.ptp(cooks))
    psize = (cooks - cooks.min()) * (MAX_BUBBLE**2 - 8**2) / ptp_cooks + 8**2

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(h[~infl_mask], r[~infl_mask], s=psize[~infl_mask],
               alpha=0.35, color="gray", edgecolors="none", label="non-influential")
    ax.scatter(h[infl_mask], r[infl_mask], s=psize[infl_mask],
               alpha=0.8, color="red", edgecolors="black",
               label=f"influential D > 4/n (n={n_infl})")
    ax.axvline(2 * k / n, color="blue", linestyle="--", linewidth=1, label="2p/n leverage")
    ax.axvline(3 * k / n, color="blue", linestyle=":", linewidth=1)
    ax.axhline(cutoff, color="green", linestyle="--", linewidth=1,
               label=f"studentized residual ±t cutoff ({cutoff:.2f})")
    ax.axhline(-cutoff, color="green", linestyle="--", linewidth=1)
    for i in np.argsort(cooks)[-5:][::-1]:
        ax.annotate(f"{df['trip_distance'].iloc[i]:.1f}mi",
                    (h[i], r[i]), fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("Leverage (hat values)")
    ax.set_ylabel("Externally studentized residuals")
    ax.set_title(f"Influence plot — log-fare model (N={n}, bubble area ∝ Cook's distance)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)

    print(f"influential trips (Cook's D > 4/n = {cooks_threshold:.2e}): {n_infl} of {n}")
    print(f"studentized-residual cutoff: ±{cutoff:.2f} | leverage cutoffs: {2*k/n:.4f}, {3*k/n:.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
