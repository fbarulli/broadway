"""24: residuals vs leverage — Cook's distance contour boundaries (classic R plot.lm style).

Leverage (hat values) on x, internally studentized residuals on y, with
hyperbolic contour arcs at Cook's distance D = 0.5 and D = 1.0 (red dashed).
Trips beyond the D = 1 arc are flagged as highly influential. Contour equation
(R plot.lm which=5): y = ±sqrt(D * (1 - h) / h * k), k = number of parameters.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _common import RESULTS, load_metered
from _ols_bp import fit_log_hc3
from matplotlib.axes import Axes

OUT = RESULTS / f"{Path(__file__).stem}.png"

COOK_LEVELS = (0.5, 1.0)  # Cook's-distance contour levels to draw
GRID_X = 200              # x resolution of each contour arc
POINT_SIZE = 12
FLAG_SIZE = 18


def cook_contours(lev: np.ndarray, resid: np.ndarray, k: int,
                  level: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the visible positive branch of a Cook's-distance contour arc."""
    xs = np.linspace(lev.min(), lev.max(), GRID_X)
    ys = np.sqrt(level * (1.0 - xs) / xs * k)
    ymax = float(np.abs(resid).max()) * 1.05
    keep = ys <= ymax
    return xs[keep], ys[keep]


def plot_resid_leverage(ax: Axes, h: np.ndarray, r: np.ndarray,
                        cooks: np.ndarray, k: int) -> None:
    """Draw the residuals-vs-leverage scatter with Cook's-distance arcs."""
    ax.scatter(h, r, s=POINT_SIZE, alpha=0.35, color="black", label="trips")
    for level in COOK_LEVELS:
        xs, ys = cook_contours(h, r, k, level)
        ax.plot(xs, ys, "r--", linewidth=1, label=f"Cook's distance D = {level}")
        ax.plot(xs, -ys, "r--", linewidth=1)
    flagged = cooks > 1.0
    if flagged.any():
        ax.scatter(h[flagged], r[flagged], s=FLAG_SIZE, facecolors="none",
                   edgecolors="red",
                   label=f"influential D > 1 (n={int(flagged.sum())})")
    ax.set_xlabel("Leverage (hat values)")
    ax.set_ylabel("Internally studentized residuals")
    ax.set_title("Residuals vs Leverage with Cook's Distance Boundaries")
    ax.grid(True)
    ax.legend(fontsize=8)


def main() -> None:
    df = load_metered()
    model = fit_log_hc3(df)
    infl = model.get_influence()

    h = infl.hat_matrix_diag
    r = infl.resid_studentized_internal
    cooks, _ = infl.cooks_distance
    k = int(model.params.size)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_resid_leverage(ax, h, r, cooks, k)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)

    n_flagged = int((cooks > 1.0).sum())
    print(f"flagged influential trips (Cook's D > 1.0): {n_flagged} of {len(df)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
