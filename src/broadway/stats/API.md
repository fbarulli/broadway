# Stats Library API

Public contract for `src/broadway/stats/`. Every function below is the source
of truth for its signature — implementations must match exactly. No function
in the library reads config or filesystem paths; it receives plain data
(DataFrames, arrays) and returns plain results. Dataset specifics live in
`projects/<name>/data.py`.

## Conventions

- `groups: dict[str, np.ndarray]` — group name → array of target values
- Every test returns an `AnalysisPlan` (from `plan.py`), never a bare float.
- Effect sizes are always computed in pairs (eta² AND omega², Cohen's d AND
  Hedges' g); the report layer decides which is more trustworthy from group
  sizes — the functions themselves never pick one.

---

## base.py

```python
def get_spark_session(app_name: str = "stats-learning") -> SparkSession
    # create a local[*] Spark session

def stratified_sample(df: pd.DataFrame, group_col: str, frac: float, random_state: int) -> pd.DataFrame
    # stratified sample preserving per-group proportions; returns reset-index frame
```

## plan.py

```python
@dataclass
class AnalysisPlan:
    script: str                    # e.g. "04_anova"
    analysis_type: str             # "group_comparison" | "regression" | "timeseries"
    test_name: str                 # "one-way ANOVA", "Welch's ANOVA", ...
    statistics: dict               # {"p_value": ..., "statistic": ..., ...}
    effect_sizes: dict             # {"eta_squared": ..., "omega_squared": ...} | {"cohens_d": ..., "hedges_g": ...}
    threshold_context: dict        # {"imbalance_ratio": ..., "any_small_group": ...}
    reason: list[str]              # human-readable evidence trail
    warnings: list[str]            # "underpowered", "n imbalance", ...
    passed: bool                   # omnibus test passed at alpha
    next_step: str | None          # which analysis follows from this one

def save_plan(plan: AnalysisPlan, path: Path) -> None
def load_plan(path: Path) -> AnalysisPlan
```

## effect_size.py

```python
def eta_squared(f_stat: float, df1: int, df2: int) -> float
def omega_squared(f_stat: float, df1: int, df2: int, n_total: int) -> float
def cohens_d(a: np.ndarray, b: np.ndarray) -> float
def hedges_g(a: np.ndarray, b: np.ndarray) -> float
    # Cohen's d corrected for small sample bias; negligible for large n
def group_imbalance(group_sizes: dict[str, int]) -> float
    # largest / smallest group ratio; 1.0 when balanced
```

## diagnostics.py

```python
def bp_test(resid: np.ndarray, exog: np.ndarray) -> tuple[float, float]
    # Breusch-Pagan → (statistic, p_value)

def jb_test(resid: np.ndarray) -> tuple[float, float, float, float]
    # Jarque-Bera → (statistic, p_value, skew, kurtosis)

def durbin_watson(resid: np.ndarray) -> float

def plot_residuals(model, out_path: str) -> None
    # residuals vs fitted, Q-Q, histogram → save PNG
```

## anova.py

```python
def run_anova(groups: dict[str, np.ndarray], alpha: float = 0.05) -> AnalysisPlan
    # one-way ANOVA + eta²/omega²

def run_welch(groups: dict[str, np.ndarray], alpha: float = 0.05) -> AnalysisPlan
    # Welch's ANOVA (unequal variance)

def run_kruskal(groups: dict[str, np.ndarray], alpha: float = 0.05) -> AnalysisPlan
    # Kruskal-Wallis (non-parametric)
```

## assumptions.py

```python
def run_levene(groups: dict[str, np.ndarray]) -> dict[str, float]
    # {"statistic": ..., "p_value": ...}

def check_normality(groups: dict[str, np.ndarray]) -> dict[str, dict[str, float]]
    # group name → {"skew": ..., "kurtosis": ..., "shapiro_p": ...}
```

## post_hoc.py

```python
def games_howell(df: pd.DataFrame, dv: str, between: str) -> pd.DataFrame
    # pairwise Games-Howell; columns include "A", "B", "diff", "pval",
    # "cohens_d", "hedges_g", "effect_size_note"
```

## regression.py

```python
def fit_ols(df: pd.DataFrame, formula: str) -> object
    # statsmodels OLS fit

def fit_robust(model, cov_type: str = "HC3") -> object
    # refit with robust covariance

def bp_jb(model) -> dict
    # {"bp_stat": ..., "bp_pval": ..., "jb_stat": ..., "jb_pval": ...,
    #  "skew": ..., "kurtosis": ...}
```

## time_series.py

```python
def durbin_watson_test(resid: np.ndarray) -> float

def plot_acf(resid: np.ndarray, lags: int, out_path: str) -> None
    # ACF plot over bounded lag window → save PNG
```

## module.py (pipeline step)

`run(cfg: PipelineConfig) -> None` — orchestrates the library against the
configured dataset: build groups, run ANOVA (+ assumptions + post-hoc when
applicable), save an `AnalysisPlan` JSON to `cfg.stats.output_dir`.
