# Stats Library API

Public contract for `src/broadway/stats/`. Every function below is the source
of truth for its signature — implementations must match exactly. No function
in the library reads config or filesystem paths; it receives plain data
(DataFrames, arrays) and returns plain results. Dataset specifics live in
`project/data.py`.

The library is **pandas/numpy only** — there is no Spark dependency anywhere
in `src/broadway/stats/`.

## Conventions

- `groups: dict[str, np.ndarray]` — group name → array of target values
- Every test returns an `AnalysisPlan` (from `plan.py`), never a bare float.
- Effect sizes are always computed in pairs (eta² AND omega², Cohen's d AND
  Hedges' g); the report layer decides which is more trustworthy from group
  sizes — the functions themselves never pick one.

---

## guards.py

```python
def validate_groups(groups: dict[str, np.ndarray]) -> list[str]
    # fail loudly on empty / non-finite / zero-variance / fewer-than-2 groups;
    # returns non-fatal warnings (e.g. "group 'x' has zero variance")
```

## plan.py

```python
class AnalysisPlan(BaseModel):
    script: str                    # e.g. "04_anova"
    analysis_type: str             # "group_comparison" | "regression" | "timeseries"
    test_name: str                 # "one-way ANOVA", "Welch's ANOVA", ...
    statistics: dict[str, float]   # {"p_value": ..., "statistic": ..., ...}
    effect_sizes: dict[str, float] # {"eta_squared": ..., "omega_squared": ...} | {"epsilon_squared": ...} | {"cohens_d": ..., "hedges_g": ...}
    threshold_context: dict[str, float | bool]  # {"imbalance_ratio": ..., "any_small_group": ...}
    reason: list[str]              # human-readable evidence trail
    warnings: list[str]            # "underpowered", "n imbalance", ...
    passed: bool                   # omnibus test passed at alpha
    next_step: str | None          # which analysis follows from this one
    analysis_goal: str | None = None     # from the AnalysisContract
    sample_name: str | None = None       # SampleSpec name (when run with --sample)
    sample_role: SampleRole | None = None

def save_plan(plan: AnalysisPlan, path: Path) -> None
def load_plan(path: Path) -> AnalysisPlan
```

## effect_size.py

```python
def eta_squared(f_stat: float, df1: int, df2: int) -> float
def omega_squared(f_stat: float, df1: int, df2: int, n_total: int) -> float
def epsilon_squared(h_stat: float, k: int, n: int) -> float
    # rank-based ε² = (H - k + 1) / (N - k), clamped to [0, 1]
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

def plot_residuals_vs_fitted(model, out_path: str) -> None
    # residual-vs-fitted scatter only → save PNG

def plot_residuals_qq(model, out_path: str) -> None
    # Q-Q plot of residuals only → save PNG

def plot_residuals_histogram(model, out_path: str) -> None
    # residual histogram only → save PNG

def plot_cooks_distance(model, out_path: str) -> None
    # Cook's distance by observation index, 4/n threshold line → save PNG

def mean_specification_diagnostic(model, out_path: str) -> DiagnosticResult
    # persists residual-vs-fitted plot, returns Question→Evidence→Ramification

def constant_variance_diagnostic(model, out_path: str) -> DiagnosticResult
    # persists residual-vs-fitted plot + Breusch-Pagan, returns typed result

def influence_diagnostic(model, out_path: str) -> DiagnosticResult
    # persists Cook's-distance plot + max/4-n summary, returns typed result

def residual_distribution_diagnostic(model, out_path: str) -> DiagnosticResult
    # persists Q-Q plot + Jarque-Bera, returns typed result
```

## diagnostic_models.py

```python
class DiagnosticResult(BaseModel):
    question: str      # the diagnostic question being answered
    evidence: list[str]  # human-readable evidence trail
    ramification: str  # what to do if the evidence suggests a problem
    warnings: list[str]  # non-fatal concerns, defaults to []
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

## baseline.py

```python
def train_lgbm(X: pd.DataFrame, y: np.ndarray, **params) -> object
    # LGBMRegressor; hyperparams passed in, no defaults

def evaluate(model, X: pd.DataFrame, y: np.ndarray, tail_quantile: float) -> dict
    # {"mae": ..., "rmse": ..., "tail_mae": ...}
```

## describe.py (pipeline step)

```python
class GroupStat(BaseModel):
    n: int
    mean: float | None
    std: float | None

class GroupSummary(BaseModel):
    group_column: str
    source_group_column: str     # column actually present in the sample (after column_mapping)
    target: str
    total_n: int
    source_path: str
    sample_name: str
    sample_role: SampleRole
    groups: dict[str, GroupStat]  # ALL configured groups, incl. n=0
    absent_groups: list[str]
    imbalance_ratio: float        # evidence only — NO balanced/unbalanced verdict
    proportions: dict[str, float]
    warnings: list[str]

def describe(df: pd.DataFrame, group_column: str, source_group_column: str, group_values: list[str], target: str, source_path: str, sample_name: str, sample_role: SampleRole) -> GroupSummary

def plot_group_distribution(df: pd.DataFrame, source_group_column: str, group_column: str, group_values: list[str], target: str, out_path: Path) -> None
    # boxplot of target by group → save PNG

def plot_group_sizes(summary: GroupSummary, out_path: Path) -> None
    # bar chart of group sizes (imbalance evidence) → save PNG

def run(cfg: PipelineConfig, sample: SampleSpec) -> None
    # requires a hypothesis block; writes describe.json + reports/results/describe.md
    # + figures + a describe lineage sidecar
```

## module.py (pipeline step)

`run(cfg: PipelineConfig, sample: SampleSpec | None = None) -> None` — requires a
`hypothesis` block (`group_column`, `group_values`); builds groups from the
canonical dataset (or a `--sample` parquet with `column_mapping` applied), runs
`run_anova`, saves an `AnalysisPlan` JSON to `cfg.stats.output_dir`, and writes a
`stats` lineage sidecar.
