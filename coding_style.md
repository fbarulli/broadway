## broadway — Coding Standards

### Type Hints

Apply to all public functions, data pipelines, and key helpers. Skip only trivial one-liners or purely internal throwaway helpers.

```python
def run(cfg: StepConfig) -> None: ...
def load(path: Path) -> pd.DataFrame: ...
def summarize(df: pd.DataFrame, target: str) -> SummaryReport: ...
```

### Logging

Strategic only. Log pipeline stage boundaries, data quality check results, model training metrics, and errors. Do not log inside loops or in every helper function call.

```python
import logging
logger = logging.getLogger(__name__)

logger.info("EDA: summary stage started")
logger.info(f"  {col}: {n_nulls} nulls ({n_nulls/len(df):.1%})")
logger.exception("Failed to load model from registry")
```

### Exception Handling

Catch only where there is a recovery path or graceful degradation. Let everything else bubble up — the orchestrator logs the traceback and exits with a non-zero code.

```python
# RECOVERABLE — catch, log, continue
try:
    champion = mlflow.pyfunc.load_model(champion_uri)
except Exception as e:
    logger.warning(f"No champion found ({e}), promoting unconditionally")
    champion = None

# NOT recoverable — let it crash
model.fit(X_train, y_train)  # if this fails, the step should die loudly
```

### YAML Contracts

Every value in every `.yaml` file must be explicitly present. No defaults, no fallback setters, no `get(key, default)`. If the key is missing, Pydantic raises `ValidationError` — that is the right behavior.

```python
# ❌ NEVER
target = cfg.get("target", "price")
features = cfg.get("features", {"include": []})

# ✅ ALWAYS
# Schema.py enforces every field is required
# If the YAML is missing 'target', load_config() raises immediately
class DatasetContract(BaseModel):
    target: str          # required — no default
    task: Literal["regression", "classification"]  # required
```

YAML is the single source of truth. Pydantic is the enforcement layer. Nothing is silently filled in.

### Single Source of Truth

Every constant, threshold, default, path, and magic number lives in exactly one place — a config YAML, `schema.py`, or an environment variable. No hardcoded values anywhere else. If a value appears in two places, one of them is wrong by definition.

### Function Size

Single responsibility, ~25 lines max. If a function is approaching that, split it — extract a helper, break apart a pipeline stage, or decompose a conditional block. No mega-functions doing three things badly.
