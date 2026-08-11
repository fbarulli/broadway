# dataflow

## Config composition (loader.py)

```
configs/
  environment/{env}.yaml  ─┐
  dataset/{name}.yaml      ─┤  deep_merge  →  PipelineConfig (Pydantic)
  experiment/{exp}.yaml    ─┤  (env < dataset < experiment < step)
  step/{step}.yaml         ─┘
```

## Pipeline steps (full.yaml order)

| # | Step | Module | In | Out |
|---|------|--------|----|-----|
| 1 | discover | `discover/` | raw CSV | `configs/dataset/<name>.yaml` |
| 2 | etl | `etl/` → `data/` | raw CSV/parquet | `data/processed/training_data.parquet` |
| 3 | contracts | `contracts/` | processed parquet | pass/fail (schema, nulls, ranges) |
| 4 | eda | `eda/` | processed parquet | `artifacts/reports/eda.html` |
| 5 | features | `features/` | train split | fitted `FeaturePipeline` |
| 6 | stats | `stats/` | processed parquet | `artifacts/reports/statistical_analysis.html` |
| 7 | train | `training/` | train split + features | model in MLflow registry |
| 8 | evaluate | `evaluate/` | model + holdout split | `artifacts/evaluation/`, promotion decision |

## Step contract

Every step module exposes:
```python
def run(cfg: PipelineConfig) -> None:
    ...
```
