# broadway

Generalized ML experimentation platform.

## Quick Start

```bash
uv sync
docker compose up -d
uv run ds-pipeline discover --csv data/raw/mydata.csv --target price --task regression
uv run ds-pipeline full --dataset mydata --experiment baseline
```

## Pipeline Steps

| Step | Produces |
|------|----------|
| discover | `configs/dataset/<name>.yaml` |
| etl | `data/processed/training_data.parquet` |
| contracts | Pass/fail validation |
| eda | `artifacts/reports/eda.html` |
| features | Fitted feature pipeline |
| stats | `artifacts/reports/statistical_analysis.html` |
| causal | Experiment design + analysis |
| train | Trained model in MLflow registry |
| evaluate | `artifacts/evaluation/` |
| full | All of the above |
