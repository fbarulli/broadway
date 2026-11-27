# 00_ingestion.md — data loading + column verification

How to run Broadway's data-loading steps and verify their outputs, especially
the columns. Column correctness is enforced at several boundaries, each with
its own command.

The project's dataset is the Amazon consumer-behaviour survey
(`project/config/dataset/amazon.yaml`); the platform's data-agnostic demo uses
the synthetic `test` dataset (`configs/dataset/test.yaml`).

### Human-readable views

For a human-readable summary of what happened to the data, run:

```bash
uv run ds-pipeline audit --dataset amazon
```

This renders `reports/audit/` (`index.md`, `profile.md`, `transform.md`) from
the persisted typed evidence — no recompute.

## Commands to load + verify columns

### 1. Inspect a source file (read-only, no writes)

```bash
uv run ds-pipeline columns --csv data/raw/amazon/amazon_consumer_behavior_survey.csv
```

Lists the columns + dtypes of a source file.

### 2. Canonicalize + split (`etl`)

```bash
uv run ds-pipeline etl --dataset amazon --experiment amazon
```

Preserves the raw CSV as the source of record, applies the contract's
`column_renames` (two trailing-space headers are distinct variables), then
validates against the config contract. Outputs under
`data/processed/amazon/` (the dataset-scoped `processed_subdir`):

- `amazon_canonical.parquet`
- `train.parquet`
- `val.parquet`
- `amazon_clean.json` (structural clean result)
- `etl_decision.json` / `etl.md` / `etl.png` (the ETL decision record)

Column presence/dtype is enforced by `build_raw_schema(dataset).validate(df)`
(`etl/module.py`).

### 3. Contract check (pass/fail on columns + nulls)

```bash
uv run ds-pipeline contracts --dataset amazon --experiment amazon
```

Checks `check_columns` + `check_nulls` against the dataset contract (null
threshold 0.05). Implemented in `contracts/module.py`.

### 4. Re-profile observed facts (per-column dtypes/nulls)

```bash
uv run ds-pipeline profile --dataset amazon
```

Writes `artifacts/discover/amazon/profile.json` (the dataset's artifact
namespace).

### 5. Infer a contract from a new file

This is the column-authoring path for a new dataset:

```bash
uv run ds-pipeline discover --csv <path> --target <col> --task regression
```

### 6. Tests that assert column correctness

```bash
uv run pytest -q
```

## Key column facts (what "correct" means)

- **Contract source of truth:** `project/config/dataset/amazon.yaml` — 23
  columns, each with `dtype`, `null_count`, and `role`; `Shopping_Satisfaction`
  is the target, `Timestamp` is ignored.
- **Raw-boundary check** (`contracts`) verifies **presence + nulls only**;
  dtype is intentionally deferred (`contracts/checks.py`).
- **Canonical-boundary check** (`etl`) verifies **strict dtypes** via Pandera
  (`contracts/pandera.py`).

ETL outputs live under `data/processed/amazon/` (git-ignored, regenerable).
