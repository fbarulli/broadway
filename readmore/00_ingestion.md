# 00_ingestion.md — data loading + column verification

How to run Broadway's data-loading steps and verify their outputs, especially
the columns. Column correctness is enforced at several boundaries, each with
its own command.

The project's dataset is the Olist Brazilian e-commerce order table
(`project/config/dataset/olist.yaml`, grain `order_id`); the platform's
data-agnostic demo uses the synthetic `test` dataset
(`configs/dataset/test.yaml`).

### Human-readable views

For a human-readable summary of what happened to the data, run:

```bash
uv run ds-pipeline audit --dataset olist
```

This renders `reports/audit/` (`index.md`, `profile.md`, `transform.md`) from
the persisted typed evidence — no recompute.

## Commands to load + verify columns

### 0. Build the order-grain table from the nine raw CSVs

```bash
uv run python -m project.data
```

`project/data.py` (the single owner of the pre-join) reads the join/aggregation
spec `project/config/project/olist.yaml`, writes one row per `order_id` to
`data/raw/olist/olist_orders.parquet` (the contract's `path`), and writes the
funnel report `data/raw/olist/olist_build_report.json`. No row is dropped there:
the 768 orders without a review keep a null target and the ETL removes them with
an accounted reason.

### 1. Inspect a source file (read-only, no writes)

```bash
uv run ds-pipeline columns --csv data/raw/olist/olist_orders_dataset.csv
```

Lists the columns + dtypes of a source file.

### 2. Canonicalize + split (`etl`)

```bash
uv run ds-pipeline etl --dataset olist --experiment olist
```

Reads the built order-grain parquet, applies the contract's read-time renames
(none for Olist), validates against the config contract, drops the null target
with a recorded reason, then writes under `data/processed/olist/` (the
dataset-scoped `processed_subdir`):

- `olist_canonical.parquet` (98,673 rows: 14,495 bad / 84,178 good)
- `train.parquet`
- `val.parquet`
- `olist_clean.json` (structural clean result)
- `etl_decision.json` / `etl.md` / `etl.png` (the ETL decision record)

Column presence/dtype is enforced by `build_raw_schema(dataset).validate(df)`
(`etl/module.py`).

### 3. Contract check (pass/fail on columns + nulls)

```bash
uv run ds-pipeline contracts --dataset olist --experiment olist
```

Checks `check_columns` + `check_nulls` against the dataset contract (null
threshold 0.05). Implemented in `contracts/module.py`.

### 4. Re-profile observed facts (per-column dtypes/nulls)

```bash
uv run ds-pipeline profile --dataset olist
```

Writes `artifacts/discover/olist/profile.json` (the dataset's artifact
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

- **Contract source of truth:** `project/config/dataset/olist.yaml` — 32
  columns, each with `dtype`, `null_count`, and `role`; `bad_review` is the
  target (0/1 from `review_score <= 2`), the ids/timestamps/post-purchase
  context columns are `ignore`.
- **Raw-boundary check** (`contracts`) verifies **presence + nulls only**;
  dtype is intentionally deferred (`contracts/checks.py`).
- **Canonical-boundary check** (`etl`) verifies **strict dtypes** via Pandera
  (`contracts/pandera.py`).

ETL outputs live under `data/processed/olist/` (git-ignored, regenerable).
