# 00_ingestion.md — data loading + column verification

How to run Broadway's data-loading steps and verify their outputs, especially
the columns. Two surfaces produce data: the pipeline CLI (`ingest` → `etl`)
and the numbered project scripts. Column correctness is enforced at several
boundaries, each with its own command.

## Commands to load + verify columns

### 1. Inspect the produced raw data schema (read-only, no recompute)

```bash
uv run python -m project.scripts.01_load_data
```

Prints the arrow schema, row count, sample rows, and column names of
`data/processed/training_data.parquet`. Implemented in `inspect_schema`
(`project/data.py`).

### 2. Rebuild the raw data (`ingest`)

```bash
uv run ds-pipeline ingest
```

Reads `data/raw/yellow_tripdata_*.parquet`, filters/renames, then enforces the
exact schema. Produces `data/processed/training_data.parquet` and writes the
`ingest:taxi` lineage record.

It hard-fails on any column mismatch via `validate_raw_schema`
(`features/contracts.py`): exactly `RAW_FEATURES + [TARGET]`, no nulls, strict
dtypes.

Expected 6 columns (`features/schema.py`):

- `pickup_datetime`
- `passenger_count`
- `trip_distance`
- `pickup_location_id`
- `dropoff_location_id`
- `trip_duration_minutes` (target)

### 3. Canonicalize + split (`etl`)

```bash
uv run ds-pipeline etl --dataset taxi --experiment taxi
```

Joins lookup tables, canonicalizes, then validates against the config contract.
Outputs:

- `taxi_canonical.parquet`
- `train.parquet`
- `val.parquet`
- `taxi_clean.json` (structural clean result)
- `taxi_join_audit.json` (join audit)
- `taxi_lookup_value_audit.json` (lookup value audit)

Column presence/dtype is enforced by `build_raw_schema(dataset).validate(df)`
(`etl/module.py`).

### 4. Contract check (pass/fail on columns + nulls)

```bash
uv run ds-pipeline contracts --dataset taxi --experiment taxi
```

Checks `check_columns` + `check_nulls` against `configs/dataset/taxi.yaml`
(null threshold 0.05). Implemented in `contracts/module.py`.

### 5. Re-profile observed facts (per-column dtypes/nulls)

```bash
uv run ds-pipeline profile --dataset taxi
```

Writes `artifacts/discover/profile.json`.

### 6. Infer a contract from a new file

Not needed for taxi, but this is the column-authoring path for a new dataset:

```bash
uv run ds-pipeline discover --csv <path> --target <col> --task regression
```

### 7. Tests that assert column correctness

```bash
uv run pytest -q
```

`tests/test_contracts.py` validates columns and nulls against
`data/processed/training_data.parquet`.

## Key column facts (what "correct" means)

- **Contract source of truth:** `configs/dataset/taxi.yaml` — 6 columns, each
  with `dtype`, `null_count`, and `role`.
- **Raw-boundary check** (`contracts`) verifies **presence + nulls only**;
  dtype is intentionally deferred (`contracts/checks.py`).
- **Canonical-boundary check** (`etl`) verifies **strict dtypes** via Pandera
  (`contracts/pandera.py`).

All steps already have outputs on disk under `data/processed/`
(e.g. `training_data.parquet`, `taxi_canonical.parquet`).
