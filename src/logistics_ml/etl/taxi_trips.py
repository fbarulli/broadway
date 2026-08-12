from pathlib import Path
import io
import time

import pyarrow.parquet as pq
from tqdm import tqdm

from logistics_ml.config.data import data
from logistics_ml.db import engine

RENAME = {
    "pulocationid": "pickup_location_id",
    "dolocationid": "dropoff_location_id",
    "tpep_pickup_datetime": "pickup_datetime",
    "tpep_dropoff_datetime": "dropoff_datetime",
}


def get_parquet_files():
    files = sorted(data.raw_data_dir.glob("yellow_tripdata_*.parquet"))
    print(f"Found {len(files)} files: {[f.name for f in files]}")
    return files


def ensure_manifest_table(conn):
    conn.exec_driver_sql(
        """
        CREATE TABLE IF NOT EXISTS etl_file_manifest (
            filename TEXT PRIMARY KEY,
            row_count INTEGER NOT NULL,
            loaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def get_loaded_files():
    with engine.begin() as conn:
        ensure_manifest_table(conn)
        return {
            row[0]
            for row in conn.exec_driver_sql(
                "SELECT filename FROM etl_file_manifest"
            )
        }


def mark_file_loaded(raw_conn, filename, row_count):
    with raw_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO etl_file_manifest (filename, row_count) VALUES (%s, %s)",
            (filename, row_count),
        )


def get_existing_columns():
    with engine.begin() as conn:
        return [
            row[0]
            for row in conn.exec_driver_sql(
                "SELECT column_name "
                "FROM information_schema.columns "
                "WHERE table_name = 'taxi_trips'"
            )
        ]


def load_parquet_batch(batch, existing_cols):
    part = batch.to_pandas()
    part.columns = [c.lower() for c in part.columns]
    part = part.rename(columns=RENAME)
    part = part[[c for c in part.columns if c in existing_cols]]
    return part


def copy_batch(raw_conn, part):
    buf = io.StringIO()
    part.to_csv(buf, index=False, header=False)
    buf.seek(0)

    with raw_conn.cursor() as cur:
        cols = ",".join(part.columns)
        with cur.copy(
            f"COPY taxi_trips ({cols}) FROM STDIN WITH (FORMAT CSV)"
        ) as copy:
            copy.write(buf.read())


def load_taxi_trips():
    print("Loading taxi trips...")

    files = get_parquet_files()
    existing_cols = get_existing_columns()
    already_loaded = get_loaded_files()

    total_rows = 0
    raw_conn = engine.raw_connection()

    try:
        for f in files:
            if f.name in already_loaded:
                print(f"Skipping {f.name} (already loaded)")
                continue

            print(f"Processing {f.name}...")
            start = time.time()

            parquet_file = pq.ParquetFile(f)
            file_rows = 0

            try:
                for batch in tqdm(
                    parquet_file.iter_batches(batch_size=data.batch_size),
                    desc=f.name,
                    unit="batch",
                ):
                    part = load_parquet_batch(batch, existing_cols)
                    copy_batch(raw_conn, part)
                    file_rows += len(part)

                mark_file_loaded(raw_conn, f.name, file_rows)
                raw_conn.commit()
                total_rows += file_rows
                print(
                    f"Finished {f.name}: "
                    f"{file_rows:,} rows "
                    f"in {time.time() - start:.1f}s"
                )
            except Exception:
                raw_conn.rollback()
                raise

    finally:
        raw_conn.close()

    print(f"Loaded {total_rows:,} rows")
