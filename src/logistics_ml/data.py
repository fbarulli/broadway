from pathlib import Path

import pandas as pd

from logistics_ml.config import data as data_config


def load_training_data() -> pd.DataFrame:
    processed_file = data_config.processed_dir / data_config.processed_file

    print(f"Loading training data from {processed_file}...")

    if not processed_file.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_file}. "
            "Run `python scripts/prepare_data.py` first."
        )

    df = pd.read_parquet(processed_file)
    print(f"Loaded {len(df):,} rows")
    return df
