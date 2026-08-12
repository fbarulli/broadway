# src/logistics_ml/data.py
import pandas as pd
from pathlib import Path

PROCESSED_FILE = Path("data/processed/training_data.parquet")


def load_training_data() -> pd.DataFrame:
    """
    Load the offline training dataset from the processed Parquet file.
    """
    print(f"Loading training data from {PROCESSED_FILE}...")

    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_FILE}. "
            "Run `python scripts/prepare_data.py` first."
        )

    df = pd.read_parquet(PROCESSED_FILE)
    print(f"Loaded {len(df):,} rows")
    return df
