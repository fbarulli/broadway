"""01: fare prediction baseline"""

from _common import CLEAN_PARQUET, RESULTS, SAMPLE_SIZE, SEED
from project.data import read_training_sample


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = read_training_sample(
        sample=SAMPLE_SIZE, seed=SEED, columns=["fare_amount", "trip_distance"]
    )
    df.to_parquet(CLEAN_PARQUET)
    print(f"rows: {len(df)}")
    print(f"wrote {CLEAN_PARQUET}")


if __name__ == "__main__":
    main()
