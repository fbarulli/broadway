"""
Step 1: Load training_data.parquet with Spark and get a feel for it.
Run with: python projects/taxi/scripts/01_load_data.py
"""
from projects.taxi import data


def main() -> None:
    data.inspect_schema()


if __name__ == "__main__":
    main()
