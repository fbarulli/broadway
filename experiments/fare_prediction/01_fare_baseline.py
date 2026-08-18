"""01: fare prediction baseline"""

from pathlib import Path

import pandas as pd

from _common import RESULTS
from project.working import load_metered

OUT = RESULTS / f"{Path(__file__).stem}.png"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_metered()
    print(f"rows: {len(df)}")


if __name__ == "__main__":
    main()
