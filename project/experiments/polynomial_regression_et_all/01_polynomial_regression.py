"""01: polynomial regression exploration"""

from pathlib import Path

from _common import RESULTS

from project.working import load_metered

OUT = RESULTS / f"{Path(__file__).stem}.png"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_metered()
    print(f"rows: {len(df)}")


if __name__ == "__main__":
    main()
