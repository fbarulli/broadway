"""
Step 2: Join taxi_zone_lookup.csv to get pickup borough, then see
group sizes and mean trip duration per borough.
Run with: python projects/taxi/scripts/02_join_boroughs.py
"""
from projects.taxi import data


def main() -> None:
    data.write_quality_report()
    print(f"\nWrote quality report to {data._quality_report_path()}")


if __name__ == "__main__":
    main()
