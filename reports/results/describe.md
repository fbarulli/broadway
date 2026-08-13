# describe

## Question

whether trip duration differs across pickup boroughs

## Sample

- sample_name: taxi_diagnostic
- sample_role: diagnostic
- source_path: results/joined_sample_live.parquet
- source_group_column: pickup_borough
- group_column: Borough

## Groups

| group | n | mean | std |
| --- | --- | --- | --- |
| Manhattan | 179502 | 13.3611 | 9.6256 |
| Brooklyn | 1361 | 34.9321 | 26.6854 |
| Queens | 18113 | 34.5446 | 17.1056 |
| Bronx | 361 | 42.4716 | 29.2130 |
| Staten Island | 83 | 24.2540 | 20.1963 |

## Imbalance

imbalance_ratio: 2162.6747

absent_groups: none

## Warnings

none

## Figures

[describe_boxplot.png](figures/describe_boxplot.png)
[describe_group_sizes.png](figures/describe_group_sizes.png)
