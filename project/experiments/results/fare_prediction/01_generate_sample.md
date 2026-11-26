# 01_generate_sample

## question

Did sampling plus the fare policy preserve the source population?

## decision — act

- metric: ks_fare_amount = 0.0120 [0.0111, 0.0149] 95% (n=50,000)
- rule: max 0.0100 (permutation null 0.0077)
- action (fare_prediction): review the fare-policy exclusion before training

![figure](01_generate_sample.png)

## evidence

- baseline: no-effect permutation (sample labels shuffled at this n)

### segments

| segment | value | n |
| --- | ---: | ---: |
| weekday | 0.0121 | 50,000 |
| weekend | 0.0152 | 50,000 |

### sensitivity

- ks_half_sample: 0.0116
- ks_seed_7: 0.0121

### falsifier

the KS interval no longer clears the null band once the fare-policy exclusions are relaxed

### provenance

- artifact: fare_prediction_1m
- artifact_sha256: 158940b0d905b77d53f94cb849fffdff068958db642a7bbd19074a742a8aed67
- source: data/processed/training_data.parquet
- seed: 42
- exclusions: fare<=4.40 with long distance/duration/speed

## tables

- [columns](01_generate_sample_columns.csv)
- [funnel](01_generate_sample_funnel.csv)
