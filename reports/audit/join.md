# Join Audit

## Answer

All lookup keys matched successfully (17091666/17091666). This describes key matching only. It does NOT mean the matched values were usable — see the Lookup Value Audit.

## Key evidence

### Joins

| lookup | rows_attempted | matched | unmatched | null_keys | unmatched_rate |
| --- | --- | --- | --- | --- | --- |
| pickup_location_id | 8545833 | 8545833 | 0 | 0 | 0.0 |
| dropoff_location_id | 8545833 | 8545833 | 0 | 0 | 0.0 |

## Why this may matter

Nothing material to flag.

## What Broadway did not decide

Broadway did not drop or impute unmatched rows.

## Technical details

data/processed/taxi_join_audit.json
