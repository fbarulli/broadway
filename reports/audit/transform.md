# Structural Transform

## Answer

Structural canonicalization removed 168 of 8545833 rows; 8545665 rows remain. Rows were removed only for deterministic structural reasons. No domain or outlier cleaning was performed here.

## Key evidence

### Row transitions

- rows_in: 8545833
- rows_out: 8545665
- rows_dropped_total: 168
- rows_dropped_unexplained: 0

### Reasons

- duplicates: -168 rows

### Columns added

none

### Columns removed

none

### Parse failures

none

## Why this may matter

Nothing material to flag.

## What Broadway did not decide

Broadway recorded these changes but did not make any analytical judgment about them.

## Technical details

data/processed/taxi_clean.json
