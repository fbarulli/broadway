# Dataset Profile

## Answer

The dataset has 2 rows and 4 columns.

## Key evidence

### Variables

| Variable | Type | Missing | Unique | Min | Max |
| --- | --- | --- | --- | --- | --- |
| a | int64 | 0 | 2 | 1 | 4 |
| b | int64 | 0 | 2 | 2 | 5 |
| c | int64 | 0 | 2 | 3 | 6 |
| dt | object | 0 | 2 | 2024-01-01 | 2024-01-02 |

### Potentially important observations

- a has high cardinality (identifier_score=1.0) and behaves like an identifier; it may not be a meaningful grouping variable.
- b has high cardinality (identifier_score=1.0) and behaves like an identifier; it may not be a meaningful grouping variable.
- c has high cardinality (identifier_score=1.0) and behaves like an identifier; it may not be a meaningful grouping variable.
- dt has high cardinality (identifier_score=1.0) and behaves like an identifier; it may not be a meaningful grouping variable.

## Why this may matter

High-cardinality/identifier-like columns can inflate group counts or leak identity if used as grouping features.

## What Broadway did not decide

Broadway did not exclude or transform any column based on this profile.

## Technical details

artifacts/discover/profile.json
