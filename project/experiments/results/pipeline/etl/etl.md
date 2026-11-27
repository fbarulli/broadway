# etl

## question

Did the ingest lose more rows than the contract explains?

## decision — no_action

- metric: rows_dropped_unexplained_share = 0.0000 [0.0000, 0.0000] 95% (n=602, unit=share)
- rule: max 0.1000
- action (amazon): fix the source before training

![figure](etl.png)

## evidence

- baseline: contract-clean ingest: every dropped row carries a recorded reason

### segments

| segment | value | n |
| --- | ---: | ---: |
| explained_drops | 0.0000 | 602 |
| unexplained_drops | 0.0000 | 602 |

### sensitivity

- total_drop_share_n602: 0.0000
- rows_out_n602: 602.0000

### falsifier

the unexplained-share interval clears the configured drop ceiling

### provenance

- artifact: canonical.parquet
- dataset: amazon
- rows_in: 602
- rows_out: 602
- reasons: none

## tables

- none
