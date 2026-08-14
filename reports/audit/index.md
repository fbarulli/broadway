# Data Audit — taxi — supporting taxi_hypothesis

## Data used

- rows_in: 8545833
- rows_out (canonical): 8545832
- rows_dropped_total: 1

Dataset status: READY WITH WARNINGS
All required audit artifacts are present, but one or more deterministic evidence checks reported non-zero deficiencies.

## What changed

- exact-duplicate rows dropped: 1

## Enrichment quality

- Join completeness: PASS — all lookup keys matched (17091666/17091666)
- Lookup value quality: WARNING — 258461 matched rows had missing or sentinel enrichment values

## Things to consider before inference

- Borough has deficient values; analyses that group or filter by this column may use fewer or differently classified observations depending on the analytical population definition.
- Zone has deficient values; analyses that group or filter by this column may use fewer or differently classified observations depending on the analytical population definition.
- service_zone has deficient values; analyses that group or filter by this column may use fewer or differently classified observations depending on the analytical population definition.
- Borough_lookup has deficient values; analyses that group or filter by this column may use fewer or differently classified observations depending on the analytical population definition.
- Zone_lookup has deficient values; analyses that group or filter by this column may use fewer or differently classified observations depending on the analytical population definition.
- service_zone_lookup has deficient values; analyses that group or filter by this column may use fewer or differently classified observations depending on the analytical population definition.
- No decision has been made automatically.
- If your analysis compares NYC boroughs, define the analytical population explicitly.

## Details

- [profile](profile.md)
- [transform](transform.md)
- [join](join.md)
- [lookup_values](lookup_values.md)
- [lineage graph](../lineage/graph.md)
