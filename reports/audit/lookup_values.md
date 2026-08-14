# Lookup Value Audit

## Answer

Some matched enrichment values were missing or sentinel.

## Key evidence

### pickup_location_id

| column | nulls | sentinels | affected_rows | affected_rate | affected_keys |
| --- | --- | --- | --- | --- | --- |
| Borough | 1093 | Unknown: 27053 | 28146 | 0.003294 | 264, 265 |
| Zone | 27053 | - | 27053 | 0.003166 | 264 |
| service_zone | 28146 | - | 28146 | 0.003294 | 264, 265 |

### dropoff_location_id

| column | nulls | sentinels | affected_rows | affected_rate | affected_keys |
| --- | --- | --- | --- | --- | --- |
| Borough_lookup | 31176 | Unknown: 37588 | 68764 | 0.008046 | 264, 265 |
| Zone_lookup | 37588 | - | 37588 | 0.004398 | 264 |
| service_zone_lookup | 68764 | - | 68764 | 0.008046 | 264, 265 |

## Why this may matter

- Borough has deficient values; analyses that group or filter by this column may use fewer or differently classified observations depending on the analytical population definition.
- Zone has deficient values; analyses that group or filter by this column may use fewer or differently classified observations depending on the analytical population definition.
- service_zone has deficient values; analyses that group or filter by this column may use fewer or differently classified observations depending on the analytical population definition.
- Borough_lookup has deficient values; analyses that group or filter by this column may use fewer or differently classified observations depending on the analytical population definition.
- Zone_lookup has deficient values; analyses that group or filter by this column may use fewer or differently classified observations depending on the analytical population definition.
- service_zone_lookup has deficient values; analyses that group or filter by this column may use fewer or differently classified observations depending on the analytical population definition.

## What Broadway did not decide

Broadway did not exclude these rows or redefine the population.

## Technical details

data/processed/taxi_lookup_value_audit.json
