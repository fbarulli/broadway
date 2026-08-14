# Lineage — taxi_hypothesis

This graph shows how the evidence and results for this analysis were produced.

Read arrows left-to-right:

parent --produces--> child

Special relationships:

- filters -> a slice restricts its parent dataset
- raises -> evidence raised an analytical decision
- referenced_not_found -> a node references this parent, but no persisted lineage record was found

```mermaid
flowchart LR
    analysis_taxi_hypothesis["taxi_hypothesis"]
    baseline_taxi_hypothesis["baseline:taxi_hypothesis"]
    dataset_taxi["taxi"]
    decision_cast_and_bound_passenger_count["cast_and_bound_passenger_count"]
    decision_distance_duration_is_flat_rate["distance_duration_is_flat_rate"]
    decision_drop_invalid_datetime["drop_invalid_datetime"]
    describe_taxi_hypothesis["describe:taxi_hypothesis (sample=taxi_diagnostic, role=diagnostic)"]
    etl_taxi["etl:taxi"]
    ingest_taxi["ingest:taxi"]
    join_taxi["join:taxi"]
    slice_airport["airport"]
    slice_distance_duration_inconsistent["distance_duration_inconsistent"]
    slice_passenger_out_of_range["passenger_out_of_range"]
    slice_pre_2024["pre_2024"]
    stats_taxi_hypothesis["stats:taxi_hypothesis"]
    analysis_taxi_hypothesis -->|produces| describe_taxi_hypothesis
    analysis_taxi_hypothesis -->|produces| stats_taxi_hypothesis
    baseline_taxi_hypothesis -->|produces| stats_taxi_hypothesis
    dataset_taxi -->|produces| ingest_taxi
    etl_taxi -->|produces| describe_taxi_hypothesis
    ingest_taxi -->|produces| join_taxi
    join_taxi -->|produces| etl_taxi
    slice_airport -->|filters| dataset_taxi
    slice_distance_duration_inconsistent -->|filters| dataset_taxi
    slice_distance_duration_inconsistent -->|raises| decision_distance_duration_is_flat_rate
    slice_passenger_out_of_range -->|filters| dataset_taxi
    slice_passenger_out_of_range -->|raises| decision_cast_and_bound_passenger_count
    slice_pre_2024 -->|filters| dataset_taxi
    slice_pre_2024 -->|raises| decision_drop_invalid_datetime
```
