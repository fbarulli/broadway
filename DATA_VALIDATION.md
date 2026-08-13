Do the reversible, structural work first:

fix column names/types
parse dates
standardize obvious missing-value encodings
remove exact duplicate rows only if they are truly accidental duplicates
validate required columns
record impossible parse failures
preserve the raw values alongside any normalized version when practical

Hold off on:

dropping outliers
imputing missing values
winsorizing/capping
scaling/normalizing
encoding categoricals
merging rare categories
removing “high-cardinality” columns
dropping correlated features
feature engineering
target transformations
balancing classes
filtering rows based on business assumptions
removing columns because they “look useless”
leakage-driven exclusions until you’ve defined the decision moment and available information
train/test split-dependent transformations before the split

For Broadway, I’d make the first usable dataset something like:

raw
 ↓
structural cleaning only
 ↓
validated canonical dataset
 ↓
DatasetProfile + EDA
 ↓
analysis decisions
 ↓
then semantic cleaning / feature engineering

The key rule is:

Cleaning before analysis should correct representation errors, not make analytical judgments.

If you remove 2% of rows because they look like outliers, for example, that should later be a traceable analytical decision with evidence—not an invisible ETL operation.

For your current dataset, I’d especially preserve nulls, extreme values, rare categories, and suspicious rows until profiling/EDA has described them. Those are often exactly the observations that tell you what cleaning policy you actually need.


Yes — I’d introduce the concept of **named dataset snapshots/slices** for analysis and feature-engineering decisions.

Not full copies of the whole dataset every time. More like:

```text
DatasetRef
   ↓
DatasetSlice
   ├─ definition
   ├─ row count
   ├─ digest
   ├─ sampling/filter rule
   └─ purpose
```

For example:

```yaml
name: feature_distance_outlier_review
source_dataset: taxi_v3
filter: "trip_distance > 50"
columns:
  - trip_distance
  - duration
  - fare_amount
purpose: validate outlier treatment
```

Then every decision/result references the slice:

```text
DecisionRecord
  decision: keep extreme distances
  evidence:
    dataset_slice: feature_distance_outlier_review
    result_artifact: artifacts/analysis/...
```

For inference you could similarly have:

```text
taxi_v3
 ├─ full
 ├─ weekday_only
 ├─ airport_trips
 ├─ high_distance
 └─ missing_passenger_count
```

and run the same test/diagnostic across those slices.

The important distinction:

**DatasetRef** = immutable source/version
**DatasetSlice** = reproducible query/filter/sample over that source
**Result** = statistical/feature result produced from the slice
**DecisionRecord** = what you decided based on those results

So traceability becomes:

```text
decision
   ↓ based on
result
   ↓ computed on
dataset slice
   ↓ derived from
dataset version
```

I think this is the missing abstraction for Broadway’s exploratory/decision-validation phase. It lets you experiment heavily **without creating anonymous `df2`, `cleaned_final_v3`, `sample_test.csv` datasets everywhere**.
