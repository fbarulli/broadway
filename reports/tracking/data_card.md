# Data Card — taxi

Generated at: unrecorded

## Date range

2024-01-01 00:00:00 .. 2024-03-14 15:52:35

## Row counts

- training frame: 6836532
- profiled source: unknown

## Biases

- not declared

## Exclusions

- rows_in=8545833 rows_out=8545665 dropped_total=168 dropped_unexplained=0
  - reason: duplicates: -168 rows

## Missingness

- dropoff_location_id: 0/6836532 (0.0000)
- fare_amount: 0/6836532 (0.0000)
- passenger_count: 0/6836532 (0.0000)
- pickup_datetime: 0/6836532 (0.0000)
- pickup_location_id: 0/6836532 (0.0000)
- trip_distance: 0/6836532 (0.0000)
- trip_duration_minutes: 0/6836532 (0.0000)
- LocationID: 0/6836532 (0.0000)
- Borough: 861/6836532 (0.0001)
- Zone: 21626/6836532 (0.0032)
- service_zone: 22487/6836532 (0.0033)
- LocationID_lookup: 0/6836532 (0.0000)
- Borough_lookup: 24645/6836532 (0.0036)
- Zone_lookup: 29936/6836532 (0.0044)
- service_zone_lookup: 54581/6836532 (0.0080)
- pickup_hour: 0/6836532 (0.0000)
- pickup_day_of_week: 0/6836532 (0.0000)
- pickup_month: 0/6836532 (0.0000)
- is_weekend: 0/6836532 (0.0000)
- rush_hour: 0/6836532 (0.0000)
- is_night: 0/6836532 (0.0000)
- same_borough: 0/6836532 (0.0000)
- pickup_location_id_target_enc: 0/6836532 (0.0000)

## Training baseline

dataset=taxi rows=6836532 n_bins=10 top_k=10
- dropoff_location_id (numeric, n=6836532): mean=165.2190092871649 std=69.1617188056448 missing=0/6836532 bins=10
- fare_amount (numeric, n=6836532): mean=17.988658449927534 std=17.188179028478793 missing=0/6836532 bins=10
- passenger_count (numeric, n=6836532): mean=1.345698667101975 std=0.8304852447332659 missing=0/6836532 bins=10
- pickup_datetime (datetime, n=6836532): missing=0/6836532
- pickup_location_id (numeric, n=6836532): mean=166.19106353923306 std=63.25050666436511 missing=0/6836532 bins=10
- trip_distance (numeric, n=6836532): mean=3.2650749049371814 std=4.312656022694626 missing=0/6836532 bins=10
- trip_duration_minutes (numeric, n=6836532): mean=15.317586194774389 std=12.304528090197163 missing=0/6836532 bins=10
- LocationID (numeric, n=6836532): mean=166.19106353923306 std=63.25050666436511 missing=0/6836532 bins=10
- Borough (categorical, n=6835671): missing=861/6836532 top_bins=7
- Zone (categorical, n=6814906): missing=21626/6836532 top_bins=10
- service_zone (categorical, n=6814045): missing=22487/6836532 top_bins=4
- LocationID_lookup (numeric, n=6836532): mean=165.2190092871649 std=69.1617188056448 missing=0/6836532 bins=10
- Borough_lookup (categorical, n=6811887): missing=24645/6836532 top_bins=7
- Zone_lookup (categorical, n=6806596): missing=29936/6836532 top_bins=10
- service_zone_lookup (categorical, n=6781951): missing=54581/6836532 top_bins=4
- pickup_hour (numeric, n=6836532): mean=14.287487720382206 std=5.668129740758553 missing=0/6836532 bins=10
- pickup_day_of_week (numeric, n=6836532): mean=2.9649041356055967 std=1.9182256743332886 missing=0/6836532 bins=10
- pickup_month (numeric, n=6836532): mean=1.79663475575043 std=0.7468564379794522 missing=0/6836532 bins=10
- is_weekend (numeric, n=6836532): mean=0.2653209258729426 std=0.44150397583627893 missing=0/6836532 bins=10
- rush_hour (numeric, n=6836532): mean=0.3150949926073629 std=0.4645537318850515 missing=0/6836532 bins=10
- is_night (numeric, n=6836532): mean=0.2145135867132634 std=0.41048450942186976 missing=0/6836532 bins=10
- same_borough (numeric, n=6836532): mean=0.8706235851744715 std=0.3356161119319873 missing=0/6836532 bins=10
- pickup_location_id_target_enc (numeric, n=6836532): mean=15.307326014063275 std=6.728844174998813 missing=0/6836532 bins=10
