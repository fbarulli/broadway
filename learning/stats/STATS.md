| # | Script | What it does |
|---|---|---|---|
| 00 | prepare_data.py | Generate deterministic cached sample (results/joined_sample.parquet) used by scripts 08-12 |
| 01 | load_data.py | Load training_data.parquet with Spark, inspect schema |
| 02 | join_boroughs.py | Join taxi_zone_lookup.csv for pickup borough, show group sizes and mean durations |
| 04 | anova_boroughs.py | One-way ANOVA testing whether trip duration differs across boroughs |
| 05 | anova_assumptions.py | Check ANOVA assumptions: Levene's test, skew/kurtosis, Shapiro-Wilk |
| 06 | anova_comparison.py | Compare four ANOVA variants: standard, log-transformed, Welch's, Kruskal-Wallis |
| 07 | games_howell.py | Games-Howell post-hoc test to identify which specific borough pairs differ |
| 08 | ols_residuals_diagnostics.py | Baseline OLS residual diagnostics: plots, Breusch-Pagan, Jarque-Bera, Durbin-Watson |
| 09 | log_target_ols.py | Log-transformed target OLS and HC3 robust standard errors to address diagnostics from 08 |
| 10 | durbin_watson_time.py | Durbin-Watson on time-ordered data with ACF plot of residuals |
| 11 | interaction_ols.py | Interaction OLS with trip_distance * pickup_borough, nested F-test vs baseline |
| 12 | lgbm_baseline.py | LightGBM baseline with time-based train/test split, feature importance |

Shared helpers, constants, and the sample cache live in `_config.py`. Run `00_prepare_data.py` once before scripts 08-12.
