## broad-way/ — Complete Structure

```
broad-way/
│
├── configs/
│   ├── dataset/
│   │   # Auto-generated per CSV — not committed
│   ├── experiment/
│   │   ├── baseline.yaml                # Minimal features + linear model
│   │   ├── engineered.yaml              # Derived features + target encoding + LGBM
│   │   └── hyperopt.yaml                # Optuna sweep over XGB params
│   ├── environment/
│   │   ├── development.yaml             # Local: verbose logging, small samples, local paths
│   │   ├── staging.yaml                 # CI: sample size caps, remote MLflow URI
│   │   └── production.yaml              # Full data, production URIs, quiet logging
│   └── step/
│       ├── discover.yaml                # Step knobs: csv path, target prompt
│       ├── etl.yaml                     # Step knobs: CI sample size
│       ├── contracts.yaml               # Step knobs: null threshold, range tolerance
│       ├── eda.yaml                     # Step knobs: sample size, plot format, output dir
│       ├── features.yaml                # Step knobs: encoding smoothing, frequency fill
│       ├── stats.yaml                   # Step knobs: group column, group values, sample fraction
│       ├── causal.yaml                  # Step knobs: treatment col, outcome col, power, alpha
│       ├── train.yaml                   # Step knobs: random state, n_jobs, cv folds
│       ├── evaluate.yaml                # Step knobs: metric, promotion threshold
│       └── full.yaml                    # Steps list: [discover, etl, contracts, eda, ...]
│
├── src/
│   ├── cli.py                           # argparse entry point — resolves step → config → module.run()
│   ├── pipeline.py                      # Orchestrator: iterates cfg.steps, imports module, runs
│   ├── utils.py                         # deep_merge dicts, set up logging, timer decorator
│   │
│   ├── config/
│   │   ├── schema.py                    # Pydantic: DatasetContract, ExperimentConfig, StepConfig, EnvironmentConfig
│   │   ├── loader.py                    # YAML composition (dataset + env + step [+ experiment]) → Pydantic validation
│   │   └── resolver.py                  # Resolve ${paths}, interpolate env vars, expand globs
│   │
│   ├── discover/
│   │   └── module.py                    # Read CSV → infer dtypes/null counts/cardinality → write configs/dataset/<name>.yaml
│   │
│   ├── data/
│   │   ├── loader.py                    # Detect format (csv/parquet/excel) → load → optional lookup join
│   │   ├── cleaner.py                   # Filter invalid rows, derive target, drop duplicates, rename columns
│   │   ├── splitter.py                  # Time-based or stratified random train/test/val split
│   │   ├── download.py                  # Fetch raw files from URLs defined in dataset config
│   │   └── db.py                        # Postgres training view, COPY bulk loading
│   │
│   ├── etl/
│   │   └── module.py                    # Orchestrates data layer: download → load → clean → split → save parquet
│   │
│   ├── contracts/
│   │   ├── module.py                    # Validate DataFrame against DatasetContract (columns, dtypes, nulls, ranges)
│   │   └── checks.py                    # Null rate, cardinality, value range, duplicate row checks against thresholds
│   │
│   ├── eda/
│   │   ├── module.py                    # Orchestrates all EDA submodules → produces artifacts/reports/eda.html
│   │   ├── summary.py                   # Distributions, skew, kurtosis, cardinality, memory usage
│   │   ├── visualize.py                 # Histograms, boxplots, scatter matrix, correlation heatmap (Plotly)
│   │   ├── quality.py                   # Outliers (IQR/Z-score), class imbalance, constant columns, duplicate rows
│   │   ├── missing.py                   # Missingness analysis: counts, patterns, Little's MCAR test
│   │   ├── imputation.py                # Strategy-based imputation: mean, median, mode, KNN, MICE
│   │   ├── compare.py                   # Before/after ETL side-by-side comparison (distributions, counts)
│   │   └── report.py                    # Compose self-contained HTML report from all submodule outputs
│   │
│   ├── features/
│   │   ├── module.py                    # Learn stats from train (fit) → apply to train + test (transform) → save pipeline
│   │   ├── builders.py                  # Registry: @register_builder("datetime"), @register_builder("lookup_join"), etc.
│   │   ├── encodings.py                 # Smoothed target encoding, frequency encoding — pure functions
│   │   └── pipeline.py                  # FeaturePipeline dataclass: fit(train) learns, transform(df) applies
│   │
│   ├── stats/
│   │   ├── module.py                    # Orchestrates all stats submodules → produces artifacts/reports/statistical_analysis.html
│   │   ├── base.py                      # StatsContext: Spark session lifecycle, stratified sampling, load + join lookups
│   │   ├── anova.py                     # One-way ANOVA, Welch's ANOVA, Kruskal-Wallis on any group column
│   │   ├── assumptions.py               # Levene's test (equal variance), Shapiro-Wilk + skew/kurtosis (normality)
│   │   ├── post_hoc.py                  # Games-Howell pairwise, Tukey HSD — effect sizes (Hedges' g)
│   │   ├── regression.py                # OLS: baseline, log-target, interaction terms, HC3 robust SEs
│   │   ├── diagnostics.py               # Breusch-Pagan, Jarque-Bera, Q-Q plot, residual vs fitted plot
│   │   ├── time_series.py               # Durbin-Watson, ACF plot, ADF stationarity test
│   │   └── baseline.py                  # LGBM/XGB quick baseline with time-based split → benchmark metrics
│   │
│   ├── causal/
│   │   ├── module.py                    # Orchestrates causal submodules → design + analyze experiments
│   │   ├── design.py                    # Power analysis, minimum detectable effect, sample size calculation
│   │   ├── assignment.py                # Randomization, stratification, blocking schemes
│   │   ├── analysis.py                  # T-test, chi-square, difference-in-differences for experimental data
│   │   ├── multiple.py                  # Multiple testing correction: Bonferroni, Benjamini-Hochberg, FWER
│   │   ├── sequential.py                # Sequential monitoring, early stopping rules, alpha spending
│   │   └── hte.py                       # Heterogeneous treatment effects, CATE, uplift modeling
│   │
│   ├── training/
│   │   ├── module.py                    # Load data → build features → train model → log to MLflow → register
│   │   ├── trainer.py                   # Instantiate model from registry, fit, return model + training time
│   │   ├── mlflow_utils.py              # setup_mlflow, log_metrics, log_model, register_candidate, promote_candidate
│   │   ├── optuna.py                    # HPO: define objective, run study, return best params + model
│   │   └── models/
│   │       ├── base.py                  # ABC: fit, predict, feature_importance, get_params, set_params
│   │       ├── linear.py                # sklearn LinearRegression wrapper
│   │       ├── random_forest.py         # sklearn RandomForestRegressor/Classifier wrapper
│   │       ├── xgboost.py               # XGBRegressor/XGBClassifier wrapper
│   │       ├── lightgbm.py              # LGBMRegressor/LGBMClassifier wrapper
│   │       ├── registry.py              # get_model(name, **params) → model dict lookup
│   │       └── pyfunc_wrapper.py        # MLflow PyFunc: bundles FeaturePipeline + model for inference
│   │
│   ├── evaluate/
│   │   ├── module.py                    # Load trained model → evaluate on holdout → log metrics → check promotion
│   │   ├── metrics.py                   # Regression: MAE, RMSE, R², MAPE | Classification: precision, recall, F1, AUC
│   │   ├── comparison.py                # Candidate vs champion side-by-side metrics + residual plots
│   │   ├── validation.py                # Holdout scoring, cross-validation, calibration curve, residual analysis
│   │   └── promotion.py                 # should_promote(candidate, champion, metric) → bool + report
│   │
│   ├── trust/
│   │   ├── module.py                    # Orchestrates trust submodules (future: run post-train or on schedule)
│   │   ├── drift.py                     # PSI, KS test, KL divergence — train vs serve distribution comparison
│   │   ├── leakage.py                   # Target leakage detection: correlation with timestamps, ID-based checks
│   │   ├── sensitivity.py               # Perturbation analysis: how do predictions change under input noise?
│   │   ├── fairness.py                  # Subgroup disparity, equal opportunity, demographic parity
│   │   ├── interpretability.py          # SHAP summary/dependence plots, permutation importance
│   │   └── uncertainty.py               # Prediction intervals (bootstrap), conformal prediction
│   │
│   ├── monitoring/
│   │   ├── module.py                    # Orchestrates runtime monitoring (future: cron job or daemon)
│   │   ├── drift_alert.py               # Scheduled distribution drift checks, alert on threshold breach
│   │   ├── performance.py               # Prediction accuracy decay over time, rolling window metrics
│   │   ├── latency.py                   # Inference latency p50/p95/p99, throughput tracking
│   │   └── data_quality.py              # Production data quality vs training baseline regression
│   │
│   ├── selection/
│   │   ├── module.py                    # Orchestrates model selection submodules (future)
│   │   ├── nested_cv.py                 # Nested cross-validation: outer loop for evaluation, inner for HPO
│   │   ├── information.py               # AIC, BIC for model comparison (linear models)
│   │   └── learning_curves.py           # Learning curves, bias-variance decomposition
│   │
│   ├── unsupervised/
│   │   ├── module.py                    # Orchestrates unsupervised submodules (future)
│   │   ├── pca.py                       # PCA, explained variance ratio, scree plot, biplot
│   │   ├── clustering.py                # K-means, DBSCAN, HDBSCAN, silhouette scores
│   │   └── anomaly.py                   # Isolation forest, LOF, elliptic envelope outlier detection
│   │
│   └── inference/
│       └── api.py                       # FastAPI app: /health, /predict, /metrics — loads model from MLflow registry
│
├── artifacts/
│   ├── evaluation/
│   │   ├── metrics.json                 # Serialized metrics dict from evaluate step
│   │   ├── confusion_matrix.png         # Classification: confusion matrix heatmap
│   │   ├── calibration.png              # Calibration curve (reliability diagram)
│   │   └── comparison.html              # Candidate vs champion visual comparison report
│   └── reports/
│       ├── eda.html                     # Self-contained EDA report (embedded Plotly charts)
│       ├── statistical_analysis.html    # ANOVA, assumptions, diagnostics, baseline results
│       └── model_evaluation.html        # Holdout metrics, feature importance, residual analysis
│
├── scripts/
│   └── ds_pipeline.py                   # Thin wrapper: from broad_way.cli import main; main()
│
├── tests/
│   ├── test_config.py                   # Pydantic schema validation, loader round-trip
│   ├── test_discover.py                 # CSV → schema inference → YAML output verification
│   ├── test_data.py                     # Loader, cleaner, splitter with synthetic DataFrames
│   ├── test_contracts.py                # Schema enforcement: missing cols, extra cols, nulls, wrong dtypes
│   ├── test_eda.py                      # Summary stats, missingness, quality checks on known inputs
│   ├── test_features.py                 # Builders, encodings, pipeline fit/transform
│   ├── test_stats.py                    # ANOVA, assumptions, post-hoc, regression with synthetic data
│   ├── test_causal.py                   # Power analysis, randomization, HTE with synthetic data
│   ├── test_training.py                 # Model training, mlflow_utils, optuna with CI-safe small run
│   ├── test_evaluate.py                 # Metrics, comparison, validation, promotion logic
│   └── test_pipeline.py                 # Integration: full pipeline on 100-row synthetic dataset
│
├── notebooks/
│   └── exploratory.ipynb                # Quick-start notebook: load data → EDA → train baseline
│
├── Dockerfile                           # Production image: python:3.12-slim + uv sync + copy src
│
├── docker/
│   ├── mlflow/
│   │   └── Dockerfile                   # Minimal MLflow tracking server image
│   └── postgres/
│       └── init.sql                     # Schema: etl_file_manifest table
│
├── k8s/
│   ├── mlflow-deployment.yaml           # MLflow tracking server Deployment + Service
│   ├── postgres-deployment.yaml         # Postgres StatefulSet + Service + PVC
│   ├── train-job.yaml                   # Kubernetes Job: runs ds-pipeline train
│   ├── api-deployment.yaml              # FastAPI inference Deployment + Service + HPA
│   └── monitoring-cronjob.yaml          # CronJob: runs ds-pipeline monitoring.drift_alert
│
├── .github/
│   └── workflows/
│       ├── ci.yaml                      # On push: lint → test → discover --validate → etl (sample) → train (sample)
│       └── cd.yaml                      # On release: docker build → push → kubectl apply
│
├── data/
│   ├── raw/                             # Raw CSVs, parquets, lookup tables (gitignored)
│   └── processed/                       # ETL output parquets (gitignored)
│
├── docker-compose.yml                   # Local infra only: mlflow + postgres services
├── .env.example                         # MLFLOW_TRACKING_URI, DATABASE_URL, LOG_LEVEL
├── .gitignore                           # .env, data/, artifacts/, mlruns/, __pycache__, .venv/
├── pyproject.toml                       # Dependencies, [project.scripts], build-system
├── uv.lock                              # Deterministic dependency lockfile
└── README.md                            # Quick-start: install, discover, run pipeline, view reports
```
