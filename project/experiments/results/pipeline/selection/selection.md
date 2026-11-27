# selection

## question

Does the selected rmse champion beat the naive reference on identical repeated folds?

## decision — act

- metric: improvement_over_naive_rmse = 0.1834 [0.1318, 0.2371] 95% (n=481, unit=relative_improvement)
- rule: max 0.0200
- action (amazon): carry the selected champion into HPO and a holdout promotion check

![figure](selection.png)

## evidence

- baseline: naive training-mean predictor 'naive_mean' on the identical 5x5 shuffled folds
- delta vs baseline: +0.1834

### segments

| segment | value | n |
| --- | ---: | ---: |
| low_target_half | 0.1681 | 253 |
| high_target_half | 0.1968 | 228 |

### sensitivity

- champion_vs_linear_rmse: 0.0093
- champion_vs_linear_rmse_low: -0.0210
- champion_vs_linear_rmse_high: 0.0387
- second_seed_improvement_seed77: 0.1925
- champion_cv_rmse_n481: 0.8355
- naive_cv_rmse_n481: 1.0134

### falsifier

the paired improvement interval's lower bound falls at or below the configured effect floor

### provenance

- dataset: amazon
- target: ordinal 1-5 treated as interval-scale
- seed: 2026
- folds: 5x5 shuffled KFold
- selection_rule: point
- rule_fired: point (leaderboard top)
- one_se_would_pick: linear
- candidates: naive_mean, linear, rf, lgbm
- champion_params: {'n_estimators': 300, 'min_samples_leaf': 2}
- resampling: paired row bootstrap on out-of-fold predictions

## tables

- [candidates](selection_candidates.csv)
