# hpo

## question

Do the HPO-best params beat the declared defaults on training-only repeated CV?

## decision — investigate

- metric: improvement_over_default_rmse = 0.0060 [-0.0019, 0.0137] 95% (n=481, unit=relative_improvement)
- rule: max 0.0100
- action (amazon): adopt the HPO-best params only when the interval clears the floor

![figure](hpo.png)

## evidence

- baseline: the experiment's default params on the identical report folds
- delta vs baseline: +0.0060

### segments

| segment | value | n |
| --- | ---: | ---: |
| low_target_half | 0.0017 | 253 |
| high_target_half | 0.0101 | 228 |

### sensitivity

- level_90pct_low: -0.0007
- level_90pct_high: 0.0130
- best_cv_rmse_n481: 0.8206
- default_cv_rmse_n481: 0.8285

### falsifier

the paired best-vs-default interval's lower bound falls at or below the configured effect floor

### provenance

- dataset: amazon
- target: ordinal 1-5 treated as interval-scale
- search_seed: 7
- report_seed: 99
- folds: 3x5 shuffled KFold on the report seed
- trials: 30
- model: rf
- best_params: {'n_estimators': 570, 'max_depth': 14, 'min_samples_leaf': 2, 'min_samples_split': 5, 'max_samples': 0.5416264327961499}
- resampling: paired row bootstrap on out-of-fold predictions

## tables

- [folds](hpo_folds.csv)
