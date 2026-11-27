# baseline

## question

Does the fitted linear reference beat the no-skill bad-review prevalence on pr_auc?

## decision — act

- metric: improvement_over_naive_pr_auc = 0.5779 [0.4976, 0.6637] 95% (n=19,735, unit=relative_improvement)
- rule: max 0.2884 (permutation null 0.2884)
- action (olist): beat the no-skill bad-review prevalence on PR-AUC before a model ships

![figure](baseline.png)

## evidence

- baseline: naive training prevalence on the holdout rows (a constant score's PR-AUC equals the prevalence)
- delta vs baseline: +0.5779

### segments

| segment | value | n |
| --- | ---: | ---: |
| first_half | 0.6545 | 9,867 |
| second_half | 0.5152 | 9,868 |

### sensitivity

- level_90pct_low: 0.5119
- level_90pct_high: 0.6528
- prevalence_n19735: 0.1469
- linear_pr_auc_n19735: 0.2318

### falsifier

the paired improvement interval's lower bound falls at or below the permutation null / configured effect floor

### provenance

- dataset: olist
- target: binary 0/1 classification target
- artifact: baseline.json
- seed: 7
- rows: 19735
- prevalence: 0.146900
- model: linear
- null_iterations: 100
- resampling: paired row bootstrap on the holdout; label permutation null

## tables

- none
