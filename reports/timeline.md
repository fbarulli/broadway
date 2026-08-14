# Timeline — taxi_hypothesis

| # | Step | Question | Status |
| --- | --- | --- | --- |
| 1 | describe_groups | Do the groups contain enough observations? | ⚠ warning |
| 2 | normality | Is the distributional shape problematic? | ⚠ warning |
| 3 | variance | Is error/group variance homogeneous? | ⚠ warning |
| 4 | decide_omnibus | Which principal method should answer the question? | ✓ decided (method=welch) |
| 5 | omnibus | Do the group means differ? | ⚠ warning |
| 6 | decide_posthoc | Which post-hoc comparison is appropriate? | ✓ decided (method=games_howell) |
| 7 | posthoc | Which specific group pairs differ? | ✓ completed |
| 8 | conclusion | What is the conclusion? | ✓ completed |

## describe_groups

- ramification: group sizes are imbalanced (imbalance ratio 2162.6747).
- evidence_refs: describe.json
- result_summary:
  - total_n: 200130
  - imbalance_ratio: 2162.6747
  - absent_groups: 0

## normality

- ramification: distributional shape is skewed/heavy-tailed in some groups; consider this alongside sample size before choosing a method.
- evidence_refs: normality.json, figures/normality_Manhattan.png, figures/normality_Brooklyn.png, figures/normality_Queens.png, figures/normality_Bronx.png, figures/normality_Staten_Island.png
- result_summary:
  - Manhattan: {'skew': 2.6135481205438884, 'kurtosis': 12.989865616928158, 'shapiro_p': 2.740200164909112e-61}
  - Brooklyn: {'skew': 1.2038668790124056, 'kurtosis': 1.243844405097951, 'shapiro_p': 1.617933816183357e-29}
  - Queens: {'skew': 0.8409769804508647, 'kurtosis': 1.5226365999223006, 'shapiro_p': 7.503222976978351e-34}
  - Bronx: {'skew': 1.1359370476338995, 'kurtosis': 1.2215299438925866, 'shapiro_p': 2.3795031782397004e-13}
  - Staten Island: {'skew': 1.0160463931325074, 'kurtosis': 0.6058940541193816, 'shapiro_p': 6.932616221032695e-06}

## variance

- ramification: variance evidence favors considering Welch's ANOVA (or a rank-based alternative) over standard ANOVA.
- evidence_refs: variance.json
- result_summary:
  - statistic: 4199.7167447099055
  - p_value: 0.0

## omnibus

- ramification: reject H0: at least one group mean differs (p=0.0000e+00) Welch's ANOVA across 5 groups (N=199420); F(4, 507.78)=7001.649, p=0.0000e+00; reject H0: at least one group mean differs
- evidence_refs: omnibus.json
- result_summary:
  - method: welch
  - statistic: 7001.649490794274
  - p_value: 0.0
  - passed: True
  - eta_squared: 0.9822190100086721
  - omega_squared: 0.12313023513921474

## posthoc

- ramification: Games-Howell found 17 significant pairwise difference(s) at alpha=0.05.
- evidence_refs: posthoc.json
- result_summary:
  - method: games_howell
  - pairs: 21
  - significant_pairs: 17

## conclusion

- ramification: group means differ (welch p=0.0000e+00), with 17 significant pairwise difference(s).
- evidence_refs: conclusion.json
- result_summary:
  - verdict: group means differ (welch p=0.0000e+00), with 17 significant pairwise difference(s).
  - principal_method: welch
  - p_value: 0.0
  - effect_size: eta²=0.9822, omega²=0.1231
  - significant_pairs: 17