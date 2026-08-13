Model-based explanatory inference — OLS/GLM/GAM, coefficients, diagnostics.  
Estimation-first interval inference — effect sizes, confidence intervals, practical significance.  
Predictive inference — cross-validation, out-of-sample performance, prediction intervals.  
Causal inference — identification, matching, weighting, DiD, IV, RDD, sensitivity.  
Robust inference — robust SEs, quantile regression, transformations, outlier-aware methods.  
Resampling inference — bootstrap, permutation, jackknife, randomization tests.  
Bayesian inference — posterior distributions, priors, probabilistic statements.  
Multilevel inference — random effects, clustering, partial pooling.  
Spatial inference — spatial autocorrelation, spillovers, spatial regression.  
Temporal/panel inference — fixed effects, trends, seasonality, event studies.  
Model-selection inference — AIC/BIC, cross-validated loss, model averaging.  
Nonparametric/flexible inference — GAMs, trees, ML models, marginal effects.  
Heterogeneity inference — interactions, subgroup effects, varying slopes.  
Multiple-testing inference — FDR, familywise control, hierarchical testing.  
Missing-data inference — multiple imputation, weighting, MNAR sensitivity.  
Survey/design-based inference — weights, stratification, clustering, finite-population targets.  
Mediation/mechanism inference pathways, SEM, causal mediation.  
Equivalence/decision inference — TOST, ROPE, loss functions, policy thresholds.  
Sensitivity/multiverse inference — robustness across reasonable analytic choices.  
Exploratory/diagnostic inference — EDA, clustering, dimension reduction, anomaly detection.














1. Model-based explanatory inference

ypical methods

    OLS
    GLMs: logistic regression, Poisson/negative binomial, Gamma, beta regression
    GAMs for nonlinear relationships
    Robust regression
    Regularized regression: ridge, lasso, elastic net
    Mixed-effects models if data are nested

Typical outputs

    Coefficients
    Confidence intervals
    Partial effects / marginal effects
    Model fit diagnostics
    Residual diagnostics
    Influence analysis
    Standardized or scale-aware effect sizes where useful





2. Group-comparison / ANOVA-style inference

    One-way ANOVA
    Welch ANOVA
    Kruskal–Wallis test
    Permutation ANOVA
    Bootstrap confidence intervals for group differences
    Pairwise contrasts with multiplicity adjustment
    Effect sizes: eta-squared, omega-squared, Cohen’s f, mean differences with CIs

3. Estimation-first / interval inference

    point estimates
    confidence/compatibility intervals
    practical significance
    uncertainty bands
    effect sizes on meaningful scales

Typical methods

    OLS coefficients with confidence intervals
    Robust standard errors
    Bootstrap confidence intervals
    Bayesian credible intervals
    Estimated marginal means / contrasts
    Prediction intervals where relevant



4. Predictive inference

Typical methods

    Train/test split
    Cross-validation
    Nested cross-validation if tuning is involved
    Regularized regression
    Random forests, gradient boosting, or other ML models
    Calibration analysis
    Prediction intervals
    Conformal prediction
    Baseline model comparison

Typical outputs

    RMSE, MAE, MAPE, R²
    Log loss or classification metrics if outcome is categorical
    Calibration curves
    Out-of-sample residual plots
    Prediction interval coverage
    Feature importance, if interpretable





5. Causal inference

Directed acyclic graphs / causal diagrams

Backdoor adjustment

Matching

Propensity score weighting

Inverse probability weighting

Doubly robust estimation

Difference-in-differences

Event studies

Instrumental variables

Regression discontinuity

Synthetic control

Double machine learning

Sensitivity analysis for unmeasured confounding





6. Robust / assumption-relaxed inference



Heteroskedasticity-consistent standard errors: HC3, HC4

Cluster-robust standard errors

Bootstrap inference

Permutation inference

Quantile regression

Robust regression: Huber, Tukey, MM-estimation

Transformations: log, square root, Box-Cox

GLMs for skewed positive outcomes

Winsorization or trimming, with caution and transparency



7. Resampling and randomization-based inference



    Nonparametric bootstrap
    Parametric bootstrap
    Permutation tests
    Randomization inference
    Jackknife
    Cross-validation-based uncertainty
    Bootstrap model stability analysis

When useful

    Small samples where asymptotic approximations are weak
    Complex estimands
    Nonstandard statistics
    Group comparisons with unequal variances
    Inference after transformations or robust estimators
    Assessing stability of variable selection





8. Bayesian inference



Bayesian linear regression

Bayesian hierarchical models

Informative or weakly informative priors

Posterior predictive checks

Credible intervals

Probabilities of directional effects

Probabilities that effects exceed decision thresholds





9. Multilevel / hierarchical inference



Random intercept models

Random slope models

Mixed-effects models

Cluster-robust standard errors

Intraclass correlation coefficient

Partial pooling

Cross-classified or nested hierarchical models

11. Temporal / panel inference





Fixed effects

Random effects, carefully

Difference-in-differences

Event studies

Time-series models

Panel-corrected standard errors

Autocorrelation diagnostics

Rolling or expanding window validation

Seasonal decomposition





12. Model-selection and information-theoretic inference





AIC

BIC

Likelihood ratio tests for nested models

Cross-validated loss

Out-of-sample deviance

Stacking

Bayesian model averaging

Model weights



13. Flexible / nonparametric inference



    Generalized additive models
    Splines
    Local regression
    Kernel regression
    Regression trees
    Random forests
    Gradient boosting
    Partial dependence plots
    Accumulated local effects
    SHAP values for explanation





14. Heterogeneity / subgroup inference



nteraction terms

Stratified models

Varying-slope mixed models

Subgroup analysis

Heterogeneous treatment effect estimation

Causal forests, if causal and predictive

Quantile regression

Marginal effects by subgroup



15. Multiple-testing and multivariate inference



Bonferroni correction

Holm correction

Benjamini–Hochberg false discovery rate control

Hierarchical testing

Gatekeeping procedures

MANOVA

Global tests followed by controlled follow-up

Shrinkage estimators

Empirical Bayes methods







16. Missing-data and measurement-error inference



Complete-case analysis, with explicit assumptions

Missingness mechanism analysis: MCAR, MAR, MNAR

Multiple imputation

Inverse probability weighting for missingness

Sensitivity analysis for MNAR

Measurement-error models

Latent variable models

Validation-substudy approaches





17. Survey / design-based inference



Sampling weights

Stratification

Clustering

Poststratification

Raking

Design-based variance estimation

Survey-weighted regression

Finite-population inference





18. Mediation, mechanism, and pathway analysis



Mediation analysis

Causal mediation analysis

Structural equation modeling

Path analysis

Instrumental approaches where necessary

Sensitivity analysis for mediator confounding



19. Equivalence, noninferiority, and decision-theoretic inference



TOST: two one-sided tests

ROPE: region of practical equivalence

Minimum effect size of interest

Decision thresholds

Loss functions

Expected utility analysis

Policy simulation under uncertainty



Specification curve analysis

Multiverse analysis

Leave-one-out analysis

Influence analysis

Alternative transformations

Alternative outlier policies

Alternative covariate sets

Alternative variance estimators

Alternative missing-data treatments

Bayesian prior sensitivity

Bootstrap stability analysis





20. Sensitivity, robustness, and multiverse inference