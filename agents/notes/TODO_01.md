                  ┌──────────────────────────────────────────────┐
                  │          Regression Diagnostics Matrix       │
                  └──────────────────────┬───────────────────────┘
                                         │
       ┌─────────────────────────────────┼────────────────────────────────┐
       ▼                                 ▼                                ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐ ┌──────────────────────────────┐
│  1. Error Variance Check     │ │  2. Model Brittleness Check  │ │  3. Small Sample Inference   │
├──────────────────────────────┤ ├──────────────────────────────┤ ├──────────────────────────────┤
│ Tool: Residuals-vs-Fitted    │ │ Tool: Cook's Distance Plot   │ │ Tool: Normality of Residuals │
│ Big Data: Breusch-Pagan Test │ │ Big Data: Batch Calculation  │ │ Small Data: Q-Q Plot /       │
│    Non-linearity             │ │              with Thresholds │ │              Shapiro-Wilk    │
│ Goal: Ensure unbiased        │ │ Goal: Prevent individual     │ │ Goal: Ensure accurate        │
│       p-values/CIs           │ │       outliers from tilting  │ │       hypothesis testing     │
│Heteroscedasticity            │ │       the entire model line  │ │       when data is scarce    │
└──────────────────────────────┘ └──────────────────────────────┘ └──────────────────────────────┘
The Reality (The Central Limit Theorem): If you have a large dataset (e.g., thousands of rows), the Central Limit Theorem takes over. Your parameter estimates will be normally distributed even if the individual residuals are not. Therefore, non-normal residuals don't break your p-values in big data.The Exception (Small Samples): When your sample size is small, you cannot rely on the Central Limit Theorem. If your residuals are heavily skewed in a small dataset, your t-tests, F-tests, and confidence intervals become completely invalid.

Visually: Use a Q-Q Plot (Quantile-Quantile Plot). You want to see the residuals fall cleanly along a straight diagonal 45-degree line.Statistically: Run a Shapiro-Wilk or Kolmogorov-Smirnov test if you need a formal p-value for normality (though note that these tests can become overly sensitive in massive datasets).


Use a hexbin density plot or sample a randomized subset of 10,000 rows to visually inspect the actual severity of the funnel. If the spread only widens by a fraction of a percent at the high end, I would ignore it.

Train the model twice—once using standard OLS and once using Huber-White Robust Standard Errors. I would then compare the resulting confidence intervals for our key business features. If the features remain statistically significant in both models and the coefficients don't shift in a way that changes our business strategy, I would proceed with the simpler OLS model for production efficiency."