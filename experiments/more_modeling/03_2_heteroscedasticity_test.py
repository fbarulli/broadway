import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms

from _common import RESULTS, TARGET, load_sample, INDEPENDENT_NUMERIC_FEATURES

CSV_OUT = RESULTS / "03_2_heteroscedasticity_test.csv"

def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_sample()
    
    X = df[INDEPENDENT_NUMERIC_FEATURES]
    y = df[TARGET]
    
    valid_idx = X.notna().all(axis=1) & y.notna()
    X_raw = sm.add_constant(X[valid_idx])
    y_clean = y[valid_idx]
    
    print("Fitting baseline OLS for residual extraction...")
    model = sm.OLS(y_clean, X_raw).fit()
    
    print("\n--- Breusch-Pagan Heteroscedasticity Test ---")
    bp_test = sms.het_breuschpagan(model.resid, model.model.exog)
    
    results = {
        "test": "Breusch-Pagan",
        "lagrange_multiplier_stat": bp_test[0],
        "p_value": bp_test[1],
        "f_statistic": bp_test[2],
        "f_p_value": bp_test[3],
    }
    
    print(f"Lagrange Multiplier Statistic: {results['lagrange_multiplier_stat']:,.2f}")
    print(f"p-value: {results['p_value']:.2e}")
    
    if results["p_value"] < 0.05:
        print("Verdict: REJECT Null Hypothesis. Heteroscedasticity IS present.")
        print("         (The variance in fare errors scales with the size of the fare).")
    else:
        print("Verdict: Fail to reject Null. Homoscedasticity (constant variance).")
        
    pd.DataFrame([results]).set_index("test").to_csv(CSV_OUT)
    print(f"wrote {CSV_OUT}")

if __name__ == "__main__":
    main()
