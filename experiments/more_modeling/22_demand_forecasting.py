"""
Demand Forecasting: Predict ride volume by zone and hour.

This moves from pricing physics into volume prediction using temporal features.
No post-trip variables (distance, duration, fare) are used - only pickup zone
and time features.

Outputs:
- demand_volume_by_zone_hour.csv: Aggregated volume data
- demand_forecast_model_metrics.md: Model performance summary
- demand_forecast_actual_vs_predicted.png: Forecast visualization
- demand_forecast_error_by_zone.png: Error analysis by zone
"""

"""
22_demand_forecasting.py
Demand Forecasting: Predict ride volume by zone and hour.
"""
import os
import warnings

# Suppress warnings
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"
warnings.filterwarnings("ignore", message='.*Field name "schema".*shadows an attribute in parent "BaseModel".*')
warnings.filterwarnings("ignore", category=FutureWarning, module="pandera")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
from pathlib import Path

# Add parent directory to path for _common
sys.path.insert(0, str(Path(__file__).parent))
from _common import RESULTS, load_sample


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error, safely handling zeros."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero by adding a small epsilon or masking
    mask = y_true > 0
    if not mask.any():
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("22: DEMAND FORECASTING - Volume Prediction by Zone and Hour")
    print("=" * 70)

    # [1/7] Load data
    print("\n[1/7] Loading sample data...")
    df = load_sample()
    print(f"Loaded {len(df):,} trips")

    # [2/7] Extract temporal features
    print("\n[2/7] Extracting temporal features...")
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # [3/7] Aggregate to zone-hour-day volume
    print("\n[3/7] Aggregating to zone-hour-day volume...")
    volume_data = df.groupby(['pickup_location_id', 'hour', 'day_of_week', 'is_weekend']).size().reset_index(name='ride_count')

    print(f"Created {len(volume_data):,} zone-hour-day observations")
    print(f"Unique zones: {volume_data['pickup_location_id'].nunique()}")
    print(f"Ride count stats: Mean={volume_data['ride_count'].mean():.1f}, Median={volume_data['ride_count'].median():.1f}, Max={volume_data['ride_count'].max():,}")

    volume_data.to_csv(RESULTS / '22_demand_volume_by_zone_hour.csv', index=False)
    print(f"\nSaved: {RESULTS / '22_demand_volume_by_zone_hour.csv'}")

    # [4/7] Prepare features
    print("\n[4/7] Preparing features for modeling...")
    X = volume_data[['pickup_location_id', 'hour', 'day_of_week', 'is_weekend']]
    y = volume_data['ride_count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")

    # [5/7] Train model
    print("\n[5/7] Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # [6/7] Evaluate with enhanced metrics
    print("\n[6/7] Evaluating model performance...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Core metrics
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    test_mape = calculate_mape(y_test, y_pred_test)
    test_mbe = np.mean(y_pred_test - y_test)  # Positive = overprediction

    # Volume-tier metrics (crucial for skewed count data)
    low_demand_mask = y_test < 10
    high_demand_mask = y_test >= 10
    
    mae_low = mean_absolute_error(y_test[low_demand_mask], y_pred_test[low_demand_mask]) if low_demand_mask.any() else 0
    mae_high = mean_absolute_error(y_test[high_demand_mask], y_pred_test[high_demand_mask]) if high_demand_mask.any() else 0
    mape_high = calculate_mape(y_test[high_demand_mask], y_pred_test[high_demand_mask]) if high_demand_mask.any() else 0

    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)

    # Write enhanced metrics report
    with open(RESULTS / '22_demand_forecast_model_metrics.md', 'w') as f:
        f.write("# 22: Demand Forecasting Model Metrics\n\n")
        f.write("## Core Model Performance (Test Set)\n\n")
        f.write(f"- **MAE:** {test_mae:.2f} rides\n")
        f.write(f"- **RMSE:** {test_rmse:.2f} rides\n")
        f.write(f"- **R²:** {test_r2:.3f}\n")
        f.write(f"- **MAPE:** {test_mape:.1f}% *(Mean Absolute Percentage Error)*\n")
        f.write(f"- **Mean Bias Error (MBE):** {test_mbe:+.2f} rides *({'Overpredicting' if test_mbe > 0 else 'Underpredicting'} overall)*\n\n")
        
        f.write("## Volume-Tier Performance *(Crucial for skewed data)*\n\n")
        f.write(f"Since median volume is 5.0, overall MAE is dominated by low-volume noise.\n\n")
        f.write(f"- **Low Demand (< 10 rides/hr):** MAE = {mae_low:.2f} rides\n")
        f.write(f"- **High Demand (≥ 10 rides/hr):** MAE = {mae_high:.2f} rides, MAPE = {mape_high:.1f}%\n\n")
        f.write("*Operational takeaway: High-demand zones are where forecasting accuracy matters most for supply positioning.*\n\n")

        f.write("## Feature Importance\n\n")
        f.write("| Feature | Importance |\n|---------|------------|\n")
        for _, row in feature_importance.iterrows():
            f.write(f"| {row['feature']} | {row['importance']:.3f} |\n")
        f.write("\n## Interpretation\n\n")
        f.write("The model predicts ride volume using **only pre-trip information**, avoiding leakage:\n")
        f.write("- **pickup_location_id**: Pickup zone (most important)\n")
        f.write("- **hour**: Hour of day (0-23)\n")
        f.write("- **day_of_week**: Day of week (0=Monday, 6=Sunday)\n")
        f.write("- **is_weekend**: Weekend indicator\n")

    print(f"Saved: {RESULTS / '22_demand_forecast_model_metrics.md'}")

    # [7/7] Visualizations
    print("\n[7/7] Creating visualizations...")

    # Plot 1: Actual vs Predicted
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.scatter(y_test, y_pred_test, alpha=0.3, s=10)
    ax.plot([0, y_test.max()], [0, y_test.max()], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Actual Ride Count', fontsize=10)
    ax.set_ylabel('Predicted Ride Count', fontsize=10)
    ax.set_title(f'Actual vs Predicted (Test Set)\nMAE: {test_mae:.1f}, MAPE: {test_mape:.1f}%', fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(y_test, bins=50, alpha=0.5, label='Actual', density=True)
    ax.hist(y_pred_test, bins=50, alpha=0.5, label='Predicted', density=True)
    ax.set_xlabel('Ride Count', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Distribution: Actual vs Predicted', fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    errors = y_test - y_pred_test
    ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'Error Distribution\nMean Bias: {test_mbe:+.1f} rides', fontsize=11)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    feature_importance.plot(kind='bar', ax=ax, legend=False)
    ax.set_xlabel('Feature', fontsize=10)
    ax.set_ylabel('Importance', fontsize=10)
    ax.set_title('Feature Importance', fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(RESULTS / '22_demand_forecast_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {RESULTS / '22_demand_forecast_actual_vs_predicted.png'}")

    # Plot 2: Error analysis by zone
    test_with_predictions = X_test.copy()
    test_with_predictions['actual'] = y_test.values
    test_with_predictions['predicted'] = y_pred_test
    test_with_predictions['error'] = test_with_predictions['actual'] - test_with_predictions['predicted']
    test_with_predictions['abs_error'] = test_with_predictions['error'].abs()

    zone_errors = test_with_predictions.groupby('pickup_location_id').agg({
        'actual': 'mean', 'predicted': 'mean', 'abs_error': 'mean', 'error': ['mean', 'std']
    }).reset_index()
    zone_errors.columns = ['pickup_location_id', 'avg_actual', 'avg_predicted', 'avg_abs_error', 'mean_error', 'std_error']
    zone_errors = zone_errors.sort_values('avg_abs_error', ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    top_20 = zone_errors.head(20)
    ax.barh(range(len(top_20)), top_20['avg_abs_error'], color='coral')
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels([f"Zone {int(z)}" for z in top_20['pickup_location_id']], fontsize=8)
    ax.set_xlabel('Mean Absolute Error (rides)', fontsize=10)
    ax.set_title('Top 20 Zones by Prediction Error', fontsize=11)
    ax.grid(alpha=0.3, axis='x')
    ax.invert_yaxis()

    ax = axes[0, 1]
    overprediction = zone_errors[zone_errors['mean_error'] > 0].head(20)
    ax.barh(range(len(overprediction)), overprediction['mean_error'], color='lightcoral')
    ax.set_yticks(range(len(overprediction)))
    ax.set_yticklabels([f"Zone {int(z)}" for z in overprediction['pickup_location_id']], fontsize=8)
    ax.set_xlabel('Mean Error (rides)', fontsize=10)
    ax.set_title('Top 20 Zones: Systematic Overprediction', fontsize=10)
    ax.grid(alpha=0.3, axis='x')
    ax.invert_yaxis()

    ax = axes[1, 0]
    underprediction = zone_errors[zone_errors['mean_error'] < 0].tail(20)
    ax.barh(range(len(underprediction)), underprediction['mean_error'], color='lightblue')
    ax.set_yticks(range(len(underprediction)))
    ax.set_yticklabels([f"Zone {int(z)}" for z in underprediction['pickup_location_id']], fontsize=8)
    ax.set_xlabel('Mean Error (rides)', fontsize=10)
    ax.set_title('Top 20 Zones: Systematic Underprediction', fontsize=10)
    ax.grid(alpha=0.3, axis='x')
    ax.invert_yaxis()

    ax = axes[1, 1]
    ax.scatter(zone_errors['avg_actual'], zone_errors['avg_abs_error'], alpha=0.5, s=20)
    ax.set_xlabel('Average Actual Ride Count', fontsize=10)
    ax.set_ylabel('Mean Absolute Error', fontsize=10)
    ax.set_title('Error vs Volume by Zone', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / '22_demand_forecast_error_by_zone.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {RESULTS / '22_demand_forecast_error_by_zone.png'}")

    # Summary
    print("\n" + "=" * 70)
    print("22: DEMAND FORECASTING COMPLETE")
    print("=" * 70)
    print(f"\nTest Set Performance:")
    print(f"  MAE: {test_mae:.2f} rides | RMSE: {test_rmse:.2f} | R²: {test_r2:.3f}")
    print(f"  MAPE: {test_mape:.1f}% | Mean Bias: {test_mbe:+.2f} rides")
    print(f"\nVolume-Tier MAE:")
    print(f"  Low Demand (<10 rides): {mae_low:.2f} rides")
    print(f"  High Demand (≥10 rides): {mae_high:.2f} rides (MAPE: {mape_high:.1f}%)")
    print(f"\nOutputs:")
    print(f"  - {RESULTS / '22_demand_volume_by_zone_hour.csv'}")
    print(f"  - {RESULTS / '22_demand_forecast_model_metrics.md'}")
    print(f"  - {RESULTS / '22_demand_forecast_actual_vs_predicted.png'}")
    print(f"  - {RESULTS / '22_demand_forecast_error_by_zone.png'}")


if __name__ == "__main__":
    main()