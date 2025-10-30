import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import joblib

print("=" * 80)
print("XGBOOST CHARGING PILE DEMAND FORECASTING MODEL")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING & INITIAL EXPLORATION
# ============================================================================
print("\n[STEP 1] Loading Data...")
df = pd.read_csv("c:\\proj\\internship\\electricVehicle\\charging_pile_demand_forecasting.csv")

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe()}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 2] Data Preprocessing...")
print("=" * 80)

# Handle missing values
print(f"\nHandling missing values...")
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"  - Filled {col} with {'median' if df[col].dtype in ['float64', 'int64'] else 'mode'}")

# Remove duplicates
initial_rows = len(df)
df.drop_duplicates(inplace=True)
print(f"\nRemoved {initial_rows - len(df)} duplicate rows")

# Handle outliers using IQR method for numeric columns
print(f"\nHandling outliers using IQR method...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
outliers_removed = 0
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    if outliers > 0:
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        outliers_removed += outliers
        print(f"  - {col}: removed {outliers} outliers")

print(f"Total outliers removed: {outliers_removed}")
print(f"Dataset shape after preprocessing: {df.shape}")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 3] Feature Engineering...")
print("=" * 80)

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Temporal features
print("\nCreating temporal features...")
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["day_of_month"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["quarter"] = df["timestamp"].dt.quarter
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_month_start"] = (df["day_of_month"] <= 3).astype(int)
df["is_month_end"] = (df["day_of_month"] >= 28).astype(int)

# Cyclical encoding for hour and month (sine-cosine transformation)
print("Creating cyclical features...")
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Peak hours identification
print("Identifying peak hours...")
df["is_peak_hour"] = ((df["hour"] >= 7) & (df["hour"] <= 9)) | ((df["hour"] >= 17) & (df["hour"] <= 19))
df["is_peak_hour"] = df["is_peak_hour"].astype(int)

# Lagged features
print("Creating lagged features...")
df = df.sort_values("timestamp").reset_index(drop=True)
df["usage_lag_1"] = df.groupby("charging_station_id")["charging_pile_usage_rate"].shift(1)
df["usage_lag_3"] = df.groupby("charging_station_id")["charging_pile_usage_rate"].shift(3)
df["usage_lag_24"] = df.groupby("charging_station_id")["charging_pile_usage_rate"].shift(24)

# Rolling statistics
print("Creating rolling statistics...")
df["usage_rolling_mean_3"] = df.groupby("charging_station_id")["charging_pile_usage_rate"].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)
df["usage_rolling_std_3"] = df.groupby("charging_station_id")["charging_pile_usage_rate"].transform(
    lambda x: x.rolling(window=3, min_periods=1).std()
)

# Fill NaN values from lagged/rolling features
df.fillna(df.mean(numeric_only=True), inplace=True)

print(f"Feature engineering complete. New columns added: {len(df.columns)}")

# ============================================================================
# 4. CATEGORICAL ENCODING
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 4] Categorical Encoding...")
print("=" * 80)

categorical_cols = df.select_dtypes(include=["object"]).columns
label_encoders = {}

for col in categorical_cols:
    if col not in ["timestamp"]:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"  - Encoded {col}: {len(le.classes_)} unique values")

# Drop original categorical columns and timestamp
df.drop(columns=list(categorical_cols) + ["timestamp"], inplace=True)

print(f"\nFinal dataset shape: {df.shape}")

# ============================================================================
# 5. FEATURE SELECTION & CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 5] Feature Selection & Correlation Analysis...")
print("=" * 80)

# Calculate correlations
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

# Top correlated features with target
target_correlations = correlation_matrix["charging_pile_usage_rate"].sort_values(ascending=False)
print(f"\nTop 15 features correlated with target:")
print(target_correlations.head(15))

# Visualize correlation heatmap
plt.figure(figsize=(14, 10))
top_features = target_correlations.head(20).index
sns.heatmap(df[top_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={"label": "Correlation"})
plt.title("Correlation Matrix - Top 20 Features")
plt.tight_layout()
plt.savefig("c:\\proj\\internship\\electricVehicle\\correlation_heatmap.png", dpi=300, bbox_inches="tight")
print("\n✓ Saved: correlation_heatmap.png")
plt.close()

# ============================================================================
# 6. TRAIN-TEST SPLIT & SCALING
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 6] Train-Test Split & Scaling...")
print("=" * 80)

# Select features (exclude target)
X = df.drop(columns=["charging_pile_usage_rate"])
y = df["charging_pile_usage_rate"]

print(f"Feature set shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Time-based split (important for time series)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTrain set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

# Scale features
print(f"\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create DataFrames with feature names for XGBoost
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print("✓ Features scaled and ready for training")

# ============================================================================
# 7. XGBOOST MODEL TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 7] XGBoost Model Training...")
print("=" * 80)

# Initialize XGBoost model with optimized hyperparameters
print("\nInitializing XGBoost model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    tree_method="hist",
    verbose=0
)

# Train the model
print("\nTraining XGBoost model...")
xgb_model.fit(X_train_scaled, y_train, verbose=False)

print("✓ Model training complete")

# ============================================================================
# 8. MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 8] Model Evaluation...")
print("=" * 80)

# Make predictions
y_pred_train = xgb_model.predict(X_train_scaled)
y_pred_test = xgb_model.predict(X_test_scaled)

# Calculate metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mape = mean_absolute_percentage_error(y_test, y_pred_test)

print("\n📊 TRAINING SET METRICS:")
print(f"  MAE:  {train_mae:.4f}")
print(f"  MSE:  {train_mse:.4f}")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  R²:   {train_r2:.4f}")

print("\n📊 TEST SET METRICS:")
print(f"  MAE:  {test_mae:.4f}")
print(f"  MSE:  {test_mse:.4f}")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  R²:   {test_r2:.4f}")
print(f"  MAPE: {test_mape:.4f}")

# ============================================================================
# 9. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 9] Feature Importance Analysis...")
print("=" * 80)

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_20_features = feature_importance.head(20)
sns.barplot(data=top_20_features, x="importance", y="feature", palette="viridis")
plt.title("XGBoost Feature Importance (Top 20)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("c:\\proj\\internship\\electricVehicle\\feature_importance.png", dpi=300, bbox_inches="tight")
print("\n✓ Saved: feature_importance.png")
plt.close()

# ============================================================================
# 10. VISUALIZATION - PREDICTIONS vs ACTUAL
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 10] Visualization - Predictions vs Actual...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Scatter plot: Actual vs Predicted
axes[0, 0].scatter(y_test, y_pred_test, alpha=0.5, s=20)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
axes[0, 0].set_xlabel("Actual Usage Rate")
axes[0, 0].set_ylabel("Predicted Usage Rate")
axes[0, 0].set_title(f"Actual vs Predicted (Test Set)\nR² = {test_r2:.4f}")
axes[0, 0].grid(True, alpha=0.3)

# Residuals plot
residuals = y_test - y_pred_test
axes[0, 1].scatter(y_pred_test, residuals, alpha=0.5, s=20)
axes[0, 1].axhline(y=0, color="r", linestyle="--", lw=2)
axes[0, 1].set_xlabel("Predicted Usage Rate")
axes[0, 1].set_ylabel("Residuals")
axes[0, 1].set_title("Residual Plot")
axes[0, 1].grid(True, alpha=0.3)

# Histogram of residuals
axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor="black")
axes[1, 0].set_xlabel("Residuals")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].set_title("Distribution of Residuals")
axes[1, 0].axvline(x=0, color="r", linestyle="--", lw=2)
axes[1, 0].grid(True, alpha=0.3)

# Error distribution
errors = np.abs(y_test - y_pred_test)
axes[1, 1].hist(errors, bins=50, alpha=0.7, color="green", edgecolor="black")
axes[1, 1].set_xlabel("Absolute Error")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_title("Distribution of Absolute Errors")
axes[1, 1].axvline(x=np.mean(errors), color="r", linestyle="--", lw=2, label=f"Mean: {np.mean(errors):.4f}")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("c:\\proj\\internship\\electricVehicle\\model_evaluation.png", dpi=300, bbox_inches="tight")
print("✓ Saved: model_evaluation.png")
plt.close()

# ============================================================================
# 11. TIME SERIES VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 11] Time Series Visualization...")
print("=" * 80)

# Create time series plot for last 500 predictions
sample_size = min(500, len(y_test))
test_indices = np.arange(len(y_test) - sample_size, len(y_test))

plt.figure(figsize=(15, 6))
plt.plot(test_indices, y_test.values[len(y_test) - sample_size:], label="Actual", linewidth=2, alpha=0.7)
plt.plot(test_indices, y_pred_test[len(y_test) - sample_size:], label="Predicted", linewidth=2, alpha=0.7)
plt.xlabel("Time Index")
plt.ylabel("Charging Pile Usage Rate")
plt.title(f"Time Series: Actual vs Predicted (Last {sample_size} observations)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("c:\\proj\\internship\\electricVehicle\\time_series_predictions.png", dpi=300, bbox_inches="tight")
print("✓ Saved: time_series_predictions.png")
plt.close()

# ============================================================================
# 12. SAVE MODEL AND ARTIFACTS
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 12] Saving Model & Artifacts...")
print("=" * 80)

# Save XGBoost model
model_path = "c:\\proj\\internship\\electricVehicle\\xgboost_forecasting_model.pkl"
joblib.dump(xgb_model, model_path)
print(f"✓ Saved XGBoost model: xgboost_forecasting_model.pkl")

# Save scaler
scaler_path = "c:\\proj\\internship\\electricVehicle\\feature_scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"✓ Saved feature scaler: feature_scaler.pkl")

# Save label encoders
encoders_path = "c:\\proj\\internship\\electricVehicle\\label_encoders.pkl"
joblib.dump(label_encoders, encoders_path)
print(f"✓ Saved label encoders: label_encoders.pkl")

# Save model metadata
metadata = {
    "model_type": "XGBRegressor",
    "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "n_features": X.shape[1],
    "features": X.columns.tolist(),
    "test_metrics": {
        "MAE": float(test_mae),
        "MSE": float(test_mse),
        "RMSE": float(test_rmse),
        "R2": float(test_r2),
        "MAPE": float(test_mape)
    },
    "train_metrics": {
        "MAE": float(train_mae),
        "MSE": float(train_mse),
        "RMSE": float(train_rmse),
        "R2": float(train_r2)
    }
}

import json
metadata_path = "c:\\proj\\internship\\electricVehicle\\model_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Saved model metadata: model_metadata.json")

# ============================================================================
# 13. SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY REPORT")
print("=" * 80)
print(f"""
PROJECT: Charging Pile Demand Forecasting with XGBoost

DATA:
  - Total records: {len(df)}
  - Training samples: {len(X_train)}
  - Test samples: {len(X_test)}
  - Number of features: {X.shape[1]}

MODEL PERFORMANCE:
  Test Set R² Score:    {test_r2:.4f}
  Test Set RMSE:        {test_rmse:.4f}
  Test Set MAE:         {test_mae:.4f}
  Test Set MAPE:        {test_mape:.4f}

TOP 5 IMPORTANT FEATURES:
""")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']:.<40} {row['importance']:.4f}")

print(f"""
ARTIFACTS SAVED:
  ✓ xgboost_forecasting_model.pkl
  ✓ feature_scaler.pkl
  ✓ label_encoders.pkl
  ✓ model_metadata.json
  ✓ correlation_heatmap.png
  ✓ feature_importance.png
  ✓ model_evaluation.png
  ✓ time_series_predictions.png

Training completed successfully!
""")
print("=" * 80)
