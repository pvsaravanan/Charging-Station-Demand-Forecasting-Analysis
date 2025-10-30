# XGBoost Charging Pile Demand Forecasting

## 🎯 Overview

This project develops an advanced machine learning pipeline for forecasting electric vehicle charging pile demand using historical usage patterns, traffic flow, weather conditions, and other environmental factors.

The model uses XGBoost regression with sophisticated feature engineering including temporal features, cyclical encoding, lagged features, and rolling statistics to predict charging pile usage rates with exceptional accuracy.

### Key Features
- High Accuracy: R² = 0.9875 (explains 98.75% of variance)
- Low Error Rate: MAPE = 2.69% (average prediction error)
- Advanced Feature Engineering: 38 engineered features
- Production-Ready: Complete preprocessing pipeline
- Comprehensive Visualizations: Correlation heatmaps, feature importance


## 🤖 Model Architecture

### Algorithm
XGBoost Regressor (Extreme Gradient Boosting)

### Hyperparameters
n_estimators: 200, max_depth: 7, learning_rate: 0.1
subsample: 0.8, colsample_bytree: 0.8
reg_alpha: 0.5, reg_lambda: 1.0, random_state: 42

### Training Configuration
- Train-Test Split: 80/20 (800 training, 200 test)
- Split Method: Time-based (preserves temporal ordering)
- Scaling: StandardScaler normalization


## 📈 Performance Metrics

### Test Set Results
| Metric | Value |
|--------|-------|
| R² Score | 0.9875 |
| RMSE | 0.0736 |
| MAE | 0.0587 |
| MAPE | 2.69% |

### Training Set Results
- R² Score: 0.9939
- RMSE: 0.0514
- MAE: 0.0400

### Quality Assessment
- Minimal overfitting (gap: 0.0064)
- Low prediction error across range
- Consistent performance
- Production-ready accuracy level

## 🔧 Feature Engineering

### Total Features: 38

### Top 10 Features by Importance
1. charging_duration (43.33%)
2. no_of_evs_charging (12.07%)
3. energy_consumed (10.35%)
4. usage_rolling_mean_3 (7.94%)
5. usage_rolling_std_3 (2.47%)
6. traffic_flow (2.37%)
7. hour_sin (2.03%)
8. charging_station_density (1.88%)
9. population_density (1.71%)
10. month (1.36%)

## 💾 Installation

### Requirements
- Python 3.7+
- pip (Python package manager)

### Step 1: Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

### Step 2: Verify Installation
```bash
python -c "import xgboost; print(xgboost.__version__)"
```

## 📚 Usage

### Training the Model
```bash
python train_xgboost_model.py
```

Executes:
1. Data loading and exploration
2. Preprocessing and cleaning
3. Feature engineering (38 features)
4. Train-test split (80/20 time-based)
5. Model training
6. Evaluation and metrics
7. Artifact persistence
8. Visualization generation

### Exploratory Data Analysis
```bash
python main.py
```

### Making Predictions
```python
import joblib
import json

model = joblib.load('xgboost_forecasting_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Test R² Score: {metadata['test_r2_score']:.4f}")

X_scaled = scaler.transform(X_engineered)
predictions = model.predict(X_scaled)
```

### Access Model Metadata
```python
import json

with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

print("Model Configuration:")
print(f"  Estimators: {metadata['n_estimators']}")
print(f"  Max Depth: {metadata['max_depth']}")

print("\nPerformance Metrics:")
print(f"  Test R²: {metadata['test_r2_score']:.4f}")
print(f"  Test RMSE: {metadata['test_rmse']:.4f}")
print(f"  Test MAE: {metadata['test_mae']:.4f}")
print(f"  Test MAPE: {metadata['test_mape']:.4f}")

print("\nTop 5 Features:")
for i, (feat, imp) in enumerate(metadata['top_features'][:5], 1):
    print(f"  {i}. {feat}: {imp:.2f}%")
```
## 📁 Project Structure

```
electricVehicle/
│
├── README.md                              # Documentation
├── requirements.txt                       # Python dependencies
│
├── DATA
│   └── charging_pile_demand_forecasting.csv
│
├── SCRIPTS
│   ├── main.py                           # Initial EDA
│   └── train_xgboost_model.py           # Training pipeline
│
├── MODEL ARTIFACTS
│   ├── xgboost_forecasting_model.pkl    # Trained model
│   ├── feature_scaler.pkl               # StandardScaler
│   ├── label_encoders.pkl               # Encoders
│   └── model_metadata.json              # Metadata & metrics
│
├── VISUALIZATIONS
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── model_evaluation.png
│   └── time_series_predictions.png
│
└── DOCUMENTATION
    ├── MODEL_TRAINING_REPORT.txt
    └── .zencoder/rules/repo.md
```

## 📊 Results & Visualizations

### 1. Correlation Heatmap
File: `correlation_heatmap.png`

Shows correlation of top 20 features with target variable.

Key Findings:
- Charging duration: highest correlation (0.886)
- No_of_evs_charging: 0.654
- Traffic flow: 0.456

### 2. Feature Importance Chart
File: `feature_importance.png`

XGBoost feature importance for top 20 features.

Top Contributors:
- charging_duration dominates at 43.33%
- no_of_evs_charging at 12.07%
- energy_consumed at 10.35%

### 3. Model Evaluation Dashboard
File: `model_evaluation.png`

4-panel visualization:
- Actual vs Predicted scatter plot
- Residuals vs Predicted values
- Residual distribution histogram
- Error distribution plot

Interpretation:
- Tight clustering around diagonal = excellent predictions
- Residuals centered at zero = unbiased model
- Symmetric error distribution = well-calibrated predictions

### 4. Time Series Predictions
File: `time_series_predictions.png`

Last 200 test samples showing model vs actual demand.

Demonstrates accurate tracking of demand patterns across different levels.

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.7+ |
| ML Framework | XGBoost 1.7+ |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Feature Engineering | Scikit-learn |
| Model Persistence | Joblib |
## 💡 Performance Summary

### Model Quality Indicators
- High Accuracy: R² = 0.9875 (exceeds 0.95 threshold)
- Low Error: MAPE = 2.69% (within acceptable range)
- Good Generalization: Train-Test R² gap only 0.0064
- Stable Predictions: Consistent across test set samples
- Feature Quality: Top 3 features explain 65.75% of variance

### Practical Applications
1. **Capacity Planning**: Predict peak demand periods for resource allocation
2. **Resource Optimization**: Schedule maintenance during low-demand times
3. **Pricing Strategy**: Dynamic pricing based on predicted demand
4. **Grid Management**: Balance EV charging with power supply capacity
5. **Investment Decisions**: Identify high-demand locations for new stations
6. **Service Planning**: Optimize staff scheduling and operations

## 🔄 Data Processing Pipeline

```
Raw Data (1000 records)
    |
    v
Missing Value Handling (none found)
    |
    v
Duplicate Removal (0 removed)
    |
    v
Outlier Detection (IQR method - 0 removed)
    |
    v
Feature Engineering (31 -> 38 features)
    |
    +-- Temporal features
    +-- Cyclical encoding
    +-- Lagged features (1, 3, 24-step)
    +-- Rolling statistics (3-period)
    +-- Categorical encoding
    |
    v
Train-Test Split (80/20 time-based)
    |
    v
Feature Scaling (StandardScaler)
    |
    v
Model Training (XGBoost - 200 estimators)
    |
    v
Evaluation & Metrics Calculation
    |
    v
Visualization Generation (4 charts)
    |
    v
Model Persistence (pkl files + metadata)
```



## 📄 License
Feel free to use, modify, and distribute it under the terms of the MIT License.
