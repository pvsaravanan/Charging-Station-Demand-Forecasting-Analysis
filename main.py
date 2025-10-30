import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('charging_pile_demand_forecasting.csv')

# Basic data exploration
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe())

# Convert timestamp to datetime and extract time-based features
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['day_of_month'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Analyze the target variable - charging_pile_usage_rate
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.hist(df['charging_pile_usage_rate'], bins=50, alpha=0.7, color='skyblue')
plt.title('Distribution of Charging Pile Usage Rate')
plt.xlabel('Usage Rate')
plt.ylabel('Frequency')

plt.subplot(2, 3, 2)
plt.scatter(df['hour'], df['charging_pile_usage_rate'], alpha=0.5)
plt.title('Usage Rate by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Usage Rate')

plt.subplot(2, 3, 3)
plt.scatter(df['traffic_flow'], df['charging_pile_usage_rate'], alpha=0.5)
plt.title('Usage Rate vs Traffic Flow')
plt.xlabel('Traffic Flow')
plt.ylabel('Usage Rate')

plt.subplot(2, 3, 4)
df.groupby('vehicle_type')['charging_pile_usage_rate'].mean().plot(kind='bar')
plt.title('Average Usage Rate by Vehicle Type')
plt.xticks(rotation=45)

plt.subplot(2, 3, 5)
plt.scatter(df['temperature'], df['charging_pile_usage_rate'], alpha=0.5)
plt.title('Usage Rate vs Temperature')
plt.xlabel('Temperature')
plt.ylabel('Usage Rate')

plt.subplot(2, 3, 6)
df.groupby('public_holiday_or_event')['charging_pile_usage_rate'].mean().plot(kind='bar')
plt.title('Usage Rate: Holiday vs Normal Day')
plt.xticks([0, 1], ['Normal', 'Holiday/Event'])

plt.tight_layout()
plt.show()

# Correlation analysis
numeric_columns = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix[['charging_pile_usage_rate']].sort_values('charging_pile_usage_rate', ascending=False), 
            annot=True, cmap='coolwarm', center=0)
plt.title('Correlation with Charging Pile Usage Rate')
plt.show()

# Top correlations with target variable
target_correlations = correlation_matrix['charging_pile_usage_rate'].sort_values(ascending=False)
print("Top correlations with charging_pile_usage_rate:")
print(target_correlations.head(10))

# Time series analysis of usage patterns
daily_usage = df.groupby(df['timestamp'].dt.date)['charging_pile_usage_rate'].mean()

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
daily_usage.plot()
plt.title('Daily Average Usage Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Average Usage Rate')

plt.subplot(2, 2, 2)
hourly_usage = df.groupby('hour')['charging_pile_usage_rate'].mean()
hourly_usage.plot(kind='line', marker='o')
plt.title('Average Usage Rate by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Usage Rate')

plt.subplot(2, 2, 3)
weekly_usage = df.groupby('day_of_week')['charging_pile_usage_rate'].mean()
weekly_usage.plot(kind='bar')
plt.title('Average Usage Rate by Day of Week')
plt.xlabel('Day of Week (0=Monday)')
plt.ylabel('Average Usage Rate')

plt.subplot(2, 2, 4)
monthly_usage = df.groupby('month')['charging_pile_usage_rate'].mean()
monthly_usage.plot(kind='bar')
plt.title('Average Usage Rate by Month')
plt.xlabel('Month')
plt.ylabel('Average Usage Rate')

plt.tight_layout()
plt.show()

# Analyze station-specific patterns
station_analysis = df.groupby('charging_station_id').agg({
    'charging_pile_usage_rate': ['mean', 'std', 'count'],
    'traffic_flow': 'mean',
    'population_density': 'mean',
    'charging_station_density': 'mean'
}).round(3)

print("Station-wise Analysis:")
print(station_analysis)


# Prepare data for machine learning
# Select relevant features based on correlation analysis and domain knowledge
feature_columns = [
    'traffic_flow', 'peak_traffic_hours', 'avg_vehicle_speed', 'temperature',
    'humidity', 'precipitation', 'wind_speed', 'population_density',
    'charging_station_density', 'public_holiday_or_event', 
    'charging_price_per_kWh', 'government_incentives',
    'hour', 'day_of_week', 'is_weekend', 'month',
    'no_of_evs_charging', 'charging_duration', 'energy_consumed'
]

# Handle categorical variables
le = LabelEncoder()
df['vehicle_type_encoded'] = le.fit_transform(df['vehicle_type'])

# Add the encoded vehicle type to features
feature_columns.append('vehicle_type_encoded')

# Create feature set and target
X = df[feature_columns]
y = df['charging_pile_usage_rate']

print(f"Feature set shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression()
}

# Train and evaluate models
results = {}

for name, model in models.items():
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Predictions': y_pred
    }
    
    print(f"\n{name} Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

# Feature importance from Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

print("Top 10 Most Important Features:")
print(feature_importance.head(10))

# Visualization of predictions vs actual
best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
best_predictions = results[best_model_name]['Predictions']

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, best_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Usage Rate')
plt.ylabel('Predicted Usage Rate')
plt.title(f'Actual vs Predicted - {best_model_name}\nR² = {results[best_model_name]["R2"]:.3f}')

plt.subplot(1, 2, 2)
residuals = y_test - best_predictions
plt.scatter(best_predictions, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Usage Rate')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.show()

# Time-based prediction analysis
test_dates = df['timestamp'].iloc[X_test.index]
test_with_predictions = pd.DataFrame({
    'timestamp': test_dates,
    'actual': y_test.values,
    'predicted': best_predictions
}).sort_values('timestamp')

plt.figure(figsize=(15, 6))
plt.plot(test_with_predictions['timestamp'], test_with_predictions['actual'], label='Actual', alpha=0.7)
plt.plot(test_with_predictions['timestamp'], test_with_predictions['predicted'], label='Predicted', alpha=0.7)
plt.title('Time Series: Actual vs Predicted Usage Rates')
plt.xlabel('Time')
plt.ylabel('Usage Rate')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Advanced analysis: Peak demand forecasting
# Identify peak usage hours (top 20%)
peak_threshold = df['charging_pile_usage_rate'].quantile(0.8)
df['is_peak_demand'] = (df['charging_pile_usage_rate'] >= peak_threshold).astype(int)

print(f"Peak demand threshold: {peak_threshold:.2f}")
print(f"Percentage of peak demand observations: {df['is_peak_demand'].mean()*100:.1f}%")

# Analyze conditions during peak demand
peak_analysis = df.groupby('is_peak_demand').agg({
    'hour': 'mean',
    'traffic_flow': 'mean',
    'temperature': 'mean',
    'no_of_evs_charging': 'mean',
    'charging_duration': 'mean',
    'public_holiday_or_event': 'mean'
}).round(2)

print("\nConditions during Peak vs Normal Demand:")
print(peak_analysis)

# Station-specific recommendations
station_performance = df.groupby('charging_station_id').agg({
    'charging_pile_usage_rate': ['mean', 'max', 'std'],
    'traffic_flow': 'mean',
    'population_density': 'mean'
}).round(3)

station_performance.columns = ['avg_usage', 'max_usage', 'usage_std', 'avg_traffic', 'avg_pop_density']
station_performance['utilization_ratio'] = station_performance['avg_usage'] / station_performance['max_usage']

print("Station Performance Analysis:")
print(station_performance.sort_values('avg_usage', ascending=False))

# Identify stations with highest potential for optimization
high_usage_stations = station_performance[station_performance['avg_usage'] > station_performance['avg_usage'].quantile(0.75)]
print(f"\nStations with highest usage (top 25%):")
print(high_usage_stations)



