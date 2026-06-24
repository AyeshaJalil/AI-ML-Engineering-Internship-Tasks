"""
Task 6: House Price Prediction

Objective:
Predict house prices using property features such as:
- Living Area
- Bedrooms
- Neighborhood
- Overall Quality
- Full Bath
- Garage Area

Models Used:
1. Linear Regression
2. Gradient Boosting Regressor

Evaluation Metrics:
- MAE
- RMSE
- R² Score
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==================================================
# Create Output Folder
# ==================================================

os.makedirs("output", exist_ok=True)

# ==================================================
# Load Dataset
# ==================================================

df = pd.read_csv("AmesHousing.csv")

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ==================================================
# Select Features
# ==================================================

features = [
    'Gr Liv Area',
    'Bedroom AbvGr',
    'Neighborhood',
    'Overall Qual',
    'Full Bath',
    'Garage Area'
]

target = 'SalePrice'

X = df[features].copy()
y = df[target]

# ==================================================
# Handle Missing Values
# ==================================================

# Numeric columns
for col in X.select_dtypes(include=['int64', 'float64']).columns:
    X[col] = X[col].fillna(X[col].median())

# Categorical columns
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna(X[col].mode()[0])

print("\nRemaining Missing Values:")
print(X.isnull().sum())

print("\nTotal Missing Values:")
print(X.isnull().sum().sum())

# ==================================================
# Encode Neighborhood
# ==================================================

encoder = LabelEncoder()
X['Neighborhood'] = encoder.fit_transform(X['Neighborhood'])

# ==================================================
# Feature Scaling
# ==================================================

scaler = StandardScaler()

numeric_cols = [
    'Gr Liv Area',
    'Bedroom AbvGr',
    'Overall Qual',
    'Full Bath',
    'Garage Area'
]

X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# ==================================================
# Train-Test Split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==================================================
# Linear Regression
# ==================================================

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
lr_r2 = r2_score(y_test, lr_predictions)

print("\n" + "=" * 50)
print("LINEAR REGRESSION RESULTS")
print("=" * 50)

print("MAE :", round(lr_mae, 2))
print("RMSE:", round(lr_rmse, 2))
print("R²  :", round(lr_r2, 4))

# ==================================================
# Gradient Boosting
# ==================================================

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_predictions = gb_model.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))
gb_r2 = r2_score(y_test, gb_predictions)

print("\n" + "=" * 50)
print("GRADIENT BOOSTING RESULTS")
print("=" * 50)

print("MAE :", round(gb_mae, 2))
print("RMSE:", round(gb_rmse, 2))
print("R²  :", round(gb_r2, 4))

# ==================================================
# Actual vs Predicted Graph
# ==================================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, gb_predictions)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()

plt.savefig(
    "output/actual_vs_predicted.png",
    dpi=300
)

plt.close()

print("\nactual_vs_predicted.png saved")

# ==================================================
# Feature Importance Graph
# ==================================================

importance = gb_model.feature_importances_

feature_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

feature_df = feature_df.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(8, 6))

plt.bar(
    feature_df['Feature'],
    feature_df['Importance']
)

plt.xticks(rotation=45)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")

plt.tight_layout()

plt.savefig(
    "output/feature_importance.png",
    dpi=300
)

plt.close()

print("feature_importance.png saved")

# ==================================================
# Model Comparison
# ==================================================

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Gradient Boosting'],
    'MAE': [lr_mae, gb_mae],
    'RMSE': [lr_rmse, gb_rmse],
    'R2 Score': [lr_r2, gb_r2]
})

print(comparison)

print("\nProject Completed Successfully!")
print("Graphs saved inside output folder.")