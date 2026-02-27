import pandas as pd
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


# ==============================
# Create outputs directory
# ==============================
os.makedirs("outputs", exist_ok=True)


# ==============================
# Load Dataset
# ==============================
data = pd.read_csv("dataset/winequality-red.csv", sep=';')


# ==============================
# Feature Selection
# (Selecting important features manually)
# ==============================
selected_features = [
    'alcohol',
    'sulphates',
    'citric acid',
    'volatile acidity',
    'fixed acidity'
]

target = "quality"
# X = data[selected_features]
X = data.drop(target, axis=1)
y = data[target]


# ==============================
# Preprocessing (Scaling)
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==============================
# Train-Test Split
# ==============================
test=0.2
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test, random_state=42
)


# ==============================
# Model Training
# ==============================
modelName = "Random Forest est=50 dep=10"
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42
)

# alpha_value = 0.1
# model = Lasso(alpha=alpha_value)

model.fit(X_train, y_train)


# ==============================
# Prediction
# ==============================
y_pred = model.predict(X_test)


# ==============================
# Evaluation Metrics
# ==============================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# ==============================
# Print Metrics (for GitHub Actions logs)
# ==============================
print("===== Model Training Completed =====")
print(f"Model: {modelName}")
# print(f"Alpha: {alpha_value}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")


# ==============================
# Save Trained Model
# ==============================
joblib.dump(model, "outputs/model.pkl")


# ==============================
# Save Evaluation Results
# ==============================
results = {
    "Model": modelName,
    "Selected Features": "All features",
    "Test Size": test,
    "MSE": mse,
    "R2 Score": r2
}

with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=4)


print("Results and model saved successfully.")