import pandas as pd
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
# from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score


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

X = data[selected_features]
y = data["quality"]


# ==============================
# Preprocessing (Scaling)
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==============================
# Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# ==============================
# Model Training
# ==============================
model = LinearRegression()

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
print(f"Model: Linear Regression")
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
    "Model": "Lasso",
    "Alpha": alpha_value,
    "Selected Features": selected_features,
    "Test Size": 0.2,
    "MSE": mse,
    "R2 Score": r2
}

with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=4)


print("Results and model saved successfully.")