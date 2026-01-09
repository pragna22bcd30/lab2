import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# Load dataset
data = pd.read_csv("data/winequality-red.csv", sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# Save model
joblib.dump(model, "outputs/model/model.pkl")

# Save metrics
results = {"MSE": mse, "R2": r2}
with open("outputs/results/metrics.json", "w") as f:
    json.dump(results, f, indent=4)
