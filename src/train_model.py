import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv("data/featured_youtube_data.csv")

print("Dataset Shape:", df.shape)

# -----------------------------
# Encode categorical columns
# -----------------------------

encoder = LabelEncoder()

df["category"] = encoder.fit_transform(df["category"])
df["device"] = encoder.fit_transform(df["device"])
df["country"] = encoder.fit_transform(df["country"])

# -----------------------------
# Define features and target
# -----------------------------

X = df.drop(columns=["video_id", "date", "ad_revenue_usd"])
y = df["ad_revenue_usd"]

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# -----------------------------
# Define models
# -----------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}

# -----------------------------
# Train and evaluate models
# -----------------------------

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    results[name] = r2

    print("\n", name)
    print("R2 Score:", r2)
    print("MAE:", mae)
    print("RMSE:", rmse)

# -----------------------------
# Select best model
# -----------------------------

best_model_name = max(results, key=results.get)

print("\nBest Model:", best_model_name)

best_model = models[best_model_name]

# -----------------------------
# Save model
# -----------------------------

joblib.dump(best_model, "models/revenue_model.pkl")

print("\nModel saved in models/revenue_model.pkl")