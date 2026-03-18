import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("data/featured_youtube_data.csv")

# Encode categorical columns
encoder = LabelEncoder()

df["category"] = encoder.fit_transform(df["category"])
df["device"] = encoder.fit_transform(df["device"])
df["country"] = encoder.fit_transform(df["country"])

# Load trained model
model = joblib.load("models/revenue_model.pkl")

# Prepare features
X = df.drop(columns=["video_id", "date", "ad_revenue_usd"])
y = df["ad_revenue_usd"]

# Predict
predictions = model.predict(X)

# Metrics
r2 = r2_score(y, predictions)
mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))

print("Model Evaluation Results")
print("R2 Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)