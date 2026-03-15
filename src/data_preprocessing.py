import pandas as pd

# Load dataset
df = pd.read_csv("data/youtube_ad_revenue_dataset.csv")

print("Original Shape:", df.shape)

# -----------------------------
# Remove duplicates
# -----------------------------

df = df.drop_duplicates()

print("After removing duplicates:", df.shape)

# -----------------------------
# Handle missing values
# -----------------------------

df["likes"].fillna(df["likes"].median(), inplace=True)
df["comments"].fillna(df["comments"].median(), inplace=True)
df["watch_time_minutes"].fillna(df["watch_time_minutes"].median(), inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# -----------------------------
# Convert date column
# -----------------------------

df["date"] = pd.to_datetime(df["date"])

# -----------------------------
# Save cleaned dataset
# -----------------------------

df.to_csv("data/cleaned_youtube_data.csv", index=False)

print("\nCleaned dataset saved!")