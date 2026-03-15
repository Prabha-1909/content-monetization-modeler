import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned_youtube_data.csv")

print("Dataset Shape:", df.shape)

# ----------------------------------
# Feature Engineering
# ----------------------------------

# Engagement rate
df["engagement_rate"] = (df["likes"] + df["comments"]) / df["views"]

# Watch time per view
df["watch_time_per_view"] = df["watch_time_minutes"] / df["views"]

# ----------------------------------
# Save feature engineered dataset
# ----------------------------------

df.to_csv("data/featured_youtube_data.csv", index=False)

print("\nNew features created:")
print(df[["engagement_rate","watch_time_per_view"]].head())

print("\nFeature engineered dataset saved!")