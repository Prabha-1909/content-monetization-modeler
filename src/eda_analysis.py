import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/youtube_ad_revenue_dataset.csv")

# Show first rows
print("First 5 rows:")
print(df.head())

# Dataset info
print("\nDataset Info:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Basic statistics
print("\nStatistics:")
print(df.describe())

# ------------------------------
# Visualization
# ------------------------------

# Revenue distribution
plt.figure(figsize=(8,5))
sns.histplot(df["ad_revenue_usd"], bins=50)
plt.title("Revenue Distribution")
plt.show()

# Views vs Revenue
plt.figure(figsize=(8,5))
sns.scatterplot(x="views", y="ad_revenue_usd", data=df)
plt.title("Views vs Revenue")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()