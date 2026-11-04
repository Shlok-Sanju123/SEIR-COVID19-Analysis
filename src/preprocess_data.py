
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("results").mkdir(exist_ok=True)
Path("data/processed").mkdir(exist_ok=True)

# Load and extract Maharashtra data
df = pd.read_csv("data/raw/states.csv")
mh_data = df[df["State"] == "Maharashtra"].copy()
mh_data["Date"] = pd.to_datetime(mh_data["Date"])
mh_data = mh_data.sort_values("Date").reset_index(drop=True)

# Inspect data quality
print("Dataset shape:", mh_data.shape)
print("Date range:", mh_data["Date"].min(), "to", mh_data["Date"].max())
print("\nMissing values:\n", mh_data.isnull().sum())
print("Duplicate dates:", mh_data["Date"].duplicated().sum())
print("Date intervals:\n", mh_data["Date"].diff().value_counts().head())

# Calculate daily new confirmed cases
mh_data["New_Cases"] = mh_data["Confirmed"].diff()
mh_data["New_Cases"] = mh_data["New_Cases"].fillna(mh_data["Confirmed"])

# Identify negative reporting revisions
negative_cases = mh_data[mh_data["New_Cases"] < 0]
print("\nNegative daily case records:")
print(negative_cases[["Date", "Confirmed", "New_Cases"]])
print("Number of negative records:", len(negative_cases))

# Replace negative values with zero
mh_data["New_Cases"] = mh_data["New_Cases"].clip(lower=0)

print("Negative values remaining:", (mh_data["New_Cases"] < 0).sum())

# Plot daily new cases
plt.figure(figsize=(12, 5))
plt.plot(mh_data["Date"], mh_data["New_Cases"])
plt.title("Daily New COVID-19 Cases in Maharashtra")
plt.xlabel("Date")
plt.ylabel("New Cases")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/daily_cases.png", dpi=300)
plt.show()

# Save processed dataset
mh_data.to_csv("data/processed/maharashtra_covid.csv", index=False)
print("\nProcessed dataset saved successfully!")
