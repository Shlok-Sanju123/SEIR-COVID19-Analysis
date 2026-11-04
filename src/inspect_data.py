import pandas as pd

# Load the COVID-19 dataset
df = pd.read_csv("data/raw/states.csv")

# Display the first 5 rows
print("FIRST 5 ROWS:")
print(df.head())

# Display column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# Display dataset dimensions
print("\nDATASET SHAPE:")
print(df.shape)

# Display available states
print("\nAVAILABLE STATES:")
print(df["State"].unique())