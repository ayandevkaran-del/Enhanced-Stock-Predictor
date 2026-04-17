import pickle
import numpy as np

# Check what columns the scaler expects
with open("models/TCS_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("Scaler expects:", scaler.n_features_in_, "features")

# Check what columns the CSV has
import pandas as pd
df = pd.read_csv("data/TCS_data.csv", index_col=0)
df = df.select_dtypes(include=[np.number])
df = df.dropna()
print("CSV has:", len(df.columns), "columns")
print("\nColumn list:")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col}")