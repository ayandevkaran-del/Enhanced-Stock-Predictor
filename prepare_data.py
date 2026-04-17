import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

# Configuration
SEQUENCE_LENGTH = 60  # 60 days lookback window
TEST_SPLIT = 0.2      # 20% for testing

STOCKS = ["TCS", "Infosys", "Wipro", "HCLTech", "TechM"]

os.makedirs("models", exist_ok=True)

def prepare_stock_data(stock_name):
    print(f"\nPreparing data for {stock_name}...")

    # Load CSV
    df = pd.read_csv(f"data/{stock_name}_data.csv", index_col=0)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Drop non-numeric or problematic columns
    df = df.select_dtypes(include=[np.number])
    df = df.dropna()

    # Create target variable
    # 1 = price went UP next day, 0 = price went DOWN
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    # Separate features and target
    feature_cols = [c for c in df.columns if c != "Target"]
    X = df[feature_cols].values
    y = df["Target"].values

    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Samples: {len(X)}")

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler for later use in live inference
    with open(f"models/{stock_name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Create sequences of 60 days
    X_seq, y_seq = [], []
    for i in range(SEQUENCE_LENGTH, len(X_scaled)):
        X_seq.append(X_scaled[i-SEQUENCE_LENGTH:i])
        y_seq.append(y[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    print(f"  Sequences created: {X_seq.shape}")

    # Train/test split (no shuffling — time series!)
    split = int(len(X_seq) * (1 - TEST_SPLIT))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Save prepared data
    np.save(f"models/{stock_name}_X_train.npy", X_train)
    np.save(f"models/{stock_name}_X_test.npy",  X_test)
    np.save(f"models/{stock_name}_y_train.npy", y_train)
    np.save(f"models/{stock_name}_y_test.npy",  y_test)

    print(f"  Saved to models folder")

    return X_train.shape

# Run for all stocks
shapes = {}
for stock in STOCKS:
    shape = prepare_stock_data(stock)
    shapes[stock] = shape

print("\n" + "="*50)
print("DATA PREPARATION COMPLETE!")
print("="*50)
for stock, shape in shapes.items():
    print(f"  {stock}: {shape}")

print(f"\nSequence length: {SEQUENCE_LENGTH} days")
print(f"Each model input shape: ({SEQUENCE_LENGTH}, {shapes['TCS'][2]})")