import yfinance as yf
import pandas_ta as ta
import pandas as pd
import os

# ── All 5 IT Sector Stocks ──
STOCKS = {
    "TCS":      "TCS.NS",
    "Infosys":  "INFY.NS",
    "Wipro":    "WIPRO.NS",
    "HCLTech":  "HCLTECH.NS",
    "TechM":    "TECHM.NS"
}

# ── Create data folder ──
os.makedirs("data", exist_ok=True)

def fetch_and_compute(name, symbol):
    print(f"\nFetching {name} ({symbol})...")

    # Fetch 5 years of data
    stock = yf.Ticker(symbol)
    df = stock.history(period="5y")

    # Drop unwanted columns
    df = df.drop(columns=["Dividends", "Stock Splits"], errors="ignore")

    print(f"  Got {len(df)} rows of data")

    # ── Technical Indicators ──

    # Trend
    df["SMA_10"]  = ta.sma(df["Close"], length=10)
    df["SMA_20"]  = ta.sma(df["Close"], length=20)
    df["SMA_50"]  = ta.sma(df["Close"], length=50)
    df["SMA_200"] = ta.sma(df["Close"], length=200)
    df["EMA_9"]   = ta.ema(df["Close"], length=9)
    df["EMA_21"]  = ta.ema(df["Close"], length=21)
    df["EMA_55"]  = ta.ema(df["Close"], length=55)

    # MACD
    macd = ta.macd(df["Close"])
    df["MACD"]        = macd["MACD_12_26_9"]
    df["MACD_Signal"] = macd["MACDs_12_26_9"]
    df["MACD_Hist"]   = macd["MACDh_12_26_9"]

    # Momentum
    df["RSI_14"]  = ta.rsi(df["Close"], length=14)
    df["RSI_7"]   = ta.rsi(df["Close"], length=7)
    df["ROC"]     = ta.roc(df["Close"], length=10)
    df["MOM"]     = ta.mom(df["Close"], length=10)
    df["CCI"]     = ta.cci(df["High"], df["Low"], df["Close"], length=14)
    df["WILLR"]   = ta.willr(df["High"], df["Low"], df["Close"], length=14)
    df["MFI"]     = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=14)

    # Stochastic
    stoch = ta.stoch(df["High"], df["Low"], df["Close"])
    df["STOCH_K"] = stoch["STOCHk_14_3_3"]
    df["STOCH_D"] = stoch["STOCHd_14_3_3"]

    # Volatility - fixed column names
    bb = ta.bbands(df["Close"], length=20)
    bb_cols = bb.columns.tolist()
    df["BB_Upper"]  = bb[bb_cols[0]]
    df["BB_Lower"]  = bb[bb_cols[2]]
    df["BB_Middle"] = bb[bb_cols[3]]
    df["BB_Width"]  = bb[bb_cols[1]]
    df["ATR"]       = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # Volume
    df["OBV"]  = ta.obv(df["Close"], df["Volume"])
    df["VWAP"] = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])

    # Returns
    df["Return_1d"]  = df["Close"].pct_change(1)
    df["Return_5d"]  = df["Close"].pct_change(5)
    df["Return_20d"] = df["Close"].pct_change(20)

    # Drop rows with NaN
    df = df.dropna()

    print(f"  Indicators computed. Final shape: {df.shape}")

    # Save to CSV
    filename = f"data/{name}_data.csv"
    df.to_csv(filename)
    print(f"  Saved to {filename}")

    return df

# ── Run for all stocks ──
for name, symbol in STOCKS.items():
    fetch_and_compute(name, symbol)

print("\nAll stocks processed successfully!")
print("Check your 'data' folder for CSV files.")