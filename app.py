from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import torch.nn as nn
import pickle
import yfinance as yf
import pandas_ta as ta
import pandas as pd
from datetime import datetime

app = FastAPI(title="Stock Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STOCKS = {
    "TCS":     "TCS.NS",
    "Infosys": "INFY.NS",
    "Wipro":   "WIPRO.NS",
    "HCLTech": "HCLTECH.NS",
    "TechM":   "TECHM.NS"
}

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_10", "SMA_20", "SMA_50", "SMA_200",
    "EMA_9", "EMA_21", "EMA_55",
    "MACD", "MACD_Signal", "MACD_Hist",
    "RSI_14", "RSI_7", "ROC", "MOM", "CCI", "WILLR", "MFI",
    "STOCH_K", "STOCH_D",
    "BB_Upper", "BB_Lower", "BB_Middle", "BB_Width",
    "ATR", "OBV", "VWAP",
    "Return_1d", "Return_5d", "Return_20d",
    "sentiment_mean", "sentiment_std", "sentiment_max",
    "sentiment_min", "sentiment_norm", "num_articles"
]

SEQUENCE_LENGTH = 60

class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, batch_first=True,
                              num_layers=2, dropout=0.2, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = attn_out.mean(dim=1)
        return self.fc(out)

# Load models and scalers
models = {}
scalers = {}

for stock in STOCKS.keys():
    try:
        with open(f"models/{stock}_scaler.pkl", "rb") as f:
            scalers[stock] = pickle.load(f)
        model = BiLSTMAttentionModel(input_size=40)
        model.load_state_dict(torch.load(
            f"models/{stock}_BiLSTM_Attention.pt",
            map_location=torch.device("cpu")
        ))
        model.eval()
        models[stock] = model
        print(f"Loaded model for {stock}")
    except Exception as e:
        print(f"Could not load model for {stock}: {e}")

def fetch_latest_data(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="2y")
    df = df.drop(columns=["Dividends", "Stock Splits"], errors="ignore")

    # Indicators
    df["SMA_10"]  = ta.sma(df["Close"], length=10)
    df["SMA_20"]  = ta.sma(df["Close"], length=20)
    df["SMA_50"]  = ta.sma(df["Close"], length=50)
    df["SMA_200"] = ta.sma(df["Close"], length=200)
    df["EMA_9"]   = ta.ema(df["Close"], length=9)
    df["EMA_21"]  = ta.ema(df["Close"], length=21)
    df["EMA_55"]  = ta.ema(df["Close"], length=55)

    macd = ta.macd(df["Close"])
    df["MACD"]        = macd["MACD_12_26_9"]
    df["MACD_Signal"] = macd["MACDs_12_26_9"]
    df["MACD_Hist"]   = macd["MACDh_12_26_9"]

    df["RSI_14"] = ta.rsi(df["Close"], length=14)
    df["RSI_7"]  = ta.rsi(df["Close"], length=7)
    df["ROC"]    = ta.roc(df["Close"], length=10)
    df["MOM"]    = ta.mom(df["Close"], length=10)
    df["CCI"]    = ta.cci(df["High"], df["Low"], df["Close"], length=14)
    df["WILLR"]  = ta.willr(df["High"], df["Low"], df["Close"], length=14)
    df["MFI"]    = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=14)

    stoch = ta.stoch(df["High"], df["Low"], df["Close"])
    df["STOCH_K"] = stoch["STOCHk_14_3_3"]
    df["STOCH_D"] = stoch["STOCHd_14_3_3"]

    bb = ta.bbands(df["Close"], length=20)
    bb_cols = bb.columns.tolist()
    df["BB_Upper"]  = bb[bb_cols[0]]
    df["BB_Lower"]  = bb[bb_cols[2]]
    df["BB_Middle"] = bb[bb_cols[3]]
    df["BB_Width"]  = bb[bb_cols[1]]

    df["ATR"]  = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["OBV"]  = ta.obv(df["Close"], df["Volume"])
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    df["Return_1d"]  = df["Close"].pct_change(1)
    df["Return_5d"]  = df["Close"].pct_change(5)
    df["Return_20d"] = df["Close"].pct_change(20)

    # Sentiment dummy values
    df["sentiment_mean"]  = 0.0
    df["sentiment_std"]   = 0.0
    df["sentiment_max"]   = 0.0
    df["sentiment_min"]   = 0.0
    df["sentiment_norm"]  = 0.0
    df["num_articles"]    = 0.0

    # Select exact columns in exact order
    df = df[FEATURE_COLUMNS]

    # Fill NaN then drop remaining
    df = df.ffill().bfill()
    df = df.dropna()

    print(f"  {symbol}: {len(df)} rows after processing")

    return df

def predict(stock_name):
    symbol = STOCKS[stock_name]
    df = fetch_latest_data(symbol)

    if len(df) < SEQUENCE_LENGTH:
        return {
            "error": f"Not enough data. Got {len(df)} rows, need {SEQUENCE_LENGTH}"
        }

    # Get last 60 days
    features = df.values[-SEQUENCE_LENGTH:]

    # Scale
    scaler = scalers[stock_name]
    features_scaled = scaler.transform(features)

    # Predict
    x = torch.FloatTensor(features_scaled).unsqueeze(0)
    model = models[stock_name]

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1).numpy()[0]
        pred = output.argmax(dim=1).item()

    signal     = "BUY" if pred == 1 else "SELL"
    confidence = round(float(probs[pred]) * 100, 1)

    return {
        "stock":         stock_name,
        "signal":        signal,
        "confidence":    confidence,
        "current_price": round(float(df["Close"].iloc[-1]), 2),
        "rsi":           round(float(df["RSI_14"].iloc[-1]), 2),
        "macd":          round(float(df["MACD"].iloc[-1]), 2),
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/")
def home():
    return {"message": "Stock Prediction API is running!"}

@app.get("/predict/{stock_name}")
def predict_stock(stock_name: str):
    if stock_name not in STOCKS:
        return {"error": f"Stock not found. Available: {list(STOCKS.keys())}"}
    return predict(stock_name)

@app.get("/predict_all")
def predict_all():
    results = {}
    for stock in STOCKS.keys():
        results[stock] = predict(stock)
    return results

@app.get("/stocks")
def get_stocks():
    return {"stocks": list(STOCKS.keys())}