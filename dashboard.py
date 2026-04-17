import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import time
from datetime import datetime

# ── Page Config ──
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# ── Styling ──
st.markdown("""
<style>
    .main { background-color: #0D1B2A; }
    .stApp { background-color: #0D1B2A; color: white; }
    .metric-card {
        background: #1B4F72;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
    .buy-signal {
        background: #27AE60;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    .sell-signal {
        background: #E74C3C;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    .hold-signal {
        background: #F39C12;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

STOCKS = {
    "TCS":     "TCS.NS",
    "Infosys": "INFY.NS",
    "Wipro":   "WIPRO.NS",
    "HCLTech": "HCLTECH.NS",
    "TechM":   "TECHM.NS"
}

# ── Helper Functions ──
def get_prediction(stock):
    try:
        r = requests.get(f"{API_URL}/predict/{stock}", timeout=30)
        return r.json()
    except:
        return None

def get_all_predictions():
    try:
        r = requests.get(f"{API_URL}/predict_all", timeout=60)
        return r.json()
    except:
        return {}

def get_stock_history(symbol, period="3mo"):
    df = yf.Ticker(symbol).history(period=period)
    return df

def plot_candlestick(df, stock_name):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=stock_name,
        increasing_line_color="#27AE60",
        decreasing_line_color="#E74C3C"
    ))

    fig.update_layout(
        title=f"{stock_name} — Price Chart",
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#0D1B2A",
        font_color="white",
        xaxis_rangeslider_visible=False,
        height=400
    )

    return fig

def plot_rsi(df_history):
    import pandas_ta as ta
    rsi = ta.rsi(df_history["Close"], length=14)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_history.index, y=rsi,
        line=dict(color="#2E86C1", width=2),
        name="RSI"
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="#E74C3C", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="#27AE60", annotation_text="Oversold (30)")

    fig.update_layout(
        title="RSI (14)",
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#0D1B2A",
        font_color="white",
        height=250,
        yaxis=dict(range=[0, 100])
    )
    return fig

def plot_macd(df_history):
    import pandas_ta as ta
    macd_df = ta.macd(df_history["Close"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_history.index,
        y=macd_df["MACD_12_26_9"],
        line=dict(color="#2E86C1", width=2),
        name="MACD"
    ))
    fig.add_trace(go.Scatter(
        x=df_history.index,
        y=macd_df["MACDs_12_26_9"],
        line=dict(color="#E67E22", width=2),
        name="Signal"
    ))
    fig.add_bar(
        x=df_history.index,
        y=macd_df["MACDh_12_26_9"],
        name="Histogram",
        marker_color="#27AE60"
    )

    fig.update_layout(
        title="MACD",
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#0D1B2A",
        font_color="white",
        height=250
    )
    return fig

def plot_model_comparison():
    import json
    try:
        with open("models/comparison_results.json") as f:
            results = json.load(f)

        stocks = list(results.keys())
        model_names = ["RNN", "CNN", "LSTM", "BiLSTM", "BiLSTM_Attention"]
        colors = ["#E74C3C", "#E67E22", "#F4D03F", "#2E86C1", "#27AE60"]

        fig = go.Figure()
        for model, color in zip(model_names, colors):
            accs = [results[s].get(model, {}).get("accuracy", 0) for s in stocks]
            fig.add_trace(go.Bar(
                name=model.replace("_", "+"),
                x=stocks,
                y=accs,
                marker_color=color
            ))

        fig.add_hline(y=0.5, line_dash="dash", line_color="white",
                      annotation_text="Random Baseline (50%)")

        fig.update_layout(
            title="Model Comparison — Directional Accuracy",
            barmode="group",
            paper_bgcolor="#0D1B2A",
            plot_bgcolor="#0D1B2A",
            font_color="white",
            height=400,
            yaxis=dict(range=[0.4, 0.7], title="Accuracy"),
            legend=dict(bgcolor="#1B4F72")
        )
        return fig
    except:
        return None

# ── MAIN DASHBOARD ──
st.title("📈 Enhanced Stock Market Prediction System")
st.markdown("**Indian IT Sector — NSE India | BiLSTM + Transformer Attention**")
st.markdown("---")

# Sidebar
st.sidebar.title("⚙️ Controls")
selected_stock = st.sidebar.selectbox("Select Stock", list(STOCKS.keys()))
period = st.sidebar.selectbox("Chart Period", ["1mo", "3mo", "6mo", "1y"], index=1)
auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)

if st.sidebar.button("🔄 Refresh Now"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**About:**")
st.sidebar.markdown("BiLSTM + Transformer Attention model trained on 5 years of NSE data")
st.sidebar.markdown("Supervisor: Ms. Neetu Sardana")

# ── SECTION 1: All Signals Overview ──
st.subheader("🎯 Live Signals — All 5 IT Stocks")

with st.spinner("Fetching predictions..."):
    all_preds = get_all_predictions()

if all_preds:
    cols = st.columns(5)
    for i, (stock, pred) in enumerate(all_preds.items()):
        with cols[i]:
            if "error" not in pred:
                signal = pred.get("signal", "N/A")
                conf   = pred.get("confidence", 0)
                price  = pred.get("current_price", 0)

                color = "#27AE60" if signal == "BUY" else "#E74C3C"
                st.markdown(f"""
                <div style="background:{color};border-radius:10px;padding:15px;text-align:center;margin:3px">
                    <div style="font-size:18px;font-weight:bold;color:white">{stock}</div>
                    <div style="font-size:28px;font-weight:bold;color:white">{signal}</div>
                    <div style="font-size:14px;color:white">Confidence: {conf}%</div>
                    <div style="font-size:14px;color:white">₹{price}</div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("---")

# ── SECTION 2: Detailed Stock View ──
st.subheader(f"📊 Detailed View — {selected_stock}")

col1, col2 = st.columns([2, 1])

with col1:
    df_history = get_stock_history(STOCKS[selected_stock], period=period)
    if not df_history.empty:
        st.plotly_chart(plot_candlestick(df_history, selected_stock),
                        use_container_width=True)

with col2:
    pred = get_prediction(selected_stock)
    if pred and "error" not in pred:
        st.markdown("### 🤖 Model Prediction")

        signal = pred.get("signal", "N/A")
        color  = "#27AE60" if signal == "BUY" else "#E74C3C"

        st.markdown(f"""
        <div style="background:{color};border-radius:10px;padding:20px;text-align:center;margin-bottom:15px">
            <div style="font-size:36px;font-weight:bold;color:white">{signal}</div>
            <div style="font-size:18px;color:white">Confidence: {pred['confidence']}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Current Price", f"₹{pred['current_price']}")
        st.metric("RSI (14)", f"{pred['rsi']}")
        st.metric("MACD", f"{pred['macd']}")
        st.metric("Timestamp", pred['timestamp'])

        # RSI interpretation
        rsi_val = pred['rsi']
        if rsi_val > 70:
            st.warning("⚠️ RSI > 70 — Overbought zone")
        elif rsi_val < 30:
            st.success("✅ RSI < 30 — Oversold zone (potential buy)")
        else:
            st.info("ℹ️ RSI in neutral zone (30-70)")

# ── SECTION 3: Technical Indicators ──
st.markdown("---")
st.subheader("📉 Technical Indicators")

col3, col4 = st.columns(2)
with col3:
    if not df_history.empty:
        st.plotly_chart(plot_rsi(df_history), use_container_width=True)

with col4:
    if not df_history.empty:
        st.plotly_chart(plot_macd(df_history), use_container_width=True)

# ── SECTION 4: Model Comparison ──
st.markdown("---")
st.subheader("🏆 Model Comparison — Directional Accuracy")

fig_comparison = plot_model_comparison()
if fig_comparison:
    st.plotly_chart(fig_comparison, use_container_width=True)

# ── SECTION 5: All Predictions Table ──
st.markdown("---")
st.subheader("📋 All Predictions Summary")

if all_preds:
    table_data = []
    for stock, pred in all_preds.items():
        if "error" not in pred:
            table_data.append({
                "Stock":      stock,
                "Signal":     pred.get("signal", "N/A"),
                "Confidence": f"{pred.get('confidence', 0)}%",
                "Price":      f"₹{pred.get('current_price', 0)}",
                "RSI":        pred.get("rsi", 0),
                "MACD":       pred.get("macd", 0),
                "Time":       pred.get("timestamp", "")
            })

    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

# Auto refresh
if auto_refresh:
    time.sleep(30)
    st.rerun()

st.markdown("---")
st.markdown("*Enhanced Stock Market Prediction System | JIIT Noida | Supervisor: Ms. Neetu Sardana*")