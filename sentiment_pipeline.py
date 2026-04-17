import requests
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModel
import torch

# ── Config ──
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("NEWSAPI_KEY")

STOCKS = {
    "TCS":      "TCS Tata Consultancy",
    "Infosys":  "Infosys INFY",
    "Wipro":    "Wipro NSE",
    "HCLTech":  "HCL Technologies",
    "TechM":    "Tech Mahindra TECHM"
}

os.makedirs("data", exist_ok=True)

# ── Load FinBERT ──
print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModel.from_pretrained("ProsusAI/finbert")
model.eval()
print("FinBERT loaded!")

def get_embedding(text):
    # Convert headline to 768-dim embedding vector
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    # Use CLS token embedding as sentence representation
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding

def apply_attention(embeddings):
    # Simple self-attention: score each headline by its relevance
    # Headlines with stronger signals get higher weight
    if len(embeddings) == 0:
        return np.zeros(768)
    
    embeddings = np.array(embeddings)
    
    # Compute attention scores using dot product with mean
    mean_vec = embeddings.mean(axis=0)
    scores = embeddings @ mean_vec
    
    # Softmax to get weights
    scores = scores - scores.max()
    weights = np.exp(scores) / np.exp(scores).sum()
    
    # Weighted sum of embeddings
    attended = (embeddings * weights[:, None]).sum(axis=0)
    return attended

def fetch_news(query):
    url = (
        f"https://newsapi.org/v2/everything"
        f"?q={query}+stock+NSE+India"
        f"&language=en"
        f"&pageSize=20"
        f"&sortBy=publishedAt"
        f"&apiKey={API_KEY}"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get("status") == "ok":
            return data.get("articles", [])
        else:
            print(f"  API error: {data.get('message', 'unknown')}")
            return []
    except Exception as e:
        print(f"  Request failed: {e}")
        return []

def process_stock_sentiment(stock_name, query):
    print(f"\nProcessing sentiment for {stock_name}...")

    # Fetch news
    articles = fetch_news(query)
    print(f"  Fetched {len(articles)} articles")

    if len(articles) == 0:
        print(f"  No articles found, skipping...")
        return None

    # Get embeddings for each headline
    embeddings = []
    headlines = []

    for article in articles:
        title = article.get("title", "")
        if title and len(title) > 10:
            emb = get_embedding(title)
            embeddings.append(emb)
            headlines.append(title)

    print(f"  Generated {len(embeddings)} embeddings")

    if len(embeddings) == 0:
        return None

    # Apply attention
    attended_vector = apply_attention(embeddings)

    # Reduce to summary stats we can store in CSV
    sentiment_features = {
        "sentiment_mean":   float(attended_vector.mean()),
        "sentiment_std":    float(attended_vector.std()),
        "sentiment_max":    float(attended_vector.max()),
        "sentiment_min":    float(attended_vector.min()),
        "sentiment_norm":   float(np.linalg.norm(attended_vector)),
        "num_articles":     len(embeddings)
    }

    print(f"  Sentiment features computed")
    for k, v in sentiment_features.items():
        print(f"    {k}: {v:.4f}")

    return sentiment_features, attended_vector

def add_sentiment_to_csv(stock_name, sentiment_features):
    csv_path = f"data/{stock_name}_data.csv"

    if not os.path.exists(csv_path):
        print(f"  CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, index_col=0)

    # Add sentiment columns to all rows (current sentiment applied globally)
    # In production this would be date-aligned — for now we add as constant features
    for key, value in sentiment_features.items():
        df[key] = value

    df.to_csv(csv_path)
    print(f"  Updated {csv_path} with sentiment features")

# ── Run for all stocks ──
results = {}

for stock_name, query in STOCKS.items():
    result = process_stock_sentiment(stock_name, query)

    if result is not None:
        sentiment_features, vector = result
        results[stock_name] = sentiment_features
        add_sentiment_to_csv(stock_name, sentiment_features)

    # Wait between API calls to avoid rate limiting
    time.sleep(2)

print("\n" + "="*50)
print("SENTIMENT PIPELINE COMPLETE!")
print("="*50)
for stock, features in results.items():
    print(f"\n{stock}:")
    for k, v in features.items():
        print(f"  {k}: {v:.4f}")

print("\nAll CSV files updated with sentiment features!")