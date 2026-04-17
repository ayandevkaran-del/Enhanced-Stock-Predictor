import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import os
import json

# ── Config ──
STOCKS = ["TCS", "Infosys", "Wipro", "HCLTech", "TechM"]
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

os.makedirs("models", exist_ok=True)

# ════════════════════════════════════════
# MODEL DEFINITIONS
# ════════════════════════════════════════

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


class CNNModel(nn.Module):
    def __init__(self, input_size, seq_len=60):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, batch_first=True,
                              num_layers=2, dropout=0.2, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.fc(out[:, -1, :])


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
        # Use mean of attention output
        out = attn_out.mean(dim=1)
        return self.fc(out)


# ════════════════════════════════════════
# TRAINING FUNCTION
# ════════════════════════════════════════

def train_model(model, X_train, y_train, X_test, y_test, model_name, stock_name):
    # Convert to tensors
    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.LongTensor(y_train).to(DEVICE)
    X_te = torch.FloatTensor(X_test).to(DEVICE)
    y_te = torch.LongTensor(y_test).to(DEVICE)

    # DataLoader
    dataset = TensorDataset(X_tr, y_tr)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    best_acc = 0
    best_epoch = 0

    print(f"\n  Training {model_name}...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_out = model(X_te)
            preds = test_out.argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y_test, preds)

        scheduler.step(total_loss)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(),
                       f"models/{stock_name}_{model_name}.pt")

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Acc: {acc:.4f}")

    # Final evaluation with best model
    model.load_state_dict(torch.load(f"models/{stock_name}_{model_name}.pt"))
    model.eval()
    with torch.no_grad():
        test_out = model(X_te)
        preds = test_out.argmax(dim=1).cpu().numpy()
        final_acc = accuracy_score(y_test, preds)
        final_f1  = f1_score(y_test, preds, average="weighted")

    print(f"    Best Accuracy: {best_acc:.4f} (epoch {best_epoch})")
    print(f"    Final Acc: {final_acc:.4f} | F1: {final_f1:.4f}")

    return {"accuracy": round(final_acc, 4), "f1": round(final_f1, 4)}


# ════════════════════════════════════════
# MAIN — TRAIN ALL MODELS FOR ALL STOCKS
# ════════════════════════════════════════

all_results = {}

for stock in STOCKS:
    print(f"\n{'='*50}")
    print(f"TRAINING MODELS FOR: {stock}")
    print(f"{'='*50}")

    # Load data
    X_train = np.load(f"models/{stock}_X_train.npy")
    X_test  = np.load(f"models/{stock}_X_test.npy")
    y_train = np.load(f"models/{stock}_y_train.npy")
    y_test  = np.load(f"models/{stock}_y_test.npy")

    input_size = X_train.shape[2]
    print(f"Input size: {input_size} features")

    stock_results = {}

    # Define all 5 models
    models_to_train = {
        "RNN":              RNNModel(input_size),
        "CNN":              CNNModel(input_size),
        "LSTM":             LSTMModel(input_size),
        "BiLSTM":           BiLSTMModel(input_size),
        "BiLSTM_Attention": BiLSTMAttentionModel(input_size),
    }

    for model_name, model in models_to_train.items():
        result = train_model(
            model, X_train, y_train,
            X_test, y_test,
            model_name, stock
        )
        stock_results[model_name] = result

    all_results[stock] = stock_results

# ── Save results ──
with open("models/comparison_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

# ── Print comparison table ──
print("\n" + "="*70)
print("FINAL COMPARISON RESULTS")
print("="*70)
print(f"{'Stock':<12} {'RNN':>8} {'CNN':>8} {'LSTM':>8} {'BiLSTM':>8} {'BiLSTM+Attn':>12}")
print("-"*70)

for stock, results in all_results.items():
    rnn  = results.get("RNN",              {}).get("accuracy", 0)
    cnn  = results.get("CNN",              {}).get("accuracy", 0)
    lstm = results.get("LSTM",             {}).get("accuracy", 0)
    bi   = results.get("BiLSTM",           {}).get("accuracy", 0)
    bia  = results.get("BiLSTM_Attention", {}).get("accuracy", 0)
    print(f"{stock:<12} {rnn:>8.4f} {cnn:>8.4f} {lstm:>8.4f} {bi:>8.4f} {bia:>12.4f}")

print("="*70)
print("\nAll models saved to models/ folder!")
print("Results saved to models/comparison_results.json")