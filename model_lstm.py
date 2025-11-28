# model_lstm_torch.py
# -------------------------------------------------------
# Daily LSTM (PyTorch CPU) for NYISO load forecasting
# Mirrors the TF script's I/O so you can compare models.
# -------------------------------------------------------
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ====================== CONFIG ======================
DATA_PATH = Path("./Data/processed/nyiso_weather_merged.csv")
TEST_START_DATE = "2024-01-01"
VAL_DAYS = 90
LOOKBACK = 30
RANDOM_STATE = 42
OUT_DIR_PROCESSED = Path("./Data/processed/")
OUT_DIR_OUTPUTS = Path("./Outputs/")
BATCH_SIZE = 32
EPOCHS = 200
PATIENCE = 10
LR = 1e-3
# ====================================================

rng = np.random.default_rng(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

def load_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")
    df = pd.read_csv(path, parse_dates=["datetime"]).sort_values("datetime")
    daily = (
        df.set_index("datetime")
          .resample("D")
          .mean(numeric_only=True)
          .reset_index()
          .rename(columns={"datetime": "date"})
    )
    need = {"date", "load_mw"}
    missing = need - set(daily.columns)
    if missing:
        raise ValueError(f"Missing required columns after resample: {missing}")
    for col in ["temp_f", "humidity", "wind_speed", "precip"]:
        if col not in daily.columns:
            daily[col] = np.nan
    daily[["temp_f","humidity","wind_speed","precip"]] = (
        daily[["temp_f","humidity","wind_speed","precip"]].ffill().bfill()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    daily["dayofweek"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["is_weekend"] = (daily["dayofweek"] >= 5).astype(int)
    return daily

def split_by_date(df: pd.DataFrame, test_start: str, val_days: int):
    test_start = pd.to_datetime(test_start)
    df_tv = df[df["date"] < test_start].copy()
    df_test = df[df["date"] >= test_start].copy()
    if len(df_tv) <= val_days + LOOKBACK + 1:
        raise ValueError("Not enough history to carve validation after LOOKBACK.")
    val_cut = df_tv["date"].max() - pd.Timedelta(days=val_days - 1)
    df_train = df_tv[df_tv["date"] < val_cut].copy()
    df_val = df_tv[df_tv["date"] >= val_cut].copy()
    return df_train, df_val, df_test

def scale_fit_transform(train_df, val_df, test_df, feature_cols, target_col):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train = x_scaler.fit_transform(train_df[feature_cols].values)
    y_train = y_scaler.fit_transform(train_df[[target_col]].values)
    X_val = x_scaler.transform(val_df[feature_cols].values)
    y_val = y_scaler.transform(val_df[[target_col]].values)
    X_test = x_scaler.transform(test_df[feature_cols].values)
    y_test = y_scaler.transform(test_df[[target_col]].values)
    return (X_train, y_train, X_val, y_val, X_test, y_test, x_scaler, y_scaler)

def make_sequences(X: np.ndarray, y: np.ndarray, dates: np.ndarray, lookback: int):
    X_seq, y_seq, y_dates = [], [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i, :])
        y_seq.append(y[i, 0])
        y_dates.append(dates[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32), np.array(y_dates)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # (N, T, F)
        self.y = torch.from_numpy(y).unsqueeze(-1)  # (N, 1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMReg(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)          # (B, T, H)
        out = out[:, -1, :]            # last timestep
        out = self.drop(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(y_true == 0, 1e-9, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def plot_forecast(train_df, val_df, test_dates, y_true, y_pred, out_path: Path):
    plt.figure(figsize=(12,5))
    plt.plot(train_df["date"], train_df["load_mw"], label="Train")
    if len(val_df) > 0:
        plt.plot(val_df["date"], val_df["load_mw"], label="Validation")
    plt.plot(pd.to_datetime(test_dates), y_true, label="Actual (Test)")
    plt.plot(pd.to_datetime(test_dates), y_pred, label="LSTM Forecast")
    plt.title("Daily Load Forecast — Actual vs Forecast (LSTM, PyTorch)")
    plt.xlabel("Date"); plt.ylabel("Load (MW)"); plt.legend(); plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()

def main():
    OUT_DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT_DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)

    daily = load_daily(DATA_PATH)
    target_col = "load_mw"
    feature_cols = [
        "load_mw", "temp_f", "humidity", "wind_speed", "precip",
        "dayofweek", "month", "is_weekend"
    ]

    df_train, df_val, df_test = split_by_date(daily, TEST_START_DATE, VAL_DAYS)

    (Xtr, ytr, Xv, yv, Xte, yte, xsc, ysc) = scale_fit_transform(
        df_train, df_val, df_test, feature_cols, target_col
    )

    Xtr_seq, ytr_seq, dtr = make_sequences(Xtr, ytr, df_train["date"].values, LOOKBACK)
    Xv_seq, yv_seq, dv = make_sequences(Xv, yv, df_val["date"].values, LOOKBACK)
    Xte_seq, yte_seq, dte = make_sequences(Xte, yte, df_test["date"].values, LOOKBACK)

    if len(Xtr_seq) == 0 or len(Xte_seq) == 0:
        raise ValueError("Not enough samples after LOOKBACK to form sequences.")

    train_ds = SeqDataset(Xtr_seq, ytr_seq)
    val_ds = SeqDataset(Xv_seq, yv_seq) if len(Xv_seq) > 0 else None
    test_ds = SeqDataset(Xte_seq, yte_seq)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False) if val_ds else None

    n_features = Xtr_seq.shape[2]
    model = LSTMReg(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = PATIENCE
    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(xb)
        train_loss /= len(train_ds)

        if val_dl:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    val_loss += float(loss.item()) * len(xb)
            val_loss /= len(val_ds)
            print(f"Epoch {epoch:03d} - train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

            if val_loss + 1e-10 < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = PATIENCE
            else:
                patience -= 1
                if patience == 0:
                    print("Early stopping.")
                    break
        else:
            print(f"Epoch {epoch:03d} - train_loss={train_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict on TEST
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(torch.from_numpy(Xte_seq)).cpu().numpy().reshape(-1, 1)

    y_true = ysc.inverse_transform(yte_seq.reshape(-1,1)).reshape(-1)
    y_pred = ysc.inverse_transform(y_pred_scaled).reshape(-1)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mape_val = float(mape(y_true, y_pred))
    print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape_val:.2f}%")

    # Save metrics
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE%": mape_val,
        "lookback_days": LOOKBACK,
        "val_days": VAL_DAYS,
        "test_start": TEST_START_DATE
    }
    out_metrics = OUT_DIR_PROCESSED / "lstm_daily_metrics.json"
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved: {out_metrics}")

    # Save forecast CSV
    forecast_df = pd.DataFrame({
        "date": pd.to_datetime(dte),
        "y_true": y_true,
        "y_pred": y_pred
    }).sort_values("date")
    out_forecast = OUT_DIR_PROCESSED / "lstm_daily_forecast.csv"
    forecast_df.to_csv(out_forecast, index=False)
    print(f"✅ Saved: {out_forecast}  shape={forecast_df.shape}")

    # Plot
    out_plot = OUT_DIR_OUTPUTS / "lstm_daily_actual_vs_forecast.png"
    plot_forecast(df_train, df_val, forecast_df["date"].values,
                  forecast_df["y_true"].values, forecast_df["y_pred"].values, out_plot)
    print(f"✅ Saved: {out_plot}")

    # Save model
    out_model = OUT_DIR_OUTPUTS / "lstm_daily_model.pt"
    torch.save(model.state_dict(), out_model)
    print(f"✅ Saved: {out_model}")

if __name__ == "__main__":
    main()
