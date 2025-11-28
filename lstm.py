# model_lstm.py
# -------------------------------------------------------
# Daily LSTM for NYISO load forecasting (aligned to your project)
# - Loads: ./Data/processed/nyiso_weather_merged.csv
# - Creates calendar + weather features
# - Builds sliding windows (LOOKBACK days) for sequences
# - Split: train/val (< TEST_START_DATE), test (>= TEST_START_DATE)
# - Early stopping on validation set
# - Outputs:
#     ./Data/processed/lstm_daily_forecast.csv
#     ./Data/processed/lstm_daily_metrics.json
#     ./Outputs/lstm_daily_actual_vs_forecast.png
#     ./Outputs/lstm_daily_model.keras  (saved model)
# -------------------------------------------------------

from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quiet TF logs

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ====================== CONFIG ======================
DATA_PATH = Path("./Data/processed/nyiso_weather_merged.csv")
TEST_START_DATE = "2024-01-01"
VAL_DAYS = 90            # last N days before TEST_START used for validation
LOOKBACK = 30            # days of history to predict next day
RANDOM_STATE = 42
OUT_DIR_PROCESSED = Path("./Data/processed/")
OUT_DIR_OUTPUTS = Path("./Outputs/")
# ====================================================

# Reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def load_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    df = pd.read_csv(path, parse_dates=["datetime"]).sort_values("datetime")

    # Daily means (numeric only)
    daily = (
        df.set_index("datetime")
          .resample("D")
          .mean(numeric_only=True)
          .reset_index()
          .rename(columns={"datetime": "date"})
    )

    # Ensure required columns
    need = {"date", "load_mw"}
    missing = need - set(daily.columns)
    if missing:
        raise ValueError(f"Missing required columns after resample: {missing}")

    # Weather columns (create if missing, will be imputed)
    for col in ["temp_f", "humidity", "wind_speed", "precip"]:
        if col not in daily.columns:
            daily[col] = np.nan

    # Impute/weather smoothing
    daily[["temp_f", "humidity", "wind_speed", "precip"]] = (
        daily[["temp_f", "humidity", "wind_speed", "precip"]].ffill().bfill()
    )

    # Calendar features
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    daily["dayofweek"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["is_weekend"] = (daily["dayofweek"] >= 5).astype(int)

    return daily


def split_by_date(df: pd.DataFrame, test_start: str, val_days: int):
    test_start = pd.to_datetime(test_start)

    # Train+Val are strictly before test_start
    df_tv = df[df["date"] < test_start].copy()
    df_test = df[df["date"] >= test_start].copy()

    if len(df_tv) <= val_days + LOOKBACK + 1:
        raise ValueError("Not enough history to carve validation after applying LOOKBACK.")

    # Validation: last N calendar days BEFORE test start
    val_cut = df_tv["date"].max() - pd.Timedelta(days=val_days - 1)
    df_train = df_tv[df_tv["date"] < val_cut].copy()
    df_val = df_tv[df_tv["date"] >= val_cut].copy()

    return df_train, df_val, df_test


def scale_fit_transform(train_df, val_df, test_df, feature_cols, target_col):
    """
    Fit scalers on TRAIN ONLY, transform train/val/test.
    Returns scaled arrays and fitted scalers.
    """
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train = x_scaler.fit_transform(train_df[feature_cols].values)
    y_train = y_scaler.fit_transform(train_df[[target_col]].values)

    X_val = x_scaler.transform(val_df[feature_cols].values)
    y_val = y_scaler.transform(val_df[[target_col]].values)

    X_test = x_scaler.transform(test_df[feature_cols].values)
    y_test = y_scaler.transform(test_df[[target_col]].values)

    return (X_train, y_train, X_val, y_val, X_test, y_test, x_scaler, y_scaler)


def make_sequences(X: np.ndarray,
                   y: np.ndarray,
                   dates: np.ndarray,
                   lookback: int):
    """
    Build (samples, timesteps, features) windows.
    Align each y[i] as the target for sequence X[i-lookback:i].
    Also return the corresponding target dates (dates[i]).
    """
    X_seq, y_seq, y_dates = [], [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i, :])
        y_seq.append(y[i, 0])             # y is shape (n,1)
        y_dates.append(dates[i])          # the target's date
    return np.array(X_seq), np.array(y_seq), np.array(y_dates)


def build_model(n_timesteps: int, n_features: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(64, input_shape=(n_timesteps, n_features), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(y_true == 0, 1e-9, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def plot_forecast(train_df, val_df, test_df, y_true, y_pred, out_path: Path):
    plt.figure(figsize=(12, 5))
    # Plot TRAIN + VAL actuals for context
    plt.plot(train_df["date"], train_df["load_mw"], label="Train")
    if len(val_df) > 0:
        plt.plot(val_df["date"], val_df["load_mw"], label="Validation")
    # Test actual vs predicted
    plt.plot(test_df["date"].values, y_true, label="Actual (Test)")
    plt.plot(test_df["date"].values, y_pred, label="LSTM Forecast")
    plt.title("Daily Load Forecast — Actual vs Forecast (LSTM)")
    plt.xlabel("Date")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    OUT_DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT_DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    daily = load_daily(DATA_PATH)

    # Feature/target columns
    target_col = "load_mw"
    feature_cols = [
        # Use target + exogenous features as inputs
        "load_mw", "temp_f", "humidity", "wind_speed", "precip",
        "dayofweek", "month", "is_weekend",
    ]

    # 2) Split by date
    df_train, df_val, df_test = split_by_date(daily, TEST_START_DATE, VAL_DAYS)

    # 3) Scale (fit on train only)
    (X_train_arr, y_train_arr,
     X_val_arr, y_val_arr,
     X_test_arr, y_test_arr,
     x_scaler, y_scaler) = scale_fit_transform(
        df_train, df_val, df_test, feature_cols, target_col
    )

    # 4) Build sequences AFTER scaling, keep aligned dates
    #    Important: sequences are continuous within each split
    X_train_seq, y_train_seq, dates_train = make_sequences(
        X_train_arr, y_train_arr, df_train["date"].values, LOOKBACK
    )
    X_val_seq, y_val_seq, dates_val = make_sequences(
        X_val_arr, y_val_arr, df_val["date"].values, LOOKBACK
    )
    X_test_seq, y_test_seq, dates_test = make_sequences(
        X_test_arr, y_test_arr, df_test["date"].values, LOOKBACK
    )

    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        raise ValueError("Not enough samples after LOOKBACK to form sequences.")

    n_timesteps = X_train_seq.shape[1]
    n_features = X_train_seq.shape[2]

    # 5) Model
    model = build_model(n_timesteps, n_features)

    es = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # 6) Train
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq) if len(X_val_seq) > 0 else None,
        epochs=100,
        batch_size=32,
        callbacks=[es],
        verbose=1
    )

    # 7) Predict on TEST
    y_test_pred_scaled = model.predict(X_test_seq, verbose=0)

    # Inverse scale
    y_true = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).reshape(-1)
    y_pred = y_scaler.inverse_transform(y_test_pred_scaled).reshape(-1)

    # 8) Metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mape_val = float(mape(y_true, y_pred))

    print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape_val:.2f}%")

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE%": mape_val,
        "lookback_days": LOOKBACK,
        "val_days": VAL_DAYS,
        "test_start": TEST_START_DATE
    }

    # 9) Save metrics
    out_metrics = OUT_DIR_PROCESSED / "lstm_daily_metrics.json"
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved: {out_metrics}")

    # 10) Build forecast df aligned to test TARGET dates that have sequences
    # dates_test corresponds to the target dates AFTER LOOKBACK within the test split
    forecast_df = pd.DataFrame({
        "date": pd.to_datetime(dates_test),
        "y_true": y_true,
        "y_pred": y_pred
    }).sort_values("date")

    out_forecast = OUT_DIR_PROCESSED / "lstm_daily_forecast.csv"
    forecast_df.to_csv(out_forecast, index=False)
    print(f"✅ Saved: {out_forecast}  shape={forecast_df.shape}")

    # 11) Plot Actual vs Forecast
    out_plot = OUT_DIR_OUTPUTS / "lstm_daily_actual_vs_forecast.png"
    plot_forecast(df_train, df_val, forecast_df.rename(columns={"date": "date"}),
                  forecast_df["y_true"].values, forecast_df["y_pred"].values, out_plot)
    print(f"✅ Saved: {out_plot}")

    # 12) Save model
    out_model = OUT_DIR_OUTPUTS / "lstm_daily_model.keras"
    model.save(out_model)
    print(f"✅ Saved: {out_model}")


if __name__ == "__main__":
    main()
