import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- Prophet import (with fallback & clear error) ----
try:
    from prophet import Prophet
except Exception as e1:
    try:
        from fbprophet import Prophet  # older package name
    except Exception as e2:
        raise ImportError(
            "Prophet is not installed. Try:\n\n"
            "  pip install prophet\n\n"
            "If build errors occur, install a C++ toolchain and try again.\n"
            f"Import errors:\n- prophet: {e1}\n- fbprophet: {e2}"
        )

# ====================== CONFIG ======================
DATA_PATH = Path("./Data/processed/nyiso_weather_merged.csv")
RESAMPLE = "D"                       # daily mean
TEST_START_DATE = "2024-01-01"       # keep identical to SARIMA
FORECAST_HORIZON = 30                # next 30 daily steps
OUT_DIR_PROCESSED = Path("./Data/processed/")
OUT_DIR_OUTPUTS = Path("./Outputs/")
# ====================================================


def load_daily_series(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")
    df = pd.read_csv(path, parse_dates=["datetime"]).sort_values("datetime")
    # Daily mean, numeric only
    daily = df.resample("D", on="datetime").mean(numeric_only=True)
    # Ensure load column exists and has no NA
    if "load_mw" not in daily.columns:
        raise KeyError("Expected column 'load_mw' after resample.")
    daily = daily.dropna(subset=["load_mw"]).copy()
    daily = daily.reset_index().rename(columns={"datetime": "ds", "load_mw": "y"})
    return daily[["ds", "y"]]  # Prophet needs ds, y

def split_train_test(daily_df: pd.DataFrame, test_start: str):
    train = daily_df[daily_df["ds"] < test_start].copy()
    test = daily_df[daily_df["ds"] >= test_start].copy()
    if train.empty or test.empty:
        raise ValueError("Train/test is empty. Adjust TEST_START_DATE.")
    return train, test

def evaluate(y_true: np.ndarray, y_pred: np.ndarray):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    eps = 1e-6
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0
    return float(mae), float(rmse), float(mape)

def main():
    OUT_DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT_DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)

    # 1) Load & split
    daily = load_daily_series(DATA_PATH)
    train, test = split_train_test(daily, TEST_START_DATE)
    print(f"Train: {train['ds'].min().date()} → {train['ds'].max().date()}  |  "
          f"Test: {test['ds'].min().date()} → {test['ds'].max().date()}  "
          f"({len(train)} train rows, {len(test)} test rows)")

    # 2) Fit Prophet (yearly + weekly seasonality)
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",  # try "multiplicative" if peaks are under-fit
    )
    m.fit(train.rename(columns={"ds": "ds", "y": "y"}))

    # 3) Forecast over the test window
    future = pd.DataFrame({"ds": test["ds"]})
    fcst = m.predict(future)
    # Prophet output has 'yhat' as forecast
    test = test.copy()
    test["yhat"] = fcst["yhat"].values

    # 4) Evaluate
    mae, rmse, mape = evaluate(test["y"].values, test["yhat"].values)
    print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")

    # 5) Save forecast CSV (test window)
    out_forecast = OUT_DIR_PROCESSED / "prophet_daily_forecast.csv"
    test[["ds", "y", "yhat"]].rename(columns={"ds":"datetime", "y":"actual", "yhat":"forecast"}).to_csv(out_forecast, index=False)
    print(f"✅ Saved: {out_forecast}  shape={test.shape}")

    # 6) Save metrics JSON
    out_metrics = OUT_DIR_PROCESSED / "prophet_daily_metrics.json"
    metrics = {
        "model": "Prophet (yearly+weekly additive)",
        "resample": RESAMPLE,
        "test_start": TEST_START_DATE,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved: {out_metrics}")

    # 7) Plot Actual vs Forecast (test window)
    plt.figure(figsize=(12,5))
    plt.plot(train["ds"], train["y"], label="Train")
    plt.plot(test["ds"], test["y"], label="Actual (Test)")
    plt.plot(test["ds"], test["yhat"], label="Forecast (Prophet)")
    plt.title("Prophet Forecast vs Actual (Daily NY Load)")
    plt.xlabel("Date"); plt.ylabel("Load (MW)")
    plt.legend(); plt.tight_layout()
    out_plot = OUT_DIR_OUTPUTS / "prophet_daily_actual_vs_forecast.png"
    plt.savefig(out_plot, dpi=150)
    print(f"✅ Saved: {out_plot}")

    # 8) Next 30 days beyond the last date (refit on ALL data)
    m_full = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    m_full.fit(daily)
    last_date = daily["ds"].max()
    future_next = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_HORIZON, freq="D")
    fcst_next = m_full.predict(pd.DataFrame({"ds": future_next}))
    out_next = OUT_DIR_PROCESSED / "prophet_daily_next30.csv"
    fcst_next[["ds", "yhat"]].rename(columns={"ds":"datetime", "yhat":"forecast"}).to_csv(out_next, index=False)
    print(f"✅ Saved: {out_next}  shape={fcst_next.shape}")

if __name__ == "__main__":
    main()