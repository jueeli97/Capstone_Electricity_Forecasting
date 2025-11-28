

# model_xgboost.py
# -------------------------------------------------------
# Daily XGBoost Regressor for NYISO load forecasting
# - Loads ./Data/processed/nyiso_weather_merged.csv
# - Resamples to daily mean
# - Feature set:
#     * calendar: dayofweek, month, is_weekend
#     * lags: y(t-1), y(t-7), y(t-14), y(t-28)
#     * rolling means (shifted): mean_7, mean_14, mean_28
#     * weather: temp_f, humidity, wind_speed, precip (daily mean)
# - Train: < 2024-01-01, Test: >= 2024-01-01 (same as SARIMA/Prophet)
# - Early stopping using the last 90 days of the training set as validation
# - Outputs:
#     ./Data/processed/xgb_daily_forecast.csv
#     ./Data/processed/xgb_daily_metrics.json
#     ./Data/processed/xgb_daily_feature_importance.csv
#     ./Outputs/xgb_daily_actual_vs_forecast.png
# -------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ====================== CONFIG ======================
DATA_PATH = Path("./Data/processed/nyiso_weather_merged.csv")
TEST_START_DATE = "2024-01-01"
VAL_DAYS = 90  # last N days of train used as validation for early stopping
OUT_DIR_PROCESSED = Path("./Data/processed/")
OUT_DIR_OUTPUTS = Path("./Outputs/")
RANDOM_STATE = 42
# ====================================================


def load_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    df = pd.read_csv(path, parse_dates=["datetime"]).sort_values("datetime")

    # Daily mean aggregation for load and weather
    # Expect columns: ['datetime','load_mw','temp_f','humidity','wind_speed','precip', ...]
    # Extra columns are ignored in groupby mean
    daily = (
        df.set_index("datetime")
          .resample("D")
          .mean(numeric_only=True)
          .reset_index()
    )

    # Ensure required columns exist
    req = {"datetime", "load_mw"}
    missing = req - set(daily.columns)
    if missing:
        raise ValueError(f"Missing required columns after resample: {missing}")

    return daily


def make_features(daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create supervised learning features."""
    data = daily.copy()
    data = data.rename(columns={"datetime": "date"})
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").set_index("date")

    # Target
    data["y"] = data["load_mw"]

    # Calendar
    data["dayofweek"] = data.index.dayofweek
    data["month"] = data.index.month
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

    # Lags
    for lag in [1, 7, 14, 28]:
        data[f"lag_{lag}"] = data["y"].shift(lag)

    # Rolling means (shifted to avoid leakage)
    for win in [7, 14, 28]:
        data[f"mean_{win}"] = data["y"].shift(1).rolling(win).mean()

    # Weather (already daily means)
    # If any weather columns are missing, create them as NaN (will be imputed later)
    for col in ["temp_f", "humidity", "wind_speed", "precip"]:
        if col not in data.columns:
            data[col] = np.nan

    # Drop rows with NA from lags/rollings at the start
    data = data.dropna(subset=["lag_1", "lag_7", "lag_14", "lag_28",
                               "mean_7", "mean_14", "mean_28"])

    # Simple impute for weather (forward fill/back fill)
    data[["temp_f", "humidity", "wind_speed", "precip"]] = (
        data[["temp_f", "humidity", "wind_speed", "precip"]]
        .ffill()
        .bfill()
    )

    feature_cols = [
        # calendar
        "dayofweek", "month", "is_weekend",
        # lags
        "lag_1", "lag_7", "lag_14", "lag_28",
        # rollings
        "mean_7", "mean_14", "mean_28",
        # weather
        "temp_f", "humidity", "wind_speed", "precip",
    ]

    X = data[feature_cols]
    y = data["y"]

    return pd.concat([X, y], axis=1), feature_cols


def train_val_test_split(df_feat: pd.DataFrame, test_start: str, val_days: int):
    """Split by date: all < test_start => train+val; >= test_start => test (holdout).
    Use last `val_days` of pre-2024 data as validation."""
    df_feat = df_feat.copy()
    df_feat["date"] = df_feat.index

    train_all = df_feat[df_feat["date"] < pd.to_datetime(test_start)]
    test = df_feat[df_feat["date"] >= pd.to_datetime(test_start)]

    if len(train_all) <= val_days + 30:
        raise ValueError("Not enough training history to carve out a validation window.")

    val = train_all.iloc[-val_days:]
    train = train_all.iloc[:-val_days]

    X_train, y_train = train.drop(columns=["y", "date"]), train["y"]
    X_val, y_val = val.drop(columns=["y", "date"]), val["y"]
    X_test, y_test = test.drop(columns=["y", "date"]), test["y"]

    return (train, val, test), (X_train, y_train, X_val, y_val, X_test, y_test)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(y_true == 0, 1e-9, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def plot_forecast(train_idx, y_train, test_idx, y_true, y_pred, out_path: Path):
    plt.figure(figsize=(12, 5))
    plt.plot(train_idx, y_train, label="Train")
    plt.plot(test_idx, y_true, label="Actual")
    plt.plot(test_idx, y_pred, label="XGB Forecast")
    plt.title("Daily Load Forecast — Actual vs Forecast")
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

    # 1) Load & features
    daily = load_daily(DATA_PATH)
    df_feat, feature_cols = make_features(daily)

    # 2) Split
    (train, val, test), (X_train, y_train, X_val, y_val, X_test, y_test) = \
        train_val_test_split(df_feat, TEST_START_DATE, VAL_DAYS)

    # 3) Model
    # model = xgb.XGBRegressor(
    #     n_estimators=2000,
    #     learning_rate=0.03,
    #     max_depth=6,
    #     subsample=0.9,
    #     colsample_bytree=0.9,
    #     reg_lambda=1.0,
    #     random_state=RANDOM_STATE,
    #     tree_method="hist",
    #     objective="reg:squarederror",
    #     n_jobs=-1,
    #     eval_metric="rmse",
    #     early_stopping_rounds=100,  # <- in constructor to avoid deprecation warning
    # )

        # 3) Model
    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        tree_method="hist",
        objective="reg:squarederror",
        n_jobs=-1,
        eval_metric="rmse",
        early_stopping_rounds=100,
    )

    # 4) Fit with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # SAVE MODEL
    import joblib
    joblib.dump(model, OUT_DIR_PROCESSED / "xgb_daily_model.pkl")
    print("✅ Saved: xgb_daily_model.pkl")


    # 4) Fit with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = getattr(model, "best_iteration", None)

    # 5) Predict on test
    pred = model.predict(X_test)

    # 6) Metrics
    # mae = mean_absolute_error(y_test, pred)
    # rmse = mean_squared_error(y_test, pred, squared=False)
    # mape_val = mape(y_test.values, pred)

    # print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape_val:.2f}%")
    # if best_iter is not None:
    #     print(f"Best iteration: {best_iter}")

    # # Save metrics
    # metrics = {
    #     "MAE": float(mae),
    #     "RMSE": float(rmse),
    #     "MAPE%": float(mape_val),
    #     "best_iteration": int(best_iter) if best_iter is not None else None,
    #     "val_days": VAL_DAYS,
    #     "test_start": TEST_START_DATE,
    # }
    # out_metrics = OUT_DIR_PROCESSED / "xgb_daily_metrics.json"
    # with open(out_metrics, "w") as f:
    #     json.dump(metrics, f, indent=2)
    # print(f"✅ Saved: {out_metrics}")


    # 6) Metrics  (sklearn versions without `squared=False`)
    mae = mean_absolute_error(y_test, pred)

    # Older sklearn: mean_squared_error returns MSE only (no `squared` kwarg)
    mse = mean_squared_error(y_test, pred)
    rmse = float(np.sqrt(mse))

    mape_val = mape(y_test.values, pred)

    print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape_val:.2f}%")
    if best_iter is not None:
        print(f"Best iteration: {best_iter}")

    # Save metrics
    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE%": float(mape_val),
        "best_iteration": int(best_iter) if best_iter is not None else None,
        "val_days": VAL_DAYS,
        "test_start": TEST_START_DATE,
    }
    out_metrics = OUT_DIR_PROCESSED / "xgb_daily_metrics.json"
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved: {out_metrics}")


    # 7) Save forecast (3 columns)
    forecast_df = pd.DataFrame({
        "date": test.index,          # index is date
        "y_true": y_test.values,
        "y_pred": pred,
    })
    out_forecast = OUT_DIR_PROCESSED / "xgb_daily_forecast.csv"
    forecast_df.to_csv(out_forecast, index=False)
    print(f"✅ Saved: {out_forecast}  shape={forecast_df.shape}")

    # 8) Feature importance (aligned to feature_cols)
    booster = model.get_booster()
    gain_map = booster.get_score(importance_type="gain")
    weight_map = booster.get_score(importance_type="weight")

    importance = pd.DataFrame({
        "feature": feature_cols,
        "gain_importance": [gain_map.get(f, 0.0) for f in feature_cols],
        "weight_importance": [weight_map.get(f, 0.0) for f in feature_cols],
    }).sort_values("gain_importance", ascending=False)

    out_imp = OUT_DIR_PROCESSED / "xgb_daily_feature_importance.csv"
    importance.to_csv(out_imp, index=False)
    print(f"✅ Saved: {out_imp}")

    # 9) Plot Actual vs Forecast
    out_plot = OUT_DIR_OUTPUTS / "xgb_daily_actual_vs_forecast.png"
    plot_forecast(train.index, train["y"], test.index, y_test.values, pred, out_plot)
    print(f"✅ Saved: {out_plot}")


if __name__ == "__main__":
    main()
