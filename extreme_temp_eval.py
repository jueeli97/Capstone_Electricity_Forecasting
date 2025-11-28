import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------- 1) Load base weather + load data --------
nyiso = pd.read_csv("./Data/processed/nyiso_weather_merged.csv")
nyiso["datetime"] = pd.to_datetime(nyiso["datetime"])

# Keep only 2024 and make it daily
nyiso_2024 = nyiso[nyiso["datetime"].dt.year == 2024].copy()
nyiso_daily = (
    nyiso_2024
    .set_index("datetime")
    .resample("D")
    .agg({
        "load_mw": "mean",
        "temp_f": "mean",
        "humidity": "mean",
        "wind_speed": "mean",
        "precip": "sum"
    })
    .reset_index()
)

# -------- 2) Load all model forecasts --------
sarima = pd.read_csv("./Data/processed/sarima_daily_forecast.csv")
prophet = pd.read_csv("./Data/processed/prophet_daily_forecast.csv")
xgb = pd.read_csv("./Data/processed/xgb_daily_forecast.csv")
lstm = pd.read_csv("./Data/processed/lstm_daily_forecast.csv")

sarima["datetime"] = pd.to_datetime(sarima["datetime"])
prophet["datetime"] = pd.to_datetime(prophet["datetime"])
xgb["date"] = pd.to_datetime(xgb["date"])
lstm["date"] = pd.to_datetime(lstm["date"])

# Standardize column names
sarima = sarima.rename(columns={"forecast": "sarima_pred"})
prophet = prophet.rename(columns={"forecast": "prophet_pred"})
xgb = xgb.rename(columns={"y_true": "actual", "y_pred": "xgb_pred", "date": "datetime"})
lstm = lstm.rename(columns={"y_true": "actual", "y_pred": "lstm_pred", "date": "datetime"})

# -------- 3) Merge predictions together (daily 2024) --------
df = sarima[["datetime", "actual", "sarima_pred"]].merge(
    prophet[["datetime", "prophet_pred"]], on="datetime", how="outer"
)
df = df.merge(xgb[["datetime", "xgb_pred"]], on="datetime", how="outer")
df = df.merge(lstm[["datetime", "lstm_pred"]], on="datetime", how="left")

# Add daily temperature
df = df.merge(
    nyiso_daily[["datetime", "temp_f"]],
    on="datetime",
    how="left"
)

# Restrict to 2024 only
df = df[df["datetime"].dt.year == 2024].copy()

df.to_csv("merged_predictions_2024.csv", index=False)
print(df.head())
print(df.tail())
print(df.isna().sum())







# -------- 4) Define temperature categories --------
hot_thresh = df["temp_f"].quantile(0.90)   # top 10%
cold_thresh = df["temp_f"].quantile(0.10)  # bottom 10%

print("Hot threshold:", hot_thresh)
print("Cold threshold:", cold_thresh)

df["temp_category"] = np.where(
    df["temp_f"] >= hot_thresh, "hot",
    np.where(df["temp_f"] <= cold_thresh, "cold", "normal")
)

print(df["temp_category"].value_counts())

# -------- 5) Helper to compute errors --------
# def compute_errors(sub, pred_col):
#     sub = sub[~sub[pred_col].isna()]
#     if sub.empty:
#         return None
#     mae = mean_absolute_error(sub["actual"], sub[pred_col])
#     rmse = mean_squared_error(sub["actual"], sub[pred_col], squared=False)
#     mape = (np.abs(sub["actual"] - sub[pred_col]) / sub["actual"]).mean() * 100
#     return mae, rmse, mape, len(sub)



def compute_errors(sub, pred_col):
    # Drop rows where prediction is missing
    sub = sub[~sub[pred_col].isna()].copy()
    if sub.empty:
        return None

    mae = mean_absolute_error(sub["actual"], sub[pred_col])

    # Older sklearn: no 'squared' arg → compute RMSE manually
    mse = mean_squared_error(sub["actual"], sub[pred_col])
    rmse = mse ** 0.5

    mape = (np.abs(sub["actual"] - sub[pred_col]) / sub["actual"]).mean() * 100

    return mae, rmse, mape, len(sub)

results = {}
for cat in ["cold", "normal", "hot"]:
    subset = df[df["temp_category"] == cat]
    results[cat] = {
        "SARIMA": compute_errors(subset, "sarima_pred"),
        "Prophet": compute_errors(subset, "prophet_pred"),
        "XGB": compute_errors(subset, "xgb_pred"),
        "LSTM": compute_errors(subset, "lstm_pred"),
    }

print(results)
