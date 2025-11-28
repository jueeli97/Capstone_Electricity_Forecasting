import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. Load the base dataset (weather + load)
# ----------------------------------------------------------
df = pd.read_csv("Data/processed/nyiso_weather_merged.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

# Daily aggregation (same as in model_xgboost.py)
daily = (
    df.set_index("datetime")
      .resample("D")
      .agg({
          "load_mw": "mean",
          "temp_f": "mean",
          "humidity": "mean",
          "wind_speed": "mean",
          "precip": "sum"
      })
      .reset_index()
      .sort_values("datetime")
)

# Basic sanity check
assert len(daily) >= 365, "Daily series seems too short."

# ----------------------------------------------------------
# 2. Prepare history arrays
# ----------------------------------------------------------
# Full history of daily load (y)
y_hist = daily["load_mw"].tolist()

# We'll use average of last 7 days as proxy for future weather
last7 = daily.tail(7)
avg_temp = float(last7["temp_f"].mean())
avg_humidity = float(last7["humidity"].mean())
avg_wind = float(last7["wind_speed"].mean())
avg_precip = float(last7["precip"].mean())

last_date = daily["datetime"].max()

print("Last date in history:", last_date)
print("Avg temp last 7 days:", avg_temp)

# ----------------------------------------------------------
# 3. Load trained XGBoost model
# ----------------------------------------------------------
xgb_model = joblib.load("Data/processed/xgb_daily_model.pkl")

feature_cols = [
    "dayofweek", "month", "is_weekend",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "mean_7", "mean_14", "mean_28",
    "temp_f", "humidity", "wind_speed", "precip",
]

# ----------------------------------------------------------
# 4. Forecast function using the same 14 features
# ----------------------------------------------------------
def forecast_future(y_history, last_date, days):
    """
    y_history: list of past daily load values
    last_date: last date in the historical data (Timestamp)
    days: number of future days to forecast
    """
    forecasts = []
    y_hist_local = list(y_history)  # copy so we don't mutate original

    for i in range(1, days + 1):
        dt = last_date + pd.Timedelta(days=i)

        # Need at least 28 days of history for lags
        if len(y_hist_local) < 28:
            raise ValueError("Not enough history for lag_28 calculation.")

        # Lags: last 1, 7, 14, 28 days
        lag_1 = y_hist_local[-1]
        lag_7 = y_hist_local[-7]
        lag_14 = y_hist_local[-14]
        lag_28 = y_hist_local[-28]

        # Rolling means (shifted) exactly as in training:
        # mean_7, mean_14, mean_28 of *previous* days
        mean_7 = float(np.mean(y_hist_local[-7:]))
        mean_14 = float(np.mean(y_hist_local[-14:]))
        mean_28 = float(np.mean(y_hist_local[-28:]))

        # Calendar
        dayofweek = dt.dayofweek
        month = dt.month
        is_weekend = 1 if dayofweek >= 5 else 0

        # Build feature row
        X_row = pd.DataFrame([{
            "dayofweek": dayofweek,
            "month": month,
            "is_weekend": is_weekend,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "lag_28": lag_28,
            "mean_7": mean_7,
            "mean_14": mean_14,
            "mean_28": mean_28,
            "temp_f": avg_temp,
            "humidity": avg_humidity,
            "wind_speed": avg_wind,
            "precip": avg_precip,
        }])[feature_cols]  # ensure correct order

        # Predict
        pred = float(xgb_model.predict(X_row)[0])

        # Append prediction to history for next step
        y_hist_local.append(pred)
        forecasts.append((dt, pred))

    return pd.DataFrame(forecasts, columns=["datetime", "forecast"])

# ----------------------------------------------------------
# 5. Generate 7-day and 30-day forecasts
# ----------------------------------------------------------
forecast_7 = forecast_future(y_hist, last_date, 7)
forecast_30 = forecast_future(y_hist, last_date, 30)

forecast_7.to_csv("xgb_next7.csv", index=False)
forecast_30.to_csv("xgb_next30.csv", index=False)

print("Saved xgb_next7.csv and xgb_next30.csv")

# ----------------------------------------------------------
# 6. Plot 7-day and 30-day forecasts vs last 60 days actual
# ----------------------------------------------------------
last_60 = daily.tail(60)

# 7-day plot
plt.figure(figsize=(12, 6))
plt.plot(last_60["datetime"], last_60["load_mw"], label="Actual (Last 60 days)", linewidth=2)
plt.plot(forecast_7["datetime"], forecast_7["forecast"], label="XGBoost Forecast (Next 7 days)", linewidth=3)
plt.title("7-Day Ahead Load Forecast — XGBoost")
plt.xlabel("Date")
plt.ylabel("Load (MW)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("xgb_forecast_7day.png", dpi=150)
plt.show()

# 30-day plot
plt.figure(figsize=(12, 6))
plt.plot(last_60["datetime"], last_60["load_mw"], label="Actual (Last 60 days)", linewidth=2)
plt.plot(forecast_30["datetime"], forecast_30["forecast"], label="XGBoost Forecast (Next 30 days)", linewidth=3)
plt.title("30-Day Ahead Load Forecast — XGBoost")
plt.xlabel("Date")
plt.ylabel("Load (MW)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("xgb_forecast_30day.png", dpi=150)
plt.show()
