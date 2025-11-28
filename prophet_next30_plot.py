import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load actual daily series
# -----------------------------
df = pd.read_csv("Data/processed/nyiso_weather_merged.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

daily = (
    df.set_index("datetime")
      .resample("D")
      .agg({"load_mw": "mean"})
      .reset_index()
      .sort_values("datetime")
)

# last 60 days of actual load
last_60 = daily.tail(60)

# -----------------------------
# 2. Load Prophet next-30-day forecast
# -----------------------------
prophet_next30 = pd.read_csv("Data/processed/prophet_daily_next30.csv")
prophet_next30["datetime"] = pd.to_datetime(prophet_next30["datetime"])

# -----------------------------
# 3. Plot 30-day Prophet forecast
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(last_60["datetime"], last_60["load_mw"],
         label="Actual (Last 60 days)", linewidth=2)
plt.plot(prophet_next30["datetime"], prophet_next30["forecast"],
         label="Prophet Forecast (Next 30 days)", linewidth=3)

plt.title("30-Day Ahead Load Forecast — Prophet")
plt.xlabel("Date")
plt.ylabel("Load (MW)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("prophet_forecast_30day.png", dpi=150)
plt.show()
