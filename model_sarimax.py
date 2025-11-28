
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from pathlib import Path
import json

# Load merged dataset
df = pd.read_csv("./Data/processed/nyiso_weather_merged.csv", parse_dates=["datetime"])
df = df.sort_values("datetime")

# Downsample to daily mean for faster modeling
daily = df.resample("D", on="datetime").mean(numeric_only=True)
daily = daily.dropna(subset=["load_mw"])


result = adfuller(daily["load_mw"])
print(f"ADF Statistic: {result[0]:.3f}")
print(f"p-value: {result[1]:.3f}")


# Train–Test Split
train = daily[daily.index < "2024-01-01"]
test  = daily[daily.index >= "2024-01-01"]
print(f"Train: {train.shape}, Test: {test.shape}")


# Build SARIMA Model
model = SARIMAX(train["load_mw"],
                order=(2,1,2),
                seasonal_order=(1,1,1,7),
                enforce_stationarity=False,
                enforce_invertibility=False)

fit = model.fit(disp=False)
print(fit.summary())




# Forecast
pred = fit.forecast(steps=len(test))
test["forecast"] = pred.values

# Evaluate
mae = mean_absolute_error(test["load_mw"], test["forecast"])
rmse = np.sqrt(mean_squared_error(test["load_mw"], test["forecast"]))
print(f"MAE = {mae:.2f}, RMSE = {rmse:.2f}")

# Plot
plt.figure(figsize=(12,5))
plt.plot(train.index, train["load_mw"], label="Train")
plt.plot(test.index, test["load_mw"], label="Actual")
plt.plot(test.index, test["forecast"], label="Forecast", color="red")
plt.title("SARIMA Forecast vs Actual (Daily NY Load)")
plt.xlabel("Date"); plt.ylabel("Load (MW)")
plt.legend(); plt.tight_layout()
plt.show()




# avoid chained assignment warning
test = test.copy()

# 1) Save test forecast CSV
Path("./Data/processed").mkdir(parents=True, exist_ok=True)
out_forecast = Path("./Data/processed/sarima_daily_forecast.csv")
test_out = test[["load_mw", "forecast"]].reset_index().rename(
    columns={"index":"datetime","load_mw":"actual"}
)
test_out.to_csv(out_forecast, index=False)
print(f"✅ Saved: {out_forecast}  shape={test_out.shape}")

# 2) Compute MAPE and save metrics JSON
eps = 1e-6
mape = (np.abs((test["load_mw"] - test["forecast"]) /
               np.maximum(np.abs(test["load_mw"]), eps))).mean() * 100.0

metrics = {
    "model": "SARIMA(2,1,2)(1,1,1,7)",
    "resample": "D",
    "test_start": "2024-01-01",
    "mae": float(mae),
    "rmse": float(rmse),
    "mape": float(mape),
}
out_metrics = Path("./Data/processed/sarima_daily_metrics.json")
with open(out_metrics, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Saved: {out_metrics}")

# 3) Save the plot image too
Path("./Outputs").mkdir(parents=True, exist_ok=True)
plt.savefig("./Outputs/sarima_daily_actual_vs_forecast.png", dpi=150)
print("✅ Saved: ./Outputs/sarima_daily_actual_vs_forecast.png")

# 4) (Optional) Refit on ALL data & forecast next 30 days
model_full = SARIMAX(daily["load_mw"],
                     order=(2,1,2), seasonal_order=(1,1,1,7),
                     enforce_stationarity=False, enforce_invertibility=False)
fit_full = model_full.fit(disp=False)
next30 = fit_full.forecast(steps=30)
next30_df = next30.rename("forecast").reset_index().rename(columns={"index":"datetime"})
next_path = Path("./Data/processed/sarima_daily_next30.csv")
next30_df.to_csv(next_path, index=False)
print(f"✅ Saved: {next_path}  shape={next30_df.shape}")

