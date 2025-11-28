import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("./Data/processed/nyiso_weather_merged.csv", parse_dates=["datetime"])
# print(df.shape)
# print(df.head())
# print(df.isna().sum())



# Overall Demand Trend
# plt.figure(figsize=(12,5))
# plt.plot(df["datetime"], df["load_mw"], linewidth=0.5)
# plt.title("Hourly Electricity Demand in NY (2019–2024)")
# plt.xlabel("Date"); plt.ylabel("Load (MW)")
# plt.show()


# Seasonal Pattern (Monthly)
# df["month"] = df["datetime"].dt.month
# monthly = df.groupby("month")["load_mw"].mean()
# monthly.plot(kind="bar", title="Average Monthly Demand")

# plt.xlabel("Month")
# plt.ylabel("Average Load (MW)")
# plt.tight_layout()
# plt.show()


# Weather Correlations
# sns.heatmap(df[["load_mw","temp_f","humidity","wind_speed","precip"]].corr(),
#             annot=True, cmap="coolwarm")

# plt.title("Correlation between Demand and Weather Variables")
# plt.tight_layout()
# plt.show()  



#  Scatter Plot: Temperature vs Demand
sns.scatterplot(x="temp_f", y="load_mw", data=df.sample(5000))
plt.title("Temperature vs Electricity Demand")
plt.xlabel("Temperature (°F)")
plt.ylabel("Load (MW)")
plt.tight_layout()
plt.show() 