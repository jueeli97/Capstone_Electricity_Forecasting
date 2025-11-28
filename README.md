# 📈 Electricity Demand Forecasting for New York State (NYISO)  
### **Time-Series & Machine Learning Forecasting with Weather + Seasonal Insights**

This capstone project applies statistical, machine learning, and deep learning techniques to forecast **New York State electricity load** using historical NYISO data (2019–2024), integrated with NOAA weather variables.  
It includes a full modeling pipeline, extreme-temperature evaluation, and a fully interactive Power BI dashboard.

---

## 🌟 Key Objectives

- Forecast daily electricity load for 2024 using multiple models  
- Understand weather impacts on energy demand  
- Compare statistical vs ML performance  
- Analyze extreme temperature effects (cold, normal, hot regimes)  
- Build an interactive forecasting & seasonality dashboard (Power BI)

---

## 🧰 Tech Stack

| Category | Tools / Libraries |
|---------|-------------------|
| Programming | Python (Pandas, NumPy, Scikit-learn, Matplotlib) |
| ML Models | XGBoost, LSTM (Keras/TensorFlow), Prophet |
| Statistical Model | SARIMA (statsmodels) |
| Dashboard | Power BI |
| Data | NYISO Load Data, NOAA Weather Data |
| Forecasting Methods | Multi-step forecasting (7-day & 30-day) |

---


## 📝 Project Overview

This project predicts electricity demand using multiple modeling approaches and compares their performance on the 2024 test set.  
Models incorporate:

- Temperature (°F)  
- Humidity  
- Wind speed  
- Precipitation  
- Lag features (1-day, 7-day, 14-day, 28-day)  
- Rolling means (7, 14, 28 days)  
- Calendar features (month, weekend, day of week)

The goal is to provide highly accurate day-ahead and multi-step forecasts to support grid planning and operational reliability.

---

## 📊 Dashboard Preview

### 🔹 Electricity Load Forecasting Dashboard (Power BI)

Includes:

- Actual vs Predicted Load (SARIMA, Prophet, XGBoost, LSTM)  
- Temperature vs Load non-linear relationship (U-shaped curve)
- XGBoost Feature Importance
- Monthly & Weekly Seasonality Patterns  
- Filters for year, month, temperature bins

*(Dashboard visuals included in the uploaded PDF reports)*

---

## 🤖 Models Used

| Model | Type | Notes |
|-------|------|-------|
| **SARIMA** | Statistical | Baseline model; captures seasonal cycles; struggles with extreme temps |
| **Prophet** | Additive Time-Series | Smooth trend + seasonality; less reactive to sharp spikes |
| **XGBoost** | ML Regression | Best performer; strong dynamic response to weather & lag features |
| **LSTM** | Deep Learning | Learns temporal sequence patterns; slight overestimation early in year |

---

## 📈 Performance Comparison (Full-Year 2024)

| Model | MAE (MW) | RMSE (MW) | MAPE |
|-------|----------|-----------|-------|
| SARIMA | 2068.12 | 2744.45 | 11.15% |
| Prophet | 1396.45 | 1825.36 | 7.63% |
| **XGBoost** | **363.15** | **474.87** | **2.06%** |
| LSTM | 621.35 | 868.04 | 3.52% |

**XGBoost achieved the best overall predictive accuracy.**

---

## 🔥 Extreme Temperature Performance  
Models evaluated across:

- **Cold days (≤ 10th percentile temp)**
- **Normal days**
- **Hot days (≥ 90th percentile temp)**

### Summary:

- **XGBoost remains the most stable during cold & hot extremes**
- SARIMA & Prophet error values spike dramatically during extreme weather  
- LSTM performs well but slightly behind XGBoost  

---

## 📉 Multi-Step Forecasting

### ✔ 7-Day Ahead Forecast (XGBoost)  
- Captures short-term trends accurately  
- Stable daily transition predictions  

### ✔ 30-Day Ahead Forecast (XGBoost & Prophet)  
- XGBoost: More responsive to nonlinear weather + lag interactions  
- Prophet: Smoother long-term seasonal curve  

---

## 💡 Key Findings

- **Temperature is the strongest driver** of electricity demand (U-shaped pattern)  
- **Lag features** greatly improve model accuracy  
- **XGBoost provides the best balance of precision and interpretability**  
- **Seasonality is significant**:
  - Summer cooling peaks (July–August)  
  - Winter heating peaks (Dec–Jan)  
  - Lowest demand in spring  
- **Extreme-weather evaluation shows ML models outperform statistical models**

---
## 📥 Data Access

Large raw NOAA files is not stored in this repo due to GitHub’s 100 MB limit.  

---

## 🎓 Author

**Jueeli Rajesh Sawant**  
MS in Information Technology & Analytics  
Rochester Institute of Technology  
Email: js4023@rit.edu  

---





