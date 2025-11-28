# noaa_merge.py
import pandas as pd
from pathlib import Path
import re

# ========= PATHS (edit if needed) =========
NOAA_DIR = Path("./Data/NOAA_Data")                      # folder containing your 5 station CSVs
NYISO_FILE = Path("./Data/processed/nyiso_statewide_2019_2024.csv")
OUT_NOAA_AVG = Path("./Data/processed/noaa_statewide_avg.csv")
OUT_MERGED   = Path("./Data/processed/nyiso_weather_merged.csv")
# ==========================================

# Candidate name maps (we'll pick the first that exists)
DATE_CANDS   = ["DATE", "Date", "date", "time", "datetime", "ValidTime"]  # LCD usually "DATE"
TEMP_CANDS   = ["HourlyDryBulbTemperature","HourlyDryBulbTemperature (F)","DryBulbTemperature","temp","TMP"]
RH_CANDS     = ["HourlyRelativeHumidity","RelativeHumidity","humidity","RH"]
WSPD_CANDS   = ["HourlyWindSpeed","WindSpeed","wspd","WND_SPD","wind_speed"]
PRCP_CANDS   = ["HourlyPrecipitation","HourlyPrecipitation (in)","Precipitation","precip","PRCP"]

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def coerce_numeric(series):
    # strip everything except digits, dot, minus
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

def load_noaa_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Try to locate columns
    c_date = pick_col(df, DATE_CANDS)
    c_tmp  = pick_col(df, TEMP_CANDS)
    c_rh   = pick_col(df, RH_CANDS)
    c_wspd = pick_col(df, WSPD_CANDS)
    c_prcp = pick_col(df, PRCP_CANDS)

    if not c_date:
        raise ValueError(f"{path.name}: Could not find a datetime column among {DATE_CANDS}")

    keep_cols = [c_date, c_tmp, c_rh, c_wspd, c_prcp]
    # Drop Nones safely
    keep_cols = [c for c in keep_cols if c is not None]
    df = df[keep_cols].copy()

    # Rename to standard names
    rename_map = {}
    if c_tmp:  rename_map[c_tmp]  = "temp_f"
    if c_rh:   rename_map[c_rh]   = "humidity"
    if c_wspd: rename_map[c_wspd] = "wind_speed"
    if c_prcp: rename_map[c_prcp] = "precip"
    rename_map[c_date]            = "datetime"
    df = df.rename(columns=rename_map)

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Coerce numerics
    for c in ["temp_f","humidity","wind_speed","precip"]:
        if c in df.columns:
            df[c] = coerce_numeric(df[c])

    # Some feeds store multiple records per hour → resample to hourly mean
    df = (df.set_index("datetime")
            .resample("H")
            .mean(numeric_only=True)
            .reset_index())

    # Optional cap/clean edges
    if "humidity" in df.columns:
        df["humidity"] = df["humidity"].clip(lower=0, upper=100)

    return df

def main():
    files = sorted(NOAA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No NOAA CSVs found in {NOAA_DIR.resolve()}")

    print(f"Found {len(files)} NOAA station file(s):")
    for f in files: print(" -", f.name)

    station_dfs = []
    for f in files:
        try:
            one = load_noaa_file(f)
            print(f"  ✅ Loaded {f.name} → rows={len(one):,}, cols={list(one.columns)}")
            station_dfs.append(one)
        except Exception as e:
            print(f"  ⚠️ Skipped {f.name}: {e}")

    if not station_dfs:
        raise RuntimeError("No NOAA files could be parsed. Check column names or CSV format.")

    # Average across stations by hour
    weather_all = (pd.concat(station_dfs, ignore_index=True)
                     .set_index("datetime")
                     .groupby("datetime")
                     .mean(numeric_only=True)
                     .reset_index()
                  )

    OUT_NOAA_AVG.parent.mkdir(parents=True, exist_ok=True)
    weather_all.to_csv(OUT_NOAA_AVG, index=False)
    print(f"✅ Saved statewide hourly weather average → {OUT_NOAA_AVG}  rows={len(weather_all):,}")

    # Merge with NYISO
    if not NYISO_FILE.exists():
        raise FileNotFoundError(f"NYISO file not found at {NYISO_FILE.resolve()}.\n"
                                f"Expected the processed file created earlier (datetime, load_mw).")
    nyiso = pd.read_csv(NYISO_FILE, parse_dates=["datetime"])

    merged = pd.merge(nyiso, weather_all, on="datetime", how="inner")
    merged = merged.sort_values("datetime")

    merged.to_csv(OUT_MERGED, index=False)
    print(f"✅ Saved merged NYISO + NOAA → {OUT_MERGED}  shape={merged.shape}")

    # Quick quality checks
    missing = merged.isna().sum()
    print("Nulls per column after merge:\n", missing)

    # Optional: print quick correlations
    num_cols = [c for c in ["load_mw","temp_f","humidity","wind_speed","precip"] if c in merged.columns]
    if len(num_cols) >= 2:
        print("\nCorrelation (sample):")
        print(merged[num_cols].corr().round(3))

if __name__ == "__main__":
    main()
