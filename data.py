# data.py
import pandas as pd, glob
from pathlib import Path

# 1) Read all NYISO annual CSVs you downloaded
files = glob.glob("./Data/nyiso-loads-*.csv")   # adjust if your path differs
if not files:
    raise FileNotFoundError("No NYISO CSVs matched ./Data/nyiso-loads-*.csv")

dfs = []
for f in files:
    df = pd.read_csv(f)
    # Keep only hour columns + Y/M/D
    hour_cols = [c for c in df.columns if c.lower().startswith("hr")]
    base = df[["Year","Month","Day"] + hour_cols].copy()

    # 2) Wide -> long: one row per hour
    long = base.melt(id_vars=["Year","Month","Day"],
                     value_vars=hour_cols,
                     var_name="hour_label",
                     value_name="load_mw")

    # Drop empty loads
    long = long.dropna(subset=["load_mw"])

    # 3) Extract hour number (Hr1..Hr24/Hr25)
    long["hour_num"] = long["hour_label"].str.extract(r"(\d+)").astype(int)

    # 4) Handle DST "Hr25": simplest is to drop it for modeling
    long = long[long["hour_num"] <= 24]

    # 5) Build datetime (treat Hr1 as 01:00 → subtract 1 to get 00:00 index)
    dt = pd.to_datetime(dict(year=long["Year"],
                             month=long["Month"],
                             day=long["Day"]))
    long["datetime"] = dt + pd.to_timedelta(long["hour_num"] - 1, unit="h")

    # Keep only what we need
    dfs.append(long[["datetime","load_mw"]])

# 6) Combine all years and sort
nyiso = pd.concat(dfs, ignore_index=True).sort_values("datetime")
nyiso["load_mw"] = pd.to_numeric(nyiso["load_mw"], errors="coerce")
nyiso = nyiso.dropna(subset=["load_mw"])

# 7) Save processed NYISO
out = Path("./Data/processed/nyiso_statewide_2019_2024.csv")
out.parent.mkdir(parents=True, exist_ok=True)
nyiso.to_csv(out, index=False)
print(f"✅ Saved: {out}  | rows={len(nyiso):,}")
