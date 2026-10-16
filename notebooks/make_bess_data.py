"""
make_bess_data.py

Generates 7 days of synthetic operating data for a battery energy storage
system (BESS) and saves it to ../data/bess_ops.csv.

- 20 racks, one record per rack per minute (about 200,000 rows)
- Daily schedule: charge 10:00-14:00, discharge 17:00-21:00
- Several faults are injected near the end of the script. If you want to
  find them yourself, avoid reading the fault-injection section first.

Run from the notebooks/ folder:  python make_bess_data.py
"""
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_RACKS = 20
START = "2026-07-01 00:00"
DAYS = 7
CAPACITY_AH = 280 * 2          # rack capacity (Ah): two parallel strings of 280 Ah cells
N_SERIES = 240                 # cells in series per rack

time = pd.date_range(START, periods=DAYS * 24 * 60, freq="min")
n = len(time)
hour = time.hour + time.minute / 60
day = (time - time[0]).days

# ---------- Site-wide schedule: charge midday, discharge in the evening ----------
current = np.zeros(n)
current[(hour >= 10) & (hour < 14)] = -140      # charging (negative = charge)
current[(hour >= 17) & (hour < 21)] = 140       # discharging
ambient = 29 + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.3, n)

def soc_curve(cur):
    soc = np.empty(n)
    s = 15.0
    for i in range(n):
        s -= cur[i] / 60 / CAPACITY_AH * 100
        s = min(max(s, 5.0), 98.0)
        soc[i] = s
    return soc

def ocv_lfp(soc):
    # Simplified flat LFP open-circuit voltage curve (per cell, V)
    return 3.20 + 0.10 * (soc / 100) + 0.15 * np.exp((soc - 100) / 4) - 0.25 * np.exp(-soc / 4)

frames = []
for r in range(1, N_RACKS + 1):
    rack = f"R{r:02d}"
    cur = current + rng.normal(0, 1.0, n)
    soc = soc_curve(cur) + rng.normal(0, 0.2, n)
    r_int = 0.0008 * (1 + rng.normal(0, 0.03))          # cell internal resistance (ohm)
    i_cell = cur / 2
    cell_avg = ocv_lfp(soc) - i_cell * r_int
    spread = 0.008 + rng.normal(0, 0.001, n).clip(-0.004, 0.004)
    max_cell = cell_avg + spread * 0.5
    min_cell = cell_avg - spread * 0.5
    heat = 3.0 * (np.abs(cur) / 140) ** 2
    rack_offset = rng.normal(0, 0.4)
    temp = (ambient - 4 + rack_offset
            + pd.Series(heat).rolling(60, min_periods=1).mean().to_numpy() * 2.5
            + rng.normal(0, 0.25, n))
    frames.append(pd.DataFrame({
        "timestamp": time,
        "rack_id": rack,
        "soc_pct": soc,
        "current_A": cur,
        "rack_voltage_V": cell_avg * N_SERIES,
        "max_cell_V": max_cell,
        "min_cell_V": min_cell,
        "max_temp_C": temp,
        "ambient_C": ambient,
    }))

ops = pd.concat(frames, ignore_index=True)

# ============ Fault injection (skip this section if you want to find the faults yourself) ============
def mask(rack, t0=None, t1=None):
    m = ops["rack_id"] == rack
    if t0: m &= ops["timestamp"] >= t0
    if t1: m &= ops["timestamp"] < t1
    return m

# Fault 1: gradual cooling failure
m = mask("R07", "2026-07-04 00:00")
ramp = (ops.loc[m, "timestamp"] - pd.Timestamp("2026-07-04")).dt.total_seconds() / 86400
ops.loc[m, "max_temp_C"] += np.minimum(ramp * 2.5, 8.0)

# Fault 2: growing cell voltage imbalance within a rack
m = mask("R13", "2026-07-02 00:00")
ramp = (ops.loc[m, "timestamp"] - pd.Timestamp("2026-07-02")).dt.total_seconds() / 86400
ops.loc[m, "min_cell_V"] -= 0.012 * ramp

# Fault 3: stuck temperature sensor
m = mask("R16", "2026-07-03 08:00", "2026-07-03 20:00")
ops.loc[m, "max_temp_C"] = ops.loc[m, "max_temp_C"].iloc[0]

# Fault 4: communication loss (missing data)
m = mask("R02", "2026-07-05 13:00", "2026-07-05 16:00")
ops.loc[m, ["soc_pct", "current_A", "rack_voltage_V", "max_cell_V", "min_cell_V", "max_temp_C"]] = np.nan

# Fault 5: intermittent sensor spikes
idx = rng.choice(ops.index[ops["rack_id"] == "R11"], size=6, replace=False)
ops.loc[idx, "max_temp_C"] = rng.choice([-40.0, 120.0, 85.0], size=6)
# =================================================================

ops = ops.sort_values(["timestamp", "rack_id"]).reset_index(drop=True)
num_cols = ops.select_dtypes("number").columns
ops[num_cols] = ops[num_cols].round(4)
os.makedirs("../data", exist_ok=True)   # create the data folder if it does not exist
ops.to_csv("../data/bess_ops.csv", index=False)
print(f"Generated {len(ops):,} rows and saved them to ../data/bess_ops.csv")
