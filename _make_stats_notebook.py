"""Generate Statistical_analysis.ipynb."""
import json
import os
import uuid

target_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(target_dir, "Statistical_analysis.ipynb")


def _src(s):
    parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def md(text):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": _src(text),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": _src(text),
    }


cells = []

cells.append(md("""# Statistical Analysis — Household Energy Footprint

Loads the pre-computed `var_for_stat_analysis_<n>COICOP.pkl` produced by `Household_EF.py` / `Household_EF.ipynb` (Cell 12). Run that script first if the file does not exist.

The pickle holds one row per SHS 2019 survey household with:
- **Identifier**: `CaseID`
- **Weights / scale**: `WeightD`, `OECD_scale`, `OECD_cap`
- **Income**: `HH_TotInc`, `Inc_hh_dec`, `Inc_OECDcap_dec`
- **Socio-demographic**: `Prov`, `HHType6`, `RP_AgeGrp`, `RP_Gender`, `RP_MarStat`, `RP_Educ`, `DwelTyp`, `Tenure`, `NumBedr`, `SecResYN`, `OthPropYN`, `InternetYN`, `VehicleYN`, `HH_MajIncSrc`
- **EF totals**: `Total_aggr_EF_hh_wght`, `avg_EF_per_hh`, `avg_EF_perOECD_cap`
- **EF by COICOP**: `avg_EF_per_hh__<safe_name>` and `avg_EF_per_cap__<safe_name>`

The companion `..._labels.json` maps sanitized column names back to original COICOP labels."""))

cells.append(md("""## Setup — load the data

Set `COICOP_LEVEL` to 12, 32, or 75 depending on which aggregation you saved in Cell 12 of `Household_EF`."""))

cells.append(code("""import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COICOP_LEVEL = 12   # 75 / 32 / 12 — must match what was saved by Household_EF Cell 12

_HERE = os.path.abspath(os.getcwd())
_PKL_PATH = os.path.join(_HERE, f"var_for_stat_analysis_{COICOP_LEVEL}COICOP.pkl")
_MAP_PATH = os.path.join(_HERE, f"var_for_stat_analysis_{COICOP_LEVEL}COICOP_labels.json")

df = pd.read_pickle(_PKL_PATH)
with open(_MAP_PATH, "r", encoding="utf-8") as f:
    coicop_label_map = json.load(f)

print(f"Loaded {_PKL_PATH}")
print(f"df shape: {df.shape}")
print(f"COICOP categories: {COICOP_LEVEL}")
print(f"Label map entries: {len(coicop_label_map)}")"""))

cells.append(md("""## Data overview

Quick look at column groups, dtypes, and missingness."""))

cells.append(code("""# Column groups for navigation
ID_COLS    = ["CaseID"]
WEIGHT_COLS = ["WeightD", "OECD_scale", "OECD_cap"]
INCOME_COLS = ["HH_TotInc", "Inc_hh_dec", "Inc_OECDcap_dec"]
SOCIO_COLS  = ["Prov", "HHType6", "RP_AgeGrp", "RP_Gender", "RP_MarStat", "RP_Educ",
               "DwelTyp", "Tenure", "NumBedr", "SecResYN", "OthPropYN",
               "InternetYN", "VehicleYN", "HH_MajIncSrc"]
EF_TOTAL_COLS = ["Total_aggr_EF_hh_wght", "avg_EF_per_hh", "avg_EF_perOECD_cap"]
EF_PER_HH_COLS  = [c for c in df.columns if c.startswith("avg_EF_per_hh__")]
EF_PER_CAP_COLS = [c for c in df.columns if c.startswith("avg_EF_per_cap__")]

print(f"ID:                 {len(ID_COLS):>3} cols")
print(f"Weights / scale:    {len(WEIGHT_COLS):>3} cols")
print(f"Income:             {len(INCOME_COLS):>3} cols")
print(f"Socio-demographic:  {len(SOCIO_COLS):>3} cols")
print(f"EF totals:          {len(EF_TOTAL_COLS):>3} cols")
print(f"EF per hh x COICOP: {len(EF_PER_HH_COLS):>3} cols")
print(f"EF per cap x COICOP:{len(EF_PER_CAP_COLS):>3} cols")
print()
print("dtypes (one per group):")
print(df[ID_COLS + WEIGHT_COLS + INCOME_COLS + SOCIO_COLS + EF_TOTAL_COLS].dtypes)"""))

cells.append(code("""# Missing-value check
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if missing.empty:
    print("No missing values.")
else:
    print(f"{len(missing)} columns with missing values:")
    print(missing)"""))

cells.append(code("""df.head()"""))

cells.append(md("""## Summary statistics

Population-level (WeightD-weighted) and OECD-cap-weighted summaries for the key EF totals."""))

cells.append(code('''def weighted_describe(series, weights):
    """Return a dict with weighted mean, median, std, min, max."""
    s = pd.Series(series).astype(float)
    w = pd.Series(weights).astype(float).reindex(s.index).fillna(0)
    mask = s.notna() & (w > 0)
    s = s[mask]; w = w[mask]
    if s.empty:
        return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan,
                "min": np.nan, "max": np.nan}
    mean = (s * w).sum() / w.sum()
    var  = (w * (s - mean) ** 2).sum() / w.sum()
    # Weighted median
    order = np.argsort(s.values)
    cw = np.cumsum(w.values[order])
    cutoff = w.sum() / 2.0
    median = s.values[order][np.searchsorted(cw, cutoff)]
    return {
        "n": int(mask.sum()),
        "mean": mean, "median": median, "std": np.sqrt(var),
        "min": s.min(), "max": s.max(),
    }


print("Weighted summary (WeightD = households)")
print("-" * 60)
for col in EF_TOTAL_COLS + ["HH_TotInc"]:
    s = weighted_describe(df[col], df["WeightD"])
    print(f"{col:30s} n={s['n']:>5}  mean={s['mean']:>12.4f}  "
          f"median={s['median']:>12.4f}  std={s['std']:>12.4f}")

print()
print("OECD-cap-weighted summary (per OECD-equivalent adult)")
print("-" * 60)
for col in EF_TOTAL_COLS + ["HH_TotInc"]:
    s = weighted_describe(df[col], df["OECD_cap"])
    print(f"{col:30s} n={s['n']:>5}  mean={s['mean']:>12.4f}  "
          f"median={s['median']:>12.4f}  std={s['std']:>12.4f}")'''))

cells.append(md("""## Distribution of `avg_EF_per_hh`

WeightD-weighted histogram of total household EF (TJ per household)."""))

cells.append(code("""fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["avg_EF_per_hh"], bins=60, weights=df["WeightD"],
        color="#56B4E9", edgecolor="white")
ax.set_xlabel("Total avg_EF_per_hh (TJ per household)")
ax.set_ylabel("WeightD-weighted count of households")
ax.set_title(f"Distribution of total household EF (COICOP-{COICOP_LEVEL})")
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()"""))

cells.append(md("""## Example — look up a COICOP label

The pickle stores sanitized column names. Use `coicop_label_map` to recover the original COICOP labels:"""))

cells.append(code("""# Show the first 5 EF-per-hh columns and what they map to
for col in EF_PER_HH_COLS[:5]:
    print(f"{col}  -> {coicop_label_map.get(col, '?')}")"""))

cells.append(md("""## Starting points for your analysis

Below are stub cells you can extend. Add cells freely.

- **OLS regression** of `avg_EF_per_hh` on socio-demographics (using `statsmodels`).
- **Inequality decomposition** by income decile, household type, etc.
- **Concentration curves** / **Gini** of EF across deciles.
- **Decomposition** of EF differences (e.g., between deciles) into category contributions."""))

cells.append(code("""# Example: weighted OLS - total EF per hh against household income (uncomment to run)
# import statsmodels.api as sm
# X = sm.add_constant(df[["HH_TotInc"]])
# model = sm.WLS(df["avg_EF_per_hh"], X, weights=df["WeightD"]).fit()
# print(model.summary())"""))


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(target_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Wrote {target_path} with {len(cells)} cells")