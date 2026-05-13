"""Generate Household_EF.ipynb (no nbformat dep — stdlib json only)."""
import json
import uuid


def _src(s):
    # Notebook expects source as list of lines, each ending with \n
    # (last line may omit trailing \n; either is fine).
    lines = s.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


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

cells.append(md("""# Household Energy Footprint

This notebook computes per-household and per-capita energy footprints from SHS 2019 data combined with sectoral energy intensities from `Energy_Model.py`.

**Workflow:**

1. **Cell 1** — Load (and cache) outputs from `Energy_Model.py`.
2. **Cell 1b** — Load the raw SHS 2019 SAS file (all variables).
3. **Cell 1c** — OECD-modified equivalence scale per household.
4. **Cell 2** — COICOP aggregation matrices (32 / 12 cat) + aggregation switch.
5. **Cell 3** — Per-household spend shares at the chosen aggregation.
6. **Cell 4** — Per-household energy footprint by COICOP category.
7. **Cell 5** — Stacked bar chart: TJ / OECD-cap and TJ / household."""))

cells.append(md("""## Setup — imports and configuration

Override `DATA_DIR`, `FIGARO_FOLDER`, `COUNTRY_CODE`, `HH_CATEGORY`, or `FD_CATEGORIES` below. These values are forwarded to `Energy_Model.py` via `runpy.run_path(init_globals=...)`, so you do **not** need to edit `Energy_Model.py`.

A cache file (`_cache_household_ef_<hash>.pkl`) sits alongside the notebook; the hash is derived from the config, so each combination has its own cache and changes auto-invalidate."""))

cells.append(code(r"""import os
import pickle
import runpy
import hashlib
import json
import pandas as pd

# === Configuration ===
DATA_DIR = r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents\additional_projects\Canada_hh_energy\Data"
FIGARO_FOLDER = r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents - 403001-01 Utvikling og vedlikehold klimakost\General\Utslippsfaktorer\Kryssløpsanalyse\FIGARO data"
COUNTRY_CODE = "CA"
HH_CATEGORY = "P3_S14"
FD_CATEGORIES = ["P3_S13", "P3_S14", "P3_S15", "P51G", "P5M"]

_HERE = os.path.abspath(os.getcwd())
_ENERGY_MODEL_PATH = os.path.join(_HERE, "Energy_Model.py")
_ENERGY_MODEL_GLOBALS = {
    "DATA_DIR": DATA_DIR,
    "FIGARO_FOLDER": FIGARO_FOLDER,
    "COUNTRY_CODE": COUNTRY_CODE,
    "HH_CATEGORY": HH_CATEGORY,
    "FD_CATEGORIES": FD_CATEGORIES,
}
_config_hash = hashlib.md5(
    json.dumps(_ENERGY_MODEL_GLOBALS, sort_keys=True).encode()
).hexdigest()[:8]
_CACHE_PATH = os.path.join(_HERE, f"_cache_household_ef_{_config_hash}.pkl")
print(f"Cache file: {_CACHE_PATH}")"""))

cells.append(md("""## Cell 1 — Load Energy_Model results (cached)

Runs `Energy_Model.py` once, pickles the five outputs needed downstream, and on subsequent runs loads them instantly from the cache. The cache auto-invalidates when:

- Any of the Energy_Model source files (`Energy_Model.py`, `Figaro_economic_data.py`, etc.) is newer than the cache.
- Or the config in the Setup cell changes (different hash → different cache filename).

After loading, an `EF_tot_COICOP` column is added to `EF_COICOP_TJ_indir_dir` as `EF_indir_COICOP + EF_dir_COICOP`."""))

cells.append(code("""_SOURCE_FILES = [
    "Energy_Model.py",
    "Figaro_economic_data.py",
    "SHS_EDM_2019_data.py",
    "NACE_COICOP_mask.py",
    "RAS_balancing.py",
    "Indir_E_e.py",
]
_KEYS = [
    "e_COICOP_TJperMlnEur",
    "EF_COICOP_TJ_indir_dir",
    "SHS_2019_data_CAD_WGHT",
    "SHS_2019_data_Mln_EUR_WGHT",
    "NACE_to_COICOP_mask",
]


def _cache_is_fresh() -> bool:
    if not os.path.exists(_CACHE_PATH):
        return False
    cache_mtime = os.path.getmtime(_CACHE_PATH)
    for name in _SOURCE_FILES:
        path = os.path.join(_HERE, name)
        if os.path.exists(path) and os.path.getmtime(path) > cache_mtime:
            return False
    return True


if _cache_is_fresh():
    print("Loading cached Energy_Model outputs...")
    with open(_CACHE_PATH, "rb") as f:
        _data = pickle.load(f)
    print("Cache loaded.")
else:
    print("Cache missing or stale -- running Energy_Model.py (progress below)")
    print("-" * 70)
    _ns = runpy.run_path(
        _ENERGY_MODEL_PATH,
        run_name="__energy_model__",
        init_globals=_ENERGY_MODEL_GLOBALS,
    )
    print("-" * 70)
    _data = {k: _ns[k] for k in _KEYS}
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cached {len(_KEYS)} objects to disk for future runs.")
    del _ns

e_COICOP_TJperMlnEur       = _data["e_COICOP_TJperMlnEur"]
EF_COICOP_TJ_indir_dir     = _data["EF_COICOP_TJ_indir_dir"]
SHS_2019_data_CAD_WGHT     = _data["SHS_2019_data_CAD_WGHT"]
SHS_2019_data_Mln_EUR_WGHT = _data["SHS_2019_data_Mln_EUR_WGHT"]
NACE_to_COICOP_mask        = _data["NACE_to_COICOP_mask"]
del _data

EF_COICOP_TJ_indir_dir["EF_tot_COICOP"] = (
    EF_COICOP_TJ_indir_dir["EF_indir_COICOP"] + EF_COICOP_TJ_indir_dir["EF_dir_COICOP"]
)

print(
    f"Ready: e_COICOP_TJperMlnEur {e_COICOP_TJperMlnEur.shape}, "
    f"EF_COICOP_TJ_indir_dir {EF_COICOP_TJ_indir_dir.shape}, "
    f"SHS_CAD {SHS_2019_data_CAD_WGHT.shape}, "
    f"SHS_EUR {SHS_2019_data_Mln_EUR_WGHT.shape}, "
    f"NACE_to_COICOP_mask {NACE_to_COICOP_mask.shape}"
)"""))

cells.append(md("""## Cell 1b — Load raw (unweighted) SHS data

Reads the full SHS 2019 PUMF SAS file. No WeightD multiplication, no currency conversion — every column comes through as the raw SAS value. Stored as `SHS_2019_raw`."""))

cells.append(code(r"""_SAS_PATH = os.path.join(
    DATA_DIR,
    r"OneDrive_2026-03-09\Survey of Household Spending\SHS_EDM_2019\Data\SAS\pumf_shs2019.sas7bdat",
)
SHS_2019_raw = pd.read_sas(_SAS_PATH, format="sas7bdat", encoding="latin1")
print(f"SHS_2019_raw shape: {SHS_2019_raw.shape} (rows = households, cols = all SAS variables)")"""))

cells.append(md("""## Cell 1c — OECD-modified equivalence scale

OECD-modified scale: **1.0** for the first adult, **+0.5** for each additional adult (15+), **+0.3** for each child (<15).

`HHType6` encoding:

| Code | Description | Adults | Children |
|---|---|---|---|
| 1 | One person | 1 | 0 |
| 2 | Couple without children | 2 | 0 |
| 3 | Couple with children | 2 | `HHSize − 2` (≤2) |
| 4 | Couple + other persons | 2 + extras | 1 if `P0TO4YN`/`P5TO15YN`=1 |
| 5 | Lone parent | 1 | `HHSize − 1` (≤3) |
| 6 | Other household | heuristic from presence flags | heuristic |

Because `HHSize` is capped at 4, kid counts are lower bounds (≤2 for couples-with-kids, ≤3 for lone parents).

Produces `Unit_hh_cap` with columns `CaseID`, `WeightD`, `OECD_scale`, `OECD_cap = WeightD × OECD_scale`."""))

cells.append(code("""def _oecd_scale_row(row):
    hht, hhs = row["HHType6"], row["HHSize"]
    if pd.isna(hht) or pd.isna(hhs):
        return float("nan")
    hht, hhs = int(hht), int(hhs)
    has_kid = (row.get("P0TO4YN") == 1) or (row.get("P5TO15YN") == 1)

    if hht == 1:
        n_adults, n_kids = 1, 0
    elif hht == 2:
        n_adults, n_kids = 2, 0
    elif hht == 3:
        n_adults, n_kids = 2, max(0, hhs - 2)
    elif hht == 4:
        n_adults = 2
        extras = max(0, hhs - 2)
        if has_kid and extras >= 1:
            n_kids = 1
            n_adults += extras - 1
        else:
            n_kids = 0
            n_adults += extras
    elif hht == 5:
        n_adults, n_kids = 1, max(0, hhs - 1)
    elif hht == 6:
        if has_kid:
            n_kids = 1
            n_adults = max(1, hhs - 1)
        else:
            n_kids = 0
            n_adults = max(1, hhs)
    else:
        n_adults, n_kids = 1, 0

    return 1.0 + 0.5 * (n_adults - 1) + 0.3 * n_kids


_oecd_scale = SHS_2019_raw.apply(_oecd_scale_row, axis=1)

Unit_hh_cap = pd.DataFrame({
    "CaseID":     SHS_2019_raw["CaseID"].values,
    "WeightD":    SHS_2019_raw["WeightD"].values,
    "OECD_scale": _oecd_scale.values,
})
Unit_hh_cap["OECD_cap"] = Unit_hh_cap["WeightD"] * Unit_hh_cap["OECD_scale"]

print(f"Unit_hh_cap shape: {Unit_hh_cap.shape}")
print(f"OECD_scale: mean={Unit_hh_cap['OECD_scale'].mean():.3f}, "
      f"min={Unit_hh_cap['OECD_scale'].min():.3f}, "
      f"max={Unit_hh_cap['OECD_scale'].max():.3f}, "
      f"NaN={Unit_hh_cap['OECD_scale'].isna().sum()}")
print(f"OECD_cap (sum): {Unit_hh_cap['OECD_cap'].sum():,.0f}")
print("OECD_scale distribution (top 10):")
print(Unit_hh_cap["OECD_scale"].value_counts().sort_index().head(10))"""))

cells.append(md("""## Cell 2 — COICOP aggregation matrices + level switch

Loads two aggregation matrices from `COICOP_to_FIGARO_Bridge.xlsx` / tab `COICOP_to_COICOPaggr`:

- **`COICOP75_to_COICOP32`** — 32×75 (D25:BZ56, labels in C25:C56).
- **`COICOP75_to_COICOP12`** — 12×75 (D59:BZ70, labels in C59:C70).

Both share column headers from D8:BZ8 (the 75 detailed COICOP codes).

Set **`coicop_to_coicop_matching`** to `75` (no aggregation), `32`, or `12`. The switch aggregates both `SHS_2019_data_Mln_EUR_WGHT` and the total EF column into `SHS_aggr` and `EF_tot_aggr`."""))

cells.append(code("""_BRIDGE_PATH = os.path.join(DATA_DIR, "COICOP_to_FIGARO_Bridge.xlsx")
_BRIDGE_SHEET = "COICOP_to_COICOPaggr"


def _load_aggr_matrix(skiprows: int, nrows: int) -> pd.DataFrame:
    \"\"\"Read one aggregation block: 'nrows' x 75 with labels in col C.\"\"\"
    col_headers = (
        pd.read_excel(
            _BRIDGE_PATH, sheet_name=_BRIDGE_SHEET, usecols="D:BZ",
            header=None, skiprows=7, nrows=1,
        )
        .iloc[0].astype(str).str.strip().tolist()
    )
    df = pd.read_excel(
        _BRIDGE_PATH, sheet_name=_BRIDGE_SHEET, usecols="C:BZ",
        header=None, skiprows=skiprows, nrows=nrows,
    )
    df.index = df.iloc[:, 0].astype(str).str.strip()
    df = df.iloc[:, 1:].astype(float)
    df.columns = col_headers
    df.index.name = None
    df.columns.name = None
    return df


COICOP75_to_COICOP32 = _load_aggr_matrix(skiprows=24, nrows=32)
COICOP75_to_COICOP12 = _load_aggr_matrix(skiprows=58, nrows=12)
print(f"COICOP75_to_COICOP32 shape: {COICOP75_to_COICOP32.shape}")
print(f"COICOP75_to_COICOP12 shape: {COICOP75_to_COICOP12.shape}")

# === Switch: 75 (no aggregation), 32, or 12 ===
coicop_to_coicop_matching = 12

if coicop_to_coicop_matching == 75:
    _M = None
elif coicop_to_coicop_matching == 32:
    _M = COICOP75_to_COICOP32
elif coicop_to_coicop_matching == 12:
    _M = COICOP75_to_COICOP12
else:
    raise ValueError(
        f"coicop_to_coicop_matching must be 75, 32, or 12, got {coicop_to_coicop_matching!r}"
    )

_shs_75_vals = SHS_2019_data_Mln_EUR_WGHT.iloc[:, 2:]
if _M is None:
    SHS_aggr = SHS_2019_data_Mln_EUR_WGHT.copy()
    EF_tot_aggr = EF_COICOP_TJ_indir_dir["EF_tot_COICOP"].copy()
else:
    _M_aligned = _M[_shs_75_vals.columns]
    SHS_aggr_vals = _shs_75_vals @ _M_aligned.T
    SHS_aggr = SHS_aggr_vals.copy()
    SHS_aggr.insert(0, "WeightD", SHS_2019_data_Mln_EUR_WGHT["WeightD"].values)
    SHS_aggr.insert(0, "CaseID", SHS_2019_data_Mln_EUR_WGHT["CaseID"].values)
    EF_tot_aggr = _M_aligned @ EF_COICOP_TJ_indir_dir["EF_tot_COICOP"]
    EF_tot_aggr.name = "EF_tot_COICOP"

print(f"\\ncoicop_to_coicop_matching = {coicop_to_coicop_matching}")
print(f"SHS_aggr shape:    {SHS_aggr.shape}")
print(f"EF_tot_aggr shape: {EF_tot_aggr.shape}, sum = {EF_tot_aggr.sum():,.2f} TJ")"""))

cells.append(md("""## Cell 3 — SHS spend shares per (aggregated) COICOP category

For each spending column, divide every household's value by the column total. Each column then sums to exactly 1 — a share-of-total-spend per category, per household. Operates on the aggregated SHS table from Cell 2."""))

cells.append(code("""SHS_spend_shares = SHS_aggr.iloc[:, 2:].div(SHS_aggr.iloc[:, 2:].sum(axis=0), axis=1)
SHS_spend_shares.insert(0, "CaseID", SHS_aggr["CaseID"].values)

print(f"SHS_spend_shares shape: {SHS_spend_shares.shape}")
col_sums = SHS_spend_shares.iloc[:, 1:].sum(axis=0)
print(f"Column sums (should all be ~1.0): min={col_sums.min():.6f}, max={col_sums.max():.6f}")
SHS_spend_shares.head()"""))

cells.append(md("""## Cell 4 — Per-household energy footprint by COICOP category

Distribute each category's total energy footprint (TJ) across households using their spend shares:

`Total_EF_hh_TJ[h, c] = SHS_spend_shares[h, c] * EF_tot_aggr[c]`

At the **75-COICOP** level, the indirect/direct breakdown is also produced (`Dir_EF_hh_TJ`, `Indir_EF_hh_TJ`)."""))

cells.append(code("""_shares = SHS_spend_shares.iloc[:, 1:]
Total_EF_hh_TJ = _shares.mul(EF_tot_aggr, axis=1)
Total_EF_hh_TJ.insert(0, "CaseID", SHS_spend_shares["CaseID"].values)
print(f"Total_EF_hh_TJ shape: {Total_EF_hh_TJ.shape}, "
      f"total = {Total_EF_hh_TJ.iloc[:, 1:].values.sum():,.2f} TJ")

if coicop_to_coicop_matching == 75:
    Dir_EF_hh_TJ   = _shares.mul(EF_COICOP_TJ_indir_dir["EF_dir_COICOP"],   axis=1)
    Indir_EF_hh_TJ = _shares.mul(EF_COICOP_TJ_indir_dir["EF_indir_COICOP"], axis=1)
    Dir_EF_hh_TJ.insert(0, "CaseID", SHS_spend_shares["CaseID"].values)
    Indir_EF_hh_TJ.insert(0, "CaseID", SHS_spend_shares["CaseID"].values)
    print(f"Dir_EF_hh_TJ   shape: {Dir_EF_hh_TJ.shape},   total = {Dir_EF_hh_TJ.iloc[:, 1:].values.sum():,.2f} TJ")
    print(f"Indir_EF_hh_TJ shape: {Indir_EF_hh_TJ.shape}, total = {Indir_EF_hh_TJ.iloc[:, 1:].values.sum():,.2f} TJ")"""))

cells.append(md("""## Cell 5 — Stacked bar chart

Two bars (per OECD-cap, per household), each **stacked by COICOP category**. Largest contributors at the bottom. Long category labels are truncated to 25 chars for the legend.

- `EF_per_OECD_cap = EF_tot_aggr / sum(OECD_cap)` — TJ per OECD-equivalent adult.
- `EF_per_hh       = EF_tot_aggr / sum(WeightD)` — TJ per household."""))

cells.append(code("""import matplotlib.pyplot as plt

_sum_OECD_cap = Unit_hh_cap["OECD_cap"].sum()
_sum_WeightD  = Unit_hh_cap["WeightD"].sum()

EF_per_OECD_cap = (EF_tot_aggr / _sum_OECD_cap).sort_values(ascending=False)
EF_per_hh       = EF_tot_aggr.reindex(EF_per_OECD_cap.index) / _sum_WeightD

print(f"sum(OECD_cap) = {_sum_OECD_cap:,.0f}")
print(f"sum(WeightD)  = {_sum_WeightD:,.0f}")
print(f"EF total: {EF_tot_aggr.sum():,.0f} TJ "
      f"-> {EF_per_OECD_cap.sum():.3f} TJ/OECD-cap, "
      f"{EF_per_hh.sum():.3f} TJ/hh")


def _short(label, n=25):
    s = str(label).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


plot_df = pd.DataFrame({
    "Per OECD-cap": EF_per_OECD_cap,
    "Per household": EF_per_hh,
}).T
plot_df.columns = [_short(c) for c in plot_df.columns]

fig, ax = plt.subplots(figsize=(10, 7))
plot_df.plot.bar(
    ax=ax, stacked=True, colormap="tab20", edgecolor="white", linewidth=0.4
)
ax.set_title(f"Energy footprint by COICOP-{coicop_to_coicop_matching} category")
ax.set_ylabel("TJ per unit")
ax.set_xlabel("")
ax.tick_params(axis="x", labelrotation=0)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(title="COICOP category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Household_EF.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Wrote {out_path} with {len(cells)} cells")