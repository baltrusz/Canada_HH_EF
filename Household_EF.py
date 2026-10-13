"""
Household_EF - household energy footprint downstream of Energy_Model.

Runs Energy_Model.py once and caches the four needed objects to a pickle
on disk. On subsequent runs, if none of the source files have changed,
the cache is loaded instantly instead of re-running the full model.

Objects exposed:
  - e_COICOP_TJperMlnEur       : combined indir + dir intensity (TJ / mln EUR)
  - EF_COICOP_TJ_indir_dir     : combined indir + dir energy footprint (TJ)
  - SHS_2019_data_CAD_WGHT     : weighted SHS spends in CAD
  - SHS_2019_data_Mln_EUR_WGHT : weighted SHS spends in million EUR
  - NACE_to_COICOP_mask        : 0/1 bridge mask (62 NACE by 75 COICOP)

Delete the cache file (printed at startup) to force a rebuild.
"""

# %% Cell 1 - Load Energy_Model results (cached)
import os
import pickle
import runpy
import pandas as pd

# ============================================================================
# Configuration — override these to point at your own data locations or to
# change the country/category being analysed. These values are forwarded to
# Energy_Model.py via runpy init_globals, so you can run Household_EF.py from
# anywhere without editing Energy_Model.py.
# ============================================================================
DATA_DIR = r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents\additional_projects\Canada_hh_energy\Data"
FIGARO_FOLDER = r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents - 403001-01 Utvikling og vedlikehold klimakost\General\Utslippsfaktorer\Kryssløpsanalyse\FIGARO data"
COUNTRY_CODE = "CA"
HH_CATEGORY = "P3_S14"
FD_CATEGORIES = ["P3_S13", "P3_S14", "P3_S15", "P51G", "P5M"]

import hashlib
import json

_HERE = os.path.dirname(__file__)
_ENERGY_MODEL_PATH = os.path.join(_HERE, "Energy_Model.py")
_ENERGY_MODEL_GLOBALS = {
    "DATA_DIR": DATA_DIR,
    "FIGARO_FOLDER": FIGARO_FOLDER,
    "COUNTRY_CODE": COUNTRY_CODE,
    "HH_CATEGORY": HH_CATEGORY,
    "FD_CATEGORIES": FD_CATEGORIES,
}
# Hash the config into the cache filename so each path/country combination
# gets its own cache and changes auto-invalidate.
_config_hash = hashlib.md5(
    json.dumps(_ENERGY_MODEL_GLOBALS, sort_keys=True).encode()
).hexdigest()[:8]
_CACHE_PATH = os.path.join(_HERE, f"_cache_household_ef_{_config_hash}.pkl")

# Files whose mtime invalidates the cache (Energy_Model.py + its local imports).
_SOURCE_FILES = [
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


print(f"Cache file: {_CACHE_PATH}")

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

# Add total column = indirect + direct.
EF_COICOP_TJ_indir_dir["EF_tot_COICOP"] = (
    EF_COICOP_TJ_indir_dir["EF_indir_COICOP"]
    + EF_COICOP_TJ_indir_dir["EF_dir_COICOP"]
)

print(
    f"Ready: e_COICOP_TJperMlnEur {e_COICOP_TJperMlnEur.shape}, "
    f"EF_COICOP_TJ_indir_dir {EF_COICOP_TJ_indir_dir.shape}, "
    f"SHS_CAD {SHS_2019_data_CAD_WGHT.shape}, "
    f"SHS_EUR {SHS_2019_data_Mln_EUR_WGHT.shape}, "
    f"NACE_to_COICOP_mask {NACE_to_COICOP_mask.shape}"
)

# %% Cell 1b - Load raw SHS data (full SAS file, all variables)
# Read the entire SHS 2019 PUMF SAS file as-is: every column, raw values, no
# WeightD multiplication, no currency conversion.
_SAS_PATH = os.path.join(
    DATA_DIR,
    r"OneDrive_2026-03-09\Survey of Household Spending\SHS_EDM_2019\Data\SAS\pumf_shs2019.sas7bdat",
)
SHS_2019_raw = pd.read_sas(_SAS_PATH, format="sas7bdat", encoding="latin1")
print(f"SHS_2019_raw shape: {SHS_2019_raw.shape} (rows = households, cols = all SAS variables)")

# %% Cell 1c - OECD-modified equivalence scale per household
# Weights: 1.0 for the first adult, +0.5 per additional adult (15+),
#         +0.3 per child (under 15).
#
# Inputs we have:
#   HHType6  : encoded household type
#     1 One person                                          -> 1 adult, 0 kids
#     2 Couple without children                             -> 2 adults, 0 kids
#     3 Couple with children                                -> 2 adults + (HHSize-2) kids
#     4 Couple with other related or unrelated persons      -> 2 adults + (HHSize-2) extras
#     5 Lone parent, no additional persons                  -> 1 adult + (HHSize-1) kids
#     6 Other household with related or unrelated persons   -> heterogeneous
#   HHSize   : total persons, capped at 4 (= "4 or more").
#   P0TO4YN, P5TO15YN : 1/0 flags for children present (under 15).
#
# Notes:
# - Because HHSize is capped at 4, kid counts inferred for types 3 and 5 are
#   lower bounds (max 2 kids for couples-with-kids, max 3 for lone parents).
# - For ambiguous types (4 and 6) we use the child-presence indicators
#   (P0TO4YN, P5TO15YN) to flag whether to count at least one extra as a
#   child; remaining extras count as adults.

def _oecd_scale_row(row):
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
print(Unit_hh_cap["OECD_scale"].value_counts().sort_index().head(10))

# %% Cell 1d - Weighted income decile (Inc_hh_dec) attached to Unit_hh_cap
# Build a 1..10 decile assignment from the continuous HH_TotInc in SHS_2019_raw,
# weighted by WeightD. Each survey row's WeightD is treated as its "mass" so
# the breakpoints are at the cumulative-weight 10% / 20% / ... quantiles.
# Cumulative-midpoint assignment is used (each row is placed at its own
# weight midpoint inside the cumulative distribution), which avoids the edge
# bias that a plain floor/cumsum approach would introduce.

import numpy as np


def _weighted_decile(values, weights, n=10):
    """Return decile (1..n) per row using cumulative-midpoint quantile."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v, kind="mergesort")
    cum_w = np.cumsum(w[order])
    total = cum_w[-1]
    cum_mid = cum_w - w[order] / 2.0
    decile_sorted = np.minimum(np.floor(cum_mid / total * n).astype(int) + 1, n)
    decile = np.empty_like(decile_sorted)
    decile[order] = decile_sorted
    return decile


_inc_dec_by_caseid = pd.Series(
    _weighted_decile(SHS_2019_raw["HH_TotInc"].values, SHS_2019_raw["WeightD"].values, n=10),
    index=SHS_2019_raw["CaseID"].values,
    name="Inc_hh_dec",
)
Unit_hh_cap["Inc_hh_dec"] = Unit_hh_cap["CaseID"].map(_inc_dec_by_caseid)

# Per-OECD-capita income decile.
# - Income variable: HH_TotInc / OECD_scale (equivalised income, $ per
#   OECD-equivalent adult).
# - Population weight: OECD_cap = WeightD * OECD_scale (number of
#   equivalent adults represented).
# Each decile then contains ~10% of OECD-equivalent adults rather than
# ~10% of households.
_uhc_by_id = Unit_hh_cap.set_index("CaseID")
_caseid_arr = SHS_2019_raw["CaseID"].values
_hh_totinc  = SHS_2019_raw["HH_TotInc"].values
_oecd_scale = _uhc_by_id["OECD_scale"].reindex(_caseid_arr).values
_oecd_cap   = _uhc_by_id["OECD_cap"].reindex(_caseid_arr).values

_equiv_income = _hh_totinc / _oecd_scale

_inc_oecd_dec_by_caseid = pd.Series(
    _weighted_decile(_equiv_income, _oecd_cap, n=10),
    index=_caseid_arr,
    name="Inc_OECDcap_dec",
)
Unit_hh_cap["Inc_OECDcap_dec"] = Unit_hh_cap["CaseID"].map(_inc_oecd_dec_by_caseid)

print("Inc_hh_dec distribution (count of survey rows per decile):")
print(Unit_hh_cap["Inc_hh_dec"].value_counts().sort_index())
print("Inc_hh_dec weighted-population per decile (WeightD totals):")
print(Unit_hh_cap.groupby("Inc_hh_dec")["WeightD"].sum().round(0))

print("\nInc_OECDcap_dec distribution (count of survey rows per decile):")
print(Unit_hh_cap["Inc_OECDcap_dec"].value_counts().sort_index())
print("Inc_OECDcap_dec OECD-equivalent-adults per decile (OECD_cap totals):")
print(Unit_hh_cap.groupby("Inc_OECDcap_dec")["OECD_cap"].sum().round(0))

# %% Cell 2 - COICOP aggregation matrices + level switch
# Load two aggregation matrices from COICOP_to_FIGARO_Bridge_2019.xlsx,
# tab "COICOP_to_COICOPaggr":
#   COICOP75_to_COICOP32 : D25:BZ56 (32 x 75), row labels in C25:C56,
#                          column labels (COICOP75) in D8:BZ8.
#   COICOP75_to_COICOP12 : D59:BZ70 (12 x 75), row labels in C59:C70,
#                          column labels (COICOP75) in D8:BZ8.
# Choose the aggregation level via `coicop_to_coicop_matching` (75, 32, or 12).
# - 75: no aggregation, work with the original 75-COICOP categories.
# - 32 or 12: aggregate both SHS spends and the EF total column.

_BRIDGE_PATH = os.path.join(DATA_DIR, "COICOP_to_FIGARO_Bridge_2019.xlsx")
_BRIDGE_SHEET = "COICOP_to_COICOPaggr"


def _load_aggr_matrix(skiprows: int, nrows: int) -> pd.DataFrame:
    """Read one aggregation block: 'nrows' x 75 with labels in col C."""
    # COICOP75 column headers from Excel row 8 (cells D8:BZ8).
    col_headers = (
        pd.read_excel(
            _BRIDGE_PATH,
            sheet_name=_BRIDGE_SHEET,
            usecols="D:BZ",
            header=None,
            skiprows=7,
            nrows=1,
        )
        .iloc[0]
        .astype(str)
        .str.strip()
        .tolist()
    )
    # Data block: column C = aggregate label, columns D:BZ = 0/1 values.
    df = pd.read_excel(
        _BRIDGE_PATH,
        sheet_name=_BRIDGE_SHEET,
        usecols="C:BZ",
        header=None,
        skiprows=skiprows,
        nrows=nrows,
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

# Aggregate SHS spends (drop CaseID + WeightD for the math, then add CaseID back).
_shs_75_vals = SHS_2019_data_Mln_EUR_WGHT.iloc[:, 2:]
if _M is None:
    SHS_aggr = SHS_2019_data_Mln_EUR_WGHT.copy()
    EF_tot_aggr = EF_COICOP_TJ_indir_dir["EF_tot_COICOP"].copy()
else:
    # Align mask columns to SHS columns to be safe (both are COICOP75 codes).
    _M_aligned = _M[_shs_75_vals.columns]
    SHS_aggr_vals = _shs_75_vals @ _M_aligned.T   # (n_hh, n_aggr)
    SHS_aggr = SHS_aggr_vals.copy()
    SHS_aggr.insert(0, "WeightD", SHS_2019_data_Mln_EUR_WGHT["WeightD"].values)
    SHS_aggr.insert(0, "CaseID", SHS_2019_data_Mln_EUR_WGHT["CaseID"].values)
    EF_tot_aggr = _M_aligned @ EF_COICOP_TJ_indir_dir["EF_tot_COICOP"]   # (n_aggr,)
    EF_tot_aggr.name = "EF_tot_COICOP"

print(f"\ncoicop_to_coicop_matching = {coicop_to_coicop_matching}")
print(f"SHS_aggr shape:    {SHS_aggr.shape}")
print(f"EF_tot_aggr shape: {EF_tot_aggr.shape}, sum = {EF_tot_aggr.sum():,.2f} TJ")

# %% Cell 3 - SHS spend shares per (aggregated) COICOP category
# For each spending column, divide every household's value by the column total
# so each column sums to exactly 1. Operates on the aggregated SHS table from
# Cell 2 (i.e. 75, 32, or 12 categories depending on the switch).
SHS_spend_shares = SHS_aggr.iloc[:, 2:].div(SHS_aggr.iloc[:, 2:].sum(axis=0), axis=1)
SHS_spend_shares.insert(0, "CaseID", SHS_aggr["CaseID"].values)

print(f"SHS_spend_shares shape: {SHS_spend_shares.shape}")
col_sums = SHS_spend_shares.iloc[:, 1:].sum(axis=0)
print(f"Column sums (should all be ~1.0): min={col_sums.min():.6f}, max={col_sums.max():.6f}")
print(SHS_spend_shares.head())

# %% Cell 4 - Per-household-group energy footprint at the chosen COICOP level
# Each SHS sample row represents a "household group" — one survey household
# standing in for WeightD population households. Three tables are produced,
# with the COICOP aggregation level encoded in the variable name (xx = 75/32/12):
#
#   Total_EF_HHgrouped_TJ_xxCOICOP
#       Total energy footprint (TJ) per COICOP category for each household
#       group. Computed as: SHS_spend_shares * EF_tot_aggr. The row value is
#       the sum of EF for all population households the survey row represents.
#
#   avg_EF_per_hh_TJ_xxCOICOP
#       TJ per actual household = Total_EF_HHgrouped_TJ_xxCOICOP / WeightD
#       (values from Unit_hh_cap, matched on CaseID).
#
#   avg_EF_per_OECDcap_TJ_xxCOICOP
#       TJ per OECD-equivalent adult = Total_EF_HHgrouped_TJ_xxCOICOP / OECD_cap
#       (values from Unit_hh_cap, matched on CaseID).
#
# All three keep CaseID as the first column. At the 75-COICOP level the
# indir/dir breakdown is also produced.

_n = coicop_to_coicop_matching
_shares = SHS_spend_shares.iloc[:, 1:]   # drop CaseID for the math

# Per-group total (one row = one survey household, value already weighted).
_total = _shares.mul(EF_tot_aggr, axis=1)
_total.insert(0, "CaseID", SHS_spend_shares["CaseID"].values)

# Align WeightD and OECD_cap to CaseID for safety.
_uhc = Unit_hh_cap.set_index("CaseID")
_w = _uhc["WeightD"].reindex(_total["CaseID"]).values
_c = _uhc["OECD_cap"].reindex(_total["CaseID"]).values

_per_hh = _total.iloc[:, 1:].div(_w, axis=0)
_per_hh.insert(0, "CaseID", _total["CaseID"].values)

_per_cap = _total.iloc[:, 1:].div(_c, axis=0)
_per_cap.insert(0, "CaseID", _total["CaseID"].values)

# Expose under dynamic names that include the aggregation level.
globals()[f"Total_EF_HHgrouped_TJ_{_n}COICOP"] = _total
globals()[f"avg_EF_per_hh_TJ_{_n}COICOP"]          = _per_hh
globals()[f"avg_EF_per_OECDcap_TJ_{_n}COICOP"]     = _per_cap

# Per-row totals across all COICOP categories — one value per survey household.
#   Total_aggr_EF_hh_wght = total TJ for the WeightD households the row represents
#                           (sum across COICOP of Total_EF_HHgrouped_TJ_xxCOICOP).
#                           A sum, not an average. Population totals sum to the
#                           national EF total.
#   avg_EF_per_hh         = total TJ per actual household for that segment
#                           (sum across COICOP of avg_EF_per_hh_TJ_xxCOICOP).
#   avg_EF_perOECD_cap    = total TJ per OECD-equivalent adult for that segment
#                           (sum across COICOP of avg_EF_per_OECDcap_TJ_xxCOICOP).
Total_aggr_EF_hh_wght = pd.DataFrame({
    "CaseID":                _total["CaseID"].values,
    "Total_aggr_EF_hh_wght": _total.iloc[:, 1:].sum(axis=1).values,
})
avg_EF_per_hh = pd.DataFrame({
    "CaseID":        _per_hh["CaseID"].values,
    "avg_EF_per_hh": _per_hh.iloc[:, 1:].sum(axis=1).values,
})
avg_EF_perOECD_cap = pd.DataFrame({
    "CaseID":             _per_cap["CaseID"].values,
    "avg_EF_perOECD_cap": _per_cap.iloc[:, 1:].sum(axis=1).values,
})

print(f"Total_EF_HHgrouped_TJ_{_n}COICOP shape: {_total.shape}, "
      f"total = {_total.iloc[:, 1:].values.sum():,.2f} TJ")
print(f"avg_EF_per_hh_TJ_{_n}COICOP        shape: {_per_hh.shape}")
print(f"avg_EF_per_OECDcap_TJ_{_n}COICOP   shape: {_per_cap.shape}")
print(f"Total_aggr_EF_hh_wght          shape: {Total_aggr_EF_hh_wght.shape}, "
      f"sum  = {Total_aggr_EF_hh_wght['Total_aggr_EF_hh_wght'].sum():,.2f} TJ")
print(f"avg_EF_per_hh                  shape: {avg_EF_per_hh.shape}, "
      f"mean = {avg_EF_per_hh['avg_EF_per_hh'].mean():.4f} TJ/hh")
print(f"avg_EF_perOECD_cap             shape: {avg_EF_perOECD_cap.shape}, "
      f"mean = {avg_EF_perOECD_cap['avg_EF_perOECD_cap'].mean():.4f} TJ/OECD-cap")

if coicop_to_coicop_matching == 75:
    Dir_EF_hh_TJ   = _shares.mul(EF_COICOP_TJ_indir_dir["EF_dir_COICOP"],   axis=1)
    Indir_EF_hh_TJ = _shares.mul(EF_COICOP_TJ_indir_dir["EF_indir_COICOP"], axis=1)
    Dir_EF_hh_TJ.insert(0, "CaseID", SHS_spend_shares["CaseID"].values)
    Indir_EF_hh_TJ.insert(0, "CaseID", SHS_spend_shares["CaseID"].values)
    print(f"Dir_EF_hh_TJ   shape: {Dir_EF_hh_TJ.shape},   total = {Dir_EF_hh_TJ.iloc[:, 1:].values.sum():,.2f} TJ")
    print(f"Indir_EF_hh_TJ shape: {Indir_EF_hh_TJ.shape}, total = {Indir_EF_hh_TJ.iloc[:, 1:].values.sum():,.2f} TJ")

# %% Cell 5 - Stacked bar chart: per OECD-cap and per household EF, segmented by COICOP category
# Two bars (per OECD-cap, per household). Each bar is stacked by COICOP
# category, largest contributors at the bottom. Long category labels are
# truncated for the legend.
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Colourblind-friendly categorical palette (Okabe-Ito 8 + Paul Tol "muted" 7).
# Tested for deuteranopia, protanopia and tritanopia. The colormap cycles if
# there are more categories than colors (use 12-COICOP for cleanest result).
_COLORBLIND_PALETTE = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
    "#D55E00", "#CC79A7", "#999999", "#332288", "#117733",
    "#882255", "#44AA99", "#DDCC77", "#88CCEE", "#AA4499",
]
_CB_CMAP = ListedColormap(_COLORBLIND_PALETTE, name="cb15")
_CB_HATCHES = ("", "////", "....", "xxxx")


def _add_hatching(ax, palette_size=len(_COLORBLIND_PALETTE), hatches=_CB_HATCHES):
    """Add hatch patterns to stacked-bar segments past `palette_size` so cycled
    colours become distinguishable. Works after `df.plot.bar(stacked=True)`."""
    for i, container in enumerate(ax.containers):
        hatch = hatches[(i // palette_size) % len(hatches)]
        if hatch:
            for patch in container:
                patch.set_hatch(hatch)
    leg = ax.get_legend()
    if leg is None:
        return
    handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", [])
    for i, handle in enumerate(handles):
        hatch = hatches[(i // palette_size) % len(hatches)]
        if hatch:
            handle.set_hatch(hatch)

_sum_OECD_cap = Unit_hh_cap["OECD_cap"].sum()
_sum_WeightD  = Unit_hh_cap["WeightD"].sum()

avg_EF_weighted_OECDcap_per_COICOP = (EF_tot_aggr / _sum_OECD_cap).sort_values(ascending=False)
avg_EF_weighted_hh_per_COICOP = EF_tot_aggr.reindex(avg_EF_weighted_OECDcap_per_COICOP.index) / _sum_WeightD

print(f"sum(OECD_cap) = {_sum_OECD_cap:,.0f}")
print(f"sum(WeightD)  = {_sum_WeightD:,.0f}")
print(f"EF total: {EF_tot_aggr.sum():,.0f} TJ "
      f"-> {avg_EF_weighted_OECDcap_per_COICOP.sum():.3f} TJ/OECD-cap, "
      f"{avg_EF_weighted_hh_per_COICOP.sum():.3f} TJ/hh")


def _short(label, n=25):
    s = str(label).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


plot_df = pd.DataFrame({
    "Per OECD-cap": avg_EF_weighted_OECDcap_per_COICOP,
    "Per household": avg_EF_weighted_hh_per_COICOP,
}).T
plot_df.columns = [_short(c) for c in plot_df.columns]

fig, ax = plt.subplots(figsize=(10, 7))
plot_df.plot.bar(
    ax=ax, stacked=True, color=_COLORBLIND_PALETTE, edgecolor="white", linewidth=0.4
)
ax.set_title(f"Energy footprint by COICOP-{coicop_to_coicop_matching} category")
ax.set_ylabel("TJ per unit")
ax.set_xlabel("")
ax.tick_params(axis="x", labelrotation=0)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(title="COICOP category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
_add_hatching(ax)
plt.tight_layout()
plt.show()

# %% Cell 6 - Stacked bar: average household EF by province (PROV)
# Group households by PROV from SHS_2019_raw, then compute the weight-average
# TJ per household per COICOP category:
#   avg_TJ_per_hh[prov, cat] = sum(Total_EF_hh_TJ[h, cat] for h in prov)
#                              / sum(WeightD[h] for h in prov)
# Each bar = one province, stacked by COICOP category. The same `_short`
# label-truncation helper from Cell 5 is reused.

_PROV_NAMES = {
    "10": "NL", "11": "PE", "12": "NS", "13": "NB",
    "24": "QC", "35": "ON",
    "46": "MB", "47": "SK", "48": "AB", "59": "BC",
    "63": "Territories",   # YT+NT+NU bundled in PUMF for confidentiality
}

_total_df = globals()[f"Total_EF_HHgrouped_TJ_{coicop_to_coicop_matching}COICOP"]
_meta_prov = SHS_2019_raw[["CaseID", "Prov", "WeightD"]]
_merged = _total_df.merge(_meta_prov, on="CaseID", how="inner")

_cat_cols = [c for c in _total_df.columns if c != "CaseID"]
_grp = _merged.groupby("Prov")
EF_per_hh_by_PROV = _grp[_cat_cols].sum().div(_grp["WeightD"].sum(), axis=0)

# Sort columns by total contribution so largest stacks at the bottom.
_col_order = EF_per_hh_by_PROV.sum(axis=0).sort_values(ascending=False).index
EF_per_hh_by_PROV = EF_per_hh_by_PROV[_col_order]

# Pretty province labels (fall back to raw code if unknown). Prov is stored
# as a string like "10", "24", ...
EF_per_hh_by_PROV.index = [
    _PROV_NAMES.get(str(p).strip(), str(p).strip()) for p in EF_per_hh_by_PROV.index
]

print(f"EF_per_hh_by_PROV shape: {EF_per_hh_by_PROV.shape}")
print(EF_per_hh_by_PROV.round(4))

fig, ax = plt.subplots(figsize=(12, 7))
EF_per_hh_by_PROV.rename(columns=_short).plot.bar(
    ax=ax, stacked=True, color=_COLORBLIND_PALETTE, edgecolor="white", linewidth=0.4
)
ax.set_title(f"Average household EF by province (COICOP-{coicop_to_coicop_matching})")
ax.set_ylabel("TJ per household")
ax.set_xlabel("Province")
ax.tick_params(axis="x", labelrotation=0)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(title="COICOP category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
_add_hatching(ax)
plt.tight_layout()
plt.show()

# %% Cell 7 - Stacked bar: average per-OECDcap EF by household type (HHType6)
# Group households by HHType6 from SHS_2019_raw, then compute the OECD_cap-
# weighted mean of EF_per_OECDcap_TJ per COICOP category:
#   avg[hht, c] = sum_h(EF_per_OECDcap[h, c] * OECD_cap[h]) / sum_h(OECD_cap[h])
# This gives TJ per OECD-equivalent adult, averaged across the people living
# in households of that type (rather than averaging across survey rows).

_HHTYPE_NAMES = {
    1: "One person",
    2: "Couple, no kids",
    3: "Couple with kids",
    4: "Couple + others",
    5: "Lone parent",
    6: "Other household",
}

_n = coicop_to_coicop_matching
_per_cap_df = globals()[f"avg_EF_per_OECDcap_TJ_{_n}COICOP"]

_meta_h = SHS_2019_raw[["CaseID", "HHType6"]].merge(
    Unit_hh_cap[["CaseID", "OECD_cap"]], on="CaseID", how="inner"
)
_merged = _per_cap_df.merge(_meta_h, on="CaseID", how="inner")
_cat_cols = [c for c in _per_cap_df.columns if c != "CaseID"]

# OECD_cap-weighted mean per HHType6.
_weighted = _merged[_cat_cols].multiply(_merged["OECD_cap"], axis=0)
_weighted["HHType6"] = _merged["HHType6"].values
EF_per_OECDcap_by_HHType6 = (
    _weighted.groupby("HHType6").sum()
    .div(_merged.groupby("HHType6")["OECD_cap"].sum(), axis=0)
)

# Sort columns by total contribution so largest stacks at the bottom.
_col_order = EF_per_OECDcap_by_HHType6.sum(axis=0).sort_values(ascending=False).index
EF_per_OECDcap_by_HHType6 = EF_per_OECDcap_by_HHType6[_col_order]

# Pretty household-type labels (fall back to raw code if unknown).
EF_per_OECDcap_by_HHType6.index = [
    _HHTYPE_NAMES.get(int(t), str(int(t))) for t in EF_per_OECDcap_by_HHType6.index
]

print(f"EF_per_OECDcap_by_HHType6 shape: {EF_per_OECDcap_by_HHType6.shape}")
print(EF_per_OECDcap_by_HHType6.round(4))

fig, ax = plt.subplots(figsize=(12, 7))
EF_per_OECDcap_by_HHType6.rename(columns=_short).plot.bar(
    ax=ax, stacked=True, color=_COLORBLIND_PALETTE, edgecolor="white", linewidth=0.4
)
ax.set_title(f"Average per-OECD-cap EF by household type (COICOP-{coicop_to_coicop_matching})")
ax.set_ylabel("TJ per OECD-cap")
ax.set_xlabel("Household type")
ax.tick_params(axis="x", labelrotation=15)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(title="COICOP category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
_add_hatching(ax)
plt.tight_layout()
plt.show()

# %% Cell 8 - Stacked bar: average per-OECDcap EF by reference-person age group
# Same OECD_cap-weighted-mean method as Cell 7, but grouped by RP_AgeGrp
# (age of the household reference person). RP_AgeGrp is stored as a string
# code "01"..."06" in the PUMF file.

_RP_AGE_NAMES = {
    "01": "<30",
    "02": "30-39",
    "03": "40-54",
    "04": "55-64",
    "05": "65-74",
    "06": "75+",
}

_per_cap_df = globals()[f"avg_EF_per_OECDcap_TJ_{coicop_to_coicop_matching}COICOP"]

_meta_a = SHS_2019_raw[["CaseID", "RP_AgeGrp"]].merge(
    Unit_hh_cap[["CaseID", "OECD_cap"]], on="CaseID", how="inner"
)
_merged = _per_cap_df.merge(_meta_a, on="CaseID", how="inner")
_cat_cols = [c for c in _per_cap_df.columns if c != "CaseID"]

# OECD_cap-weighted mean per RP_AgeGrp.
_weighted = _merged[_cat_cols].multiply(_merged["OECD_cap"], axis=0)
_weighted["RP_AgeGrp"] = _merged["RP_AgeGrp"].values
EF_per_OECDcap_by_RPAgeGrp = (
    _weighted.groupby("RP_AgeGrp").sum()
    .div(_merged.groupby("RP_AgeGrp")["OECD_cap"].sum(), axis=0)
)

# Sort columns by total contribution so largest stacks at the bottom.
_col_order = EF_per_OECDcap_by_RPAgeGrp.sum(axis=0).sort_values(ascending=False).index
EF_per_OECDcap_by_RPAgeGrp = EF_per_OECDcap_by_RPAgeGrp[_col_order]

# Pretty age-group labels (preserve numeric order on the x-axis).
EF_per_OECDcap_by_RPAgeGrp = EF_per_OECDcap_by_RPAgeGrp.sort_index()
EF_per_OECDcap_by_RPAgeGrp.index = [
    _RP_AGE_NAMES.get(str(a).strip(), str(a).strip()) for a in EF_per_OECDcap_by_RPAgeGrp.index
]

print(f"EF_per_OECDcap_by_RPAgeGrp shape: {EF_per_OECDcap_by_RPAgeGrp.shape}")
print(EF_per_OECDcap_by_RPAgeGrp.round(4))

fig, ax = plt.subplots(figsize=(12, 7))
EF_per_OECDcap_by_RPAgeGrp.rename(columns=_short).plot.bar(
    ax=ax, stacked=True, color=_COLORBLIND_PALETTE, edgecolor="white", linewidth=0.4
)
ax.set_title(
    f"Average per-OECD-cap EF by reference-person age group "
    f"(COICOP-{coicop_to_coicop_matching})"
)
ax.set_ylabel("TJ per OECD-cap")
ax.set_xlabel("Age of reference person")
ax.tick_params(axis="x", labelrotation=0)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(title="COICOP category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
_add_hatching(ax)
plt.tight_layout()
plt.show()

# %% Cell 9 - Per-OECD-cap EF by additional SHS variables (one chart per var)
# Same OECD_cap-weighted-mean method as Cells 7 / 8, factored into a helper
# so we can sweep through several socio-demographic variables.
# Note: the user-mentioned `HH_ManIncSrc` does not exist in the PUMF; the
# actual variable is `HH_MajIncSrc` (household major income source).

_VAR_SPECS = {
    "RP_Gender":   ({"1": "Male", "2": "Female"},
                    "Reference person gender", False),
    "RP_MarStat":  ({"1": "Married/Common-law", "2": "Widow/Sep/Div", "3": "Single"},
                    "Reference person marital status", False),
    "RP_Educ":     ({"1": "<High school", "2": "High school", "3": "Trade/college",
                     "4": "University", "9": "Not stated"},
                    "Reference person education", True),
    "DwelTyp":     ({"1": "Single detached", "2": "Single attached", "3": "Apartment"},
                    "Dwelling type", False),
    "Tenure":      ({"1": "Owner w/ mortgage", "2": "Owner w/o mortgage", "3": "Renter"},
                    "Tenure", False),
    "NumBedr":     ({"1": "1", "2": "2", "3": "3", "4": "4+", "9": "Not stated"},
                    "Number of bedrooms", True),
    "SecResYN":    ({"1": "Yes", "2": "No"},
                    "Has secondary residence", False),
    "OthPropYN":   ({"1": "Yes", "2": "No"},
                    "Has other property", False),
    "InternetYN":  ({"1": "Yes", "2": "No"},
                    "Has internet at home", False),
    "VehicleYN":   ({"1": "Yes", "2": "No"},
                    "Has vehicle", False),
    "HH_MajIncSrc": ({"1": "Wages/salaries", "2": "Self-employment",
                      "3": "Investment/pension", "4": "Govt transfers"},
                     "Major household income source", False),
}

_per_cap_df = globals()[f"avg_EF_per_OECDcap_TJ_{coicop_to_coicop_matching}COICOP"]
_cat_cols = [c for c in _per_cap_df.columns if c != "CaseID"]


def _plot_ef_by(col, labels_map, x_label, sort_by_index):
    """OECD_cap-weighted mean of EF_per_OECDcap per category of `col`."""
    meta = SHS_2019_raw[["CaseID", col]].merge(
        Unit_hh_cap[["CaseID", "OECD_cap"]], on="CaseID", how="inner"
    )
    merged = _per_cap_df.merge(meta, on="CaseID", how="inner")
    weighted = merged[_cat_cols].multiply(merged["OECD_cap"], axis=0)
    weighted[col] = merged[col].values
    result = (
        weighted.groupby(col).sum()
        .div(merged.groupby(col)["OECD_cap"].sum(), axis=0)
    )
    # Stack largest COICOP contributors at the bottom (same order across charts).
    col_order = result.sum(axis=0).sort_values(ascending=False).index
    result = result[col_order]
    if sort_by_index:
        result = result.sort_index()
    # Pretty x-axis labels.
    result.index = [labels_map.get(str(v).strip(), str(v).strip()) for v in result.index]

    fig, ax = plt.subplots(figsize=(11, 6))
    result.rename(columns=_short).plot.bar(
        ax=ax, stacked=True, color=_COLORBLIND_PALETTE, edgecolor="white", linewidth=0.4
    )
    ax.set_title(
        f"Avg per-OECD-cap EF by {x_label.lower()} (COICOP-{coicop_to_coicop_matching})"
    )
    ax.set_ylabel("TJ per OECD-cap")
    ax.set_xlabel(x_label)
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="COICOP category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _add_hatching(ax)
    plt.tight_layout()
    plt.show()
    return result


_results_by_var = {}
for _col, (_labels, _xlabel, _sort_idx) in _VAR_SPECS.items():
    if _col not in SHS_2019_raw.columns:
        print(f"Skipping {_col} (not in SHS_2019_raw)")
        continue
    _results_by_var[_col] = _plot_ef_by(_col, _labels, _xlabel, _sort_idx)

# %% Cell 10 - Stacked bar: average household EF by weighted income decile
# Use Total_EF_HHgrouped_TJ_xxCOICOP (which holds the energy footprint of
# every population household represented by each survey row) and aggregate
# by Inc_hh_dec (built in Cell 1d). Per decile:
#   avg_TJ_per_hh[dec, c] = sum_h(Total_EF[h, c]) / sum_h(WeightD[h])
# Each bar = one income decile (D1 = lowest income, D10 = highest), stacked
# by COICOP category, largest contributors at the bottom.

_total_df = globals()[f"Total_EF_HHgrouped_TJ_{coicop_to_coicop_matching}COICOP"]
_meta_d = Unit_hh_cap[["CaseID", "Inc_hh_dec", "WeightD"]]
_merged = _total_df.merge(_meta_d, on="CaseID", how="inner")
_cat_cols = [c for c in _total_df.columns if c != "CaseID"]

_grp = _merged.groupby("Inc_hh_dec")
EF_per_hh_by_IncDec = _grp[_cat_cols].sum().div(_grp["WeightD"].sum(), axis=0)

# Sort COICOP columns by total contribution (largest at bottom of the stack)
# and keep deciles in natural numeric order on the x-axis.
_col_order = EF_per_hh_by_IncDec.sum(axis=0).sort_values(ascending=False).index
EF_per_hh_by_IncDec = EF_per_hh_by_IncDec[_col_order].sort_index()

# Weighted mean income (HH_TotInc, CAD) per decile, for x-tick annotation:
#   mean_inc[dec] = sum_h(HH_TotInc[h] * WeightD[h]) / sum_h(WeightD[h])
_inc_meta = SHS_2019_raw[["CaseID", "HH_TotInc"]].merge(
    Unit_hh_cap[["CaseID", "Inc_hh_dec", "WeightD"]], on="CaseID", how="inner"
)
mean_inc_per_IncDec = (
    (_inc_meta["HH_TotInc"] * _inc_meta["WeightD"])
    .groupby(_inc_meta["Inc_hh_dec"]).sum()
    .div(_inc_meta.groupby("Inc_hh_dec")["WeightD"].sum())
).sort_index()

# Two-line x-tick labels: "D1\n$24k" etc.
EF_per_hh_by_IncDec.index = [
    f"D{int(d)}\n${mean_inc_per_IncDec.loc[d] / 1000:.0f}k"
    for d in EF_per_hh_by_IncDec.index
]

print(f"EF_per_hh_by_IncDec shape: {EF_per_hh_by_IncDec.shape}")
print("Weighted mean HH_TotInc per decile (CAD):")
print(mean_inc_per_IncDec.round(0))
print(EF_per_hh_by_IncDec.round(4))

fig, ax = plt.subplots(figsize=(12, 7))
EF_per_hh_by_IncDec.rename(columns=_short).plot.bar(
    ax=ax, stacked=True, color=_COLORBLIND_PALETTE, edgecolor="white", linewidth=0.4
)
ax.set_title(f"Average household EF by income decile (COICOP-{coicop_to_coicop_matching})")
ax.set_ylabel("TJ per household")
ax.set_xlabel("Income decile (D1 = lowest, D10 = highest) — mean HH income shown below")
ax.tick_params(axis="x", labelrotation=0)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(title="COICOP category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
_add_hatching(ax)
plt.tight_layout()
plt.show()

# %% Cell 11 - Stacked bar: average TJ per OECD-cap by per-OECD-cap income decile
# Uses Inc_OECDcap_dec (built in Cell 1d) — deciles of equivalised income
# (HH_TotInc / OECD_scale) weighted by OECD_cap. Each decile holds ~10% of
# OECD-equivalent adults.
# Aggregation:
#   avg_TJ_per_cap[dec, c] = sum_h(Total_EF[h, c]) / sum_h(OECD_cap[h])
# X-tick labels show the weighted-mean equivalised income per decile.

_total_df = globals()[f"Total_EF_HHgrouped_TJ_{coicop_to_coicop_matching}COICOP"]
_meta_o = Unit_hh_cap[["CaseID", "Inc_OECDcap_dec", "OECD_cap"]]
_merged = _total_df.merge(_meta_o, on="CaseID", how="inner")
_cat_cols = [c for c in _total_df.columns if c != "CaseID"]

_grp = _merged.groupby("Inc_OECDcap_dec")
EF_per_OECDcap_by_IncOECDDec = _grp[_cat_cols].sum().div(_grp["OECD_cap"].sum(), axis=0)

# Sort COICOP columns by total contribution; keep deciles in numeric order.
_col_order = EF_per_OECDcap_by_IncOECDDec.sum(axis=0).sort_values(ascending=False).index
EF_per_OECDcap_by_IncOECDDec = EF_per_OECDcap_by_IncOECDDec[_col_order].sort_index()

# Weighted-mean equivalised income (CAD per OECD-cap) per decile:
#   mean_eq_inc[dec] = sum_h((HH_TotInc/OECD_scale) * OECD_cap) / sum_h(OECD_cap)
_eq_inc_meta = pd.DataFrame({
    "CaseID":    SHS_2019_raw["CaseID"].values,
    "EqIncome":  SHS_2019_raw["HH_TotInc"].values / _oecd_scale,
}).merge(Unit_hh_cap[["CaseID", "Inc_OECDcap_dec", "OECD_cap"]], on="CaseID", how="inner")

mean_eq_inc_per_dec = (
    (_eq_inc_meta["EqIncome"] * _eq_inc_meta["OECD_cap"])
    .groupby(_eq_inc_meta["Inc_OECDcap_dec"]).sum()
    .div(_eq_inc_meta.groupby("Inc_OECDcap_dec")["OECD_cap"].sum())
).sort_index()

EF_per_OECDcap_by_IncOECDDec.index = [
    f"D{int(d)}\n${mean_eq_inc_per_dec.loc[d] / 1000:.0f}k"
    for d in EF_per_OECDcap_by_IncOECDDec.index
]

print(f"EF_per_OECDcap_by_IncOECDDec shape: {EF_per_OECDcap_by_IncOECDDec.shape}")
print("Weighted-mean equivalised income (CAD / OECD-cap) per decile:")
print(mean_eq_inc_per_dec.round(0))
print(EF_per_OECDcap_by_IncOECDDec.round(4))

fig, ax = plt.subplots(figsize=(12, 7))
EF_per_OECDcap_by_IncOECDDec.rename(columns=_short).plot.bar(
    ax=ax, stacked=True, color=_COLORBLIND_PALETTE, edgecolor="white", linewidth=0.4
)
ax.set_title(f"Average EF per OECD-cap by per-OECD-cap income decile (COICOP-{coicop_to_coicop_matching})")
ax.set_ylabel("TJ per OECD-cap")
ax.set_xlabel("Per-OECD-cap income decile — weighted mean equivalised income shown below")
ax.tick_params(axis="x", labelrotation=0)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(title="COICOP category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
_add_hatching(ax)
plt.tight_layout()
plt.show()

# %% Cell 12 - Assemble data for statistical analysis
# Build a single DataFrame `var_for_stat_analysis` with one row per survey
# household, joining everything needed for downstream regression / inequality
# work. Save to disk so a separate notebook can `pd.read_pickle(...)` it
# without re-running this script.

_socio_vars = [
    "Prov", "HHType6", "RP_AgeGrp",
    "RP_Gender", "RP_MarStat", "RP_Educ",
    "DwelTyp", "Tenure", "NumBedr",
    "SecResYN", "OthPropYN", "InternetYN", "VehicleYN",
    "HH_MajIncSrc", "HH_TotInc",
]
_existing_socio = [c for c in _socio_vars if c in SHS_2019_raw.columns]
_missing_socio  = [c for c in _socio_vars if c not in SHS_2019_raw.columns]
if _missing_socio:
    print(f"Note: missing in SHS_2019_raw, skipped: {_missing_socio}")

# 1. Start with Unit_hh_cap (CaseID, WeightD, OECD_scale, OECD_cap, deciles)
var_for_stat_analysis = Unit_hh_cap.copy()

# 2. Join socio-demographic + income vars from SHS_2019_raw
var_for_stat_analysis = var_for_stat_analysis.merge(
    SHS_2019_raw[["CaseID"] + _existing_socio], on="CaseID", how="left"
)

# 3. Join EF row-totals (each is CaseID + one column)
var_for_stat_analysis = var_for_stat_analysis.merge(Total_aggr_EF_hh_wght, on="CaseID", how="left")
var_for_stat_analysis = var_for_stat_analysis.merge(avg_EF_per_hh,         on="CaseID", how="left")
var_for_stat_analysis = var_for_stat_analysis.merge(avg_EF_perOECD_cap,    on="CaseID", how="left")

# 4. Join per-COICOP per-hh and per-OECD-cap tables, prefixing column names
#    so per-hh and per-cap variants of the same COICOP don't collide. Each
#    COICOP label is also converted into a safe Python identifier so the
#    columns work with statsmodels.formula and similar tools.
import re


def _sanitize_colname(s, prefix="C", max_len=50):
    """Convert a COICOP-like label into a safe Python identifier."""
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if s and s[0].isdigit():
        s = f"{prefix}_{s}"
    return (s[:max_len] or prefix).rstrip("_")


_per_hh  = globals()[f"avg_EF_per_hh_TJ_{coicop_to_coicop_matching}COICOP"].copy()
_per_cap = globals()[f"avg_EF_per_OECDcap_TJ_{coicop_to_coicop_matching}COICOP"].copy()

# Build sanitized column names + a label map (sanitized → original).
_orig_labels = list(_per_hh.columns[1:])
_safe_labels = [_sanitize_colname(c) for c in _orig_labels]
if len(set(_safe_labels)) != len(_safe_labels):
    # Disambiguate any duplicates by appending a suffix.
    seen = {}
    deduped = []
    for s in _safe_labels:
        n = seen.get(s, 0)
        seen[s] = n + 1
        deduped.append(s if n == 0 else f"{s}_{n}")
    _safe_labels = deduped

coicop_label_map = {}
_hh_cols, _cap_cols = [], []
for orig, safe in zip(_orig_labels, _safe_labels):
    hh_name  = f"avg_EF_per_hh__{safe}"
    cap_name = f"avg_EF_per_cap__{safe}"
    _hh_cols.append(hh_name)
    _cap_cols.append(cap_name)
    coicop_label_map[hh_name]  = orig
    coicop_label_map[cap_name] = orig

_per_hh.columns  = ["CaseID"] + _hh_cols
_per_cap.columns = ["CaseID"] + _cap_cols
var_for_stat_analysis = var_for_stat_analysis.merge(_per_hh,  on="CaseID", how="left")
var_for_stat_analysis = var_for_stat_analysis.merge(_per_cap, on="CaseID", how="left")

print(f"var_for_stat_analysis shape: {var_for_stat_analysis.shape}")
print(f"Columns ({len(var_for_stat_analysis.columns)}):")
for c in var_for_stat_analysis.columns:
    print(f"  {c}")

# Save to pickle (preserves dtypes; fast to load in another notebook).
# Filename includes the COICOP level so different aggregations don't clash.
# Also save a JSON sidecar mapping sanitized column → original COICOP label.
_OUT_PATH = os.path.join(
    _HERE, f"var_for_stat_analysis_{coicop_to_coicop_matching}COICOP.pkl"
)
_MAP_PATH = os.path.join(
    _HERE, f"var_for_stat_analysis_{coicop_to_coicop_matching}COICOP_labels.json"
)
var_for_stat_analysis.to_pickle(_OUT_PATH)
with open(_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(coicop_label_map, f, indent=2, ensure_ascii=False)

print(f"\nSaved data:   {_OUT_PATH}")
print(f"Saved labels: {_MAP_PATH}")
print("To load in another notebook:")
print(f'  import pandas as pd, json')
print(f'  df  = pd.read_pickle(r"{_OUT_PATH}")')
print(f'  lbl = json.load(open(r"{_MAP_PATH}", encoding="utf-8"))  # safe_col -> original COICOP label')

# %%