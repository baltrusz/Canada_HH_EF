"""
Household_EF - household energy footprint downstream of Energy_Model.

Runs Energy_Model.py for the chosen YEARS (any of 2017, 2019, 2021) and
caches its outputs to one pickle per year. On subsequent runs, if none of
the source files have changed, the cache is loaded instantly instead of
re-running the full model; only years without a fresh cache are recomputed.

Every per-year object is a dict keyed by year, e.g.
  - e_COICOP_TJperMlnEur[year]   : combined indir + dir intensity (TJ / mln EUR)
  - EF_COICOP_TJ_indir_dir[year] : combined indir + dir energy footprint (TJ)
  - SHS_data_CAD_WGHT[year]      : weighted SHS spends in CAD
  - SHS_data_Mln_EUR_WGHT[year]  : weighted SHS spends in million EUR
  - NACE_to_COICOP_mask[year]    : 0/1 bridge mask (62 NACE by n COICOP)

Charts show one panel per year. The data for statistical analysis (Cell 12)
stacks all chosen years with a `year` column.

Delete the cache files (printed at startup) to force a rebuild.
"""

# %% Cell 1 - Load Energy_Model results (cached per year)
import os
import pickle
import runpy
import hashlib
import json
import numpy as np
import pandas as pd

from year_config import YEAR_CONFIG, check_years, path_for, load_shs_raw

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

# === Years to analyse: any subset of [2017, 2019, 2021] ===
YEARS = check_years([2017, 2019, 2021])

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_ENERGY_MODEL_PATH = os.path.join(_HERE, "Energy_Model.py")
_ENERGY_MODEL_GLOBALS = {
    "DATA_DIR": DATA_DIR,
    "FIGARO_FOLDER": FIGARO_FOLDER,
    "COUNTRY_CODE": COUNTRY_CODE,
    "HH_CATEGORY": HH_CATEGORY,
    "FD_CATEGORIES": FD_CATEGORIES,
}
# Hash the config into the cache filename so each path/country combination
# gets its own cache and changes auto-invalidate. One file per year.
_config_hash = hashlib.md5(
    json.dumps(_ENERGY_MODEL_GLOBALS, sort_keys=True).encode()
).hexdigest()[:8]


def _cache_path(year):
    return os.path.join(_HERE, f"_cache_household_ef_{_config_hash}_{year}.pkl")


# Files whose mtime invalidates the cache (Energy_Model.py + its local imports).
_SOURCE_FILES = [
    "Energy_Model.py",
    "Figaro_economic_data.py",
    "SHS_data.py",
    "NACE_COICOP_mask.py",
    "RAS_balancing.py",
    "Indir_E_e.py",
    "year_config.py",
]
_KEYS = [
    "e_COICOP_TJperMlnEur",
    "EF_COICOP_TJ_indir_dir",
    "SHS_data_CAD_WGHT",
    "SHS_data_Mln_EUR_WGHT",
    "NACE_to_COICOP_mask",
]


def _cache_is_fresh(year) -> bool:
    path = _cache_path(year)
    if not os.path.exists(path):
        return False
    cache_mtime = os.path.getmtime(path)
    for name in _SOURCE_FILES:
        src = os.path.join(_HERE, name)
        if os.path.exists(src) and os.path.getmtime(src) > cache_mtime:
            return False
    return True


_data = {}
_stale = [y for y in YEARS if not _cache_is_fresh(y)]
for _y in YEARS:
    print(f"Cache file {_y}: {_cache_path(_y)}")
    if _y not in _stale:
        with open(_cache_path(_y), "rb") as f:
            _data[_y] = pickle.load(f)
        print(f"  loaded cached Energy_Model outputs for {_y}")

if _stale:
    print(f"Cache missing or stale for {_stale} -- running Energy_Model.py (progress below)")
    print("-" * 70)
    _ns = runpy.run_path(
        _ENERGY_MODEL_PATH,
        run_name="__energy_model__",
        init_globals={**_ENERGY_MODEL_GLOBALS, "_YEARS_OVERRIDE": _stale},
    )
    print("-" * 70)
    for _y in _stale:
        _data[_y] = {k: _ns["EM"][_y][k] for k in _KEYS}
        with open(_cache_path(_y), "wb") as f:
            pickle.dump(_data[_y], f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Cached {len(_KEYS)} objects for {_y}.")
    del _ns

e_COICOP_TJperMlnEur   = {y: _data[y]["e_COICOP_TJperMlnEur"] for y in YEARS}
EF_COICOP_TJ_indir_dir = {y: _data[y]["EF_COICOP_TJ_indir_dir"] for y in YEARS}
SHS_data_CAD_WGHT      = {y: _data[y]["SHS_data_CAD_WGHT"] for y in YEARS}
SHS_data_Mln_EUR_WGHT  = {y: _data[y]["SHS_data_Mln_EUR_WGHT"] for y in YEARS}
NACE_to_COICOP_mask    = {y: _data[y]["NACE_to_COICOP_mask"] for y in YEARS}
del _data

for y in YEARS:
    # Add total column = indirect + direct.
    EF_COICOP_TJ_indir_dir[y]["EF_tot_COICOP"] = (
        EF_COICOP_TJ_indir_dir[y]["EF_indir_COICOP"]
        + EF_COICOP_TJ_indir_dir[y]["EF_dir_COICOP"]
    )
    print(
        f"Ready {y}: e_COICOP_TJperMlnEur {e_COICOP_TJperMlnEur[y].shape}, "
        f"EF_COICOP_TJ_indir_dir {EF_COICOP_TJ_indir_dir[y].shape}, "
        f"SHS_CAD {SHS_data_CAD_WGHT[y].shape}, "
        f"SHS_EUR {SHS_data_Mln_EUR_WGHT[y].shape}, "
        f"NACE_to_COICOP_mask {NACE_to_COICOP_mask[y].shape}"
    )

# %% Cell 1b - Load raw SHS data (full PUMF file, all variables)
# Read each year's entire SHS PUMF file as-is: every column, raw values, no
# WeightD multiplication, no currency conversion. Socio-demographic columns
# are renamed to the 2019 names (year_config.harmonise_shs), so the cells
# below work unchanged for every year.
SHS_raw = {}
for y in YEARS:
    SHS_raw[y] = load_shs_raw(DATA_DIR, y)
    print(f"SHS_raw[{y}] shape: {SHS_raw[y].shape} (rows = households, cols = all variables)")

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


Unit_hh_cap = {}
for y in YEARS:
    _oecd_scale = SHS_raw[y].apply(_oecd_scale_row, axis=1)

    _u = pd.DataFrame({
        "CaseID":     SHS_raw[y]["CaseID"].values,
        "WeightD":    SHS_raw[y]["WeightD"].values,
        "OECD_scale": _oecd_scale.values,
    })
    _u["OECD_cap"] = _u["WeightD"] * _u["OECD_scale"]
    Unit_hh_cap[y] = _u

    print(f"\n[{y}] Unit_hh_cap shape: {_u.shape}")
    print(f"[{y}] OECD_scale: mean={_u['OECD_scale'].mean():.3f}, "
          f"min={_u['OECD_scale'].min():.3f}, "
          f"max={_u['OECD_scale'].max():.3f}, "
          f"NaN={_u['OECD_scale'].isna().sum()}")
    print(f"[{y}] OECD_cap (sum): {_u['OECD_cap'].sum():,.0f}")
    print(f"[{y}] OECD_scale distribution (top 10):")
    print(_u["OECD_scale"].value_counts().sort_index().head(10))

# %% Cell 1d - Weighted income decile (Inc_hh_dec) attached to Unit_hh_cap
# Build a 1..10 decile assignment from the continuous HH_TotInc in SHS_raw,
# weighted by WeightD, separately for each year. Each survey row's WeightD is
# treated as its "mass" so the breakpoints are at the cumulative-weight
# 10% / 20% / ... quantiles.
# Cumulative-midpoint assignment is used (each row is placed at its own
# weight midpoint inside the cumulative distribution), which avoids the edge
# bias that a plain floor/cumsum approach would introduce.


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


for y in YEARS:
    _raw = SHS_raw[y]
    _u = Unit_hh_cap[y]
    _inc_dec_by_caseid = pd.Series(
        _weighted_decile(_raw["HH_TotInc"].values, _raw["WeightD"].values, n=10),
        index=_raw["CaseID"].values,
        name="Inc_hh_dec",
    )
    _u["Inc_hh_dec"] = _u["CaseID"].map(_inc_dec_by_caseid)

    # Per-OECD-capita income decile.
    # - Income variable: HH_TotInc / OECD_scale (equivalised income, $ per
    #   OECD-equivalent adult).
    # - Population weight: OECD_cap = WeightD * OECD_scale (number of
    #   equivalent adults represented).
    # Each decile then contains ~10% of OECD-equivalent adults rather than
    # ~10% of households.
    _uhc_by_id = _u.set_index("CaseID")
    _caseid_arr = _raw["CaseID"].values
    _hh_totinc  = _raw["HH_TotInc"].values
    _oecd_scale = _uhc_by_id["OECD_scale"].reindex(_caseid_arr).values
    _oecd_cap   = _uhc_by_id["OECD_cap"].reindex(_caseid_arr).values

    _equiv_income = _hh_totinc / _oecd_scale

    _inc_oecd_dec_by_caseid = pd.Series(
        _weighted_decile(_equiv_income, _oecd_cap, n=10),
        index=_caseid_arr,
        name="Inc_OECDcap_dec",
    )
    _u["Inc_OECDcap_dec"] = _u["CaseID"].map(_inc_oecd_dec_by_caseid)

    print(f"\n[{y}] Inc_hh_dec weighted-population per decile (WeightD totals):")
    print(_u.groupby("Inc_hh_dec")["WeightD"].sum().round(0))
    print(f"[{y}] Inc_OECDcap_dec OECD-equivalent-adults per decile (OECD_cap totals):")
    print(_u.groupby("Inc_OECDcap_dec")["OECD_cap"].sum().round(0))

# %% Cell 2 - COICOP aggregation matrices + level switch
# Load two aggregation matrices from each year's COICOP_to_FIGARO_Bridge_<year>.xlsx,
# aggregation tab (YEAR_CONFIG[year]["aggr_sheet"]):
#   COICOP_to_COICOP32 : D25:<last>56 (32 x n), row labels in C25:C56,
#                        column labels (SHS codes) in D8:<last>8.
#   COICOP_to_COICOP12 : D59:<last>70 (12 x n), row labels in C59:C70,
#                        column labels (SHS codes) in D8:<last>8.
# n = 75 for 2019 and 2021 (different code sets), 77 for 2017 (<last> = CB).
# The 32 and 12 category labels are identical in every year, so years can be
# compared at those levels.
# Choose the aggregation level via `coicop_to_coicop_matching` (75, 32, or 12).
# - 75: no aggregation, work with each year's own detailed SHS codes.
# - 32 or 12: aggregate both SHS spends and the EF total column.


def _load_aggr_matrix(year: int, skiprows: int, nrows: int) -> pd.DataFrame:
    """Read one aggregation block: 'nrows' x n_COICOP with labels in col C."""
    cfg = YEAR_CONFIG[year]
    bridge_path = path_for(DATA_DIR, year, "bridge_file")
    last = cfg["last_col"]
    # COICOP column headers from Excel row 8 (cells D8:<last>8).
    col_headers = (
        pd.read_excel(
            bridge_path,
            sheet_name=cfg["aggr_sheet"],
            usecols=f"D:{last}",
            header=None,
            skiprows=7,
            nrows=1,
        )
        .iloc[0]
        .astype(str)
        .str.strip()
        .tolist()
    )
    # Data block: column C = aggregate label, columns D:<last> = 0/1 values.
    df = pd.read_excel(
        bridge_path,
        sheet_name=cfg["aggr_sheet"],
        usecols=f"C:{last}",
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


COICOP_to_COICOP32 = {y: _load_aggr_matrix(y, skiprows=24, nrows=32) for y in YEARS}
COICOP_to_COICOP12 = {y: _load_aggr_matrix(y, skiprows=58, nrows=12) for y in YEARS}
for y in YEARS:
    print(f"[{y}] COICOP_to_COICOP32 shape: {COICOP_to_COICOP32[y].shape}, "
          f"COICOP_to_COICOP12 shape: {COICOP_to_COICOP12[y].shape}")

# === Switch: 75 (no aggregation), 32, or 12 ===
coicop_to_coicop_matching = 12

if coicop_to_coicop_matching not in (75, 32, 12):
    raise ValueError(
        f"coicop_to_coicop_matching must be 75, 32, or 12, got {coicop_to_coicop_matching!r}"
    )

SHS_aggr = {}
EF_tot_aggr = {}
for y in YEARS:
    if coicop_to_coicop_matching == 75:
        _M = None
    elif coicop_to_coicop_matching == 32:
        _M = COICOP_to_COICOP32[y]
    else:
        _M = COICOP_to_COICOP12[y]

    # Aggregate SHS spends (drop CaseID + WeightD for the math, then add CaseID back).
    _shs = SHS_data_Mln_EUR_WGHT[y]
    _shs_vals = _shs.iloc[:, 2:]
    if _M is None:
        SHS_aggr[y] = _shs.copy()
        EF_tot_aggr[y] = EF_COICOP_TJ_indir_dir[y]["EF_tot_COICOP"].copy()
    else:
        # Align mask columns to SHS columns to be safe (both are SHS codes).
        _M_aligned = _M[_shs_vals.columns]
        _aggr_vals = _shs_vals @ _M_aligned.T   # (n_hh, n_aggr)
        _aggr = _aggr_vals.copy()
        _aggr.insert(0, "WeightD", _shs["WeightD"].values)
        _aggr.insert(0, "CaseID", _shs["CaseID"].values)
        SHS_aggr[y] = _aggr
        EF_tot_aggr[y] = _M_aligned @ EF_COICOP_TJ_indir_dir[y]["EF_tot_COICOP"]   # (n_aggr,)
        EF_tot_aggr[y].name = "EF_tot_COICOP"

    print(f"\n[{y}] coicop_to_coicop_matching = {coicop_to_coicop_matching}")
    print(f"[{y}] SHS_aggr shape:    {SHS_aggr[y].shape}")
    print(f"[{y}] EF_tot_aggr shape: {EF_tot_aggr[y].shape}, sum = {EF_tot_aggr[y].sum():,.2f} TJ")

# %% Cell 3 - SHS spend shares per (aggregated) COICOP category
# For each spending column, divide every household's value by the column total
# so each column sums to exactly 1. Operates on the aggregated SHS table from
# Cell 2 (i.e. 75, 32, or 12 categories depending on the switch).
SHS_spend_shares = {}
for y in YEARS:
    _a = SHS_aggr[y]
    _s = _a.iloc[:, 2:].div(_a.iloc[:, 2:].sum(axis=0), axis=1)
    _s.insert(0, "CaseID", _a["CaseID"].values)
    SHS_spend_shares[y] = _s

    col_sums = _s.iloc[:, 1:].sum(axis=0)
    print(f"[{y}] SHS_spend_shares shape: {_s.shape}; column sums (should all be ~1.0): "
          f"min={col_sums.min():.6f}, max={col_sums.max():.6f}")

# %% Cell 4 - Per-household-group energy footprint at the chosen COICOP level
# Each SHS sample row represents a "household group" — one survey household
# standing in for WeightD population households. Three tables are produced
# per year, at the aggregation level set in Cell 2:
#
#   Total_EF_HHgrouped_TJ[year]
#       Total energy footprint (TJ) per COICOP category for each household
#       group. Computed as: SHS_spend_shares * EF_tot_aggr. The row value is
#       the sum of EF for all population households the survey row represents.
#
#   avg_EF_per_hh_TJ[year]
#       TJ per actual household = Total_EF_HHgrouped_TJ / WeightD
#       (values from Unit_hh_cap, matched on CaseID).
#
#   avg_EF_per_OECDcap_TJ[year]
#       TJ per OECD-equivalent adult = Total_EF_HHgrouped_TJ / OECD_cap
#       (values from Unit_hh_cap, matched on CaseID).
#
# All three keep CaseID as the first column. At the 75-COICOP level the
# indir/dir breakdown is also produced.

Total_EF_HHgrouped_TJ = {}
avg_EF_per_hh_TJ = {}
avg_EF_per_OECDcap_TJ = {}
Total_aggr_EF_hh_wght = {}
avg_EF_per_hh = {}
avg_EF_perOECD_cap = {}
Dir_EF_hh_TJ = {}
Indir_EF_hh_TJ = {}

for y in YEARS:
    _shares = SHS_spend_shares[y].iloc[:, 1:]   # drop CaseID for the math

    # Per-group total (one row = one survey household, value already weighted).
    _total = _shares.mul(EF_tot_aggr[y], axis=1)
    _total.insert(0, "CaseID", SHS_spend_shares[y]["CaseID"].values)

    # Align WeightD and OECD_cap to CaseID for safety.
    _uhc = Unit_hh_cap[y].set_index("CaseID")
    _w = _uhc["WeightD"].reindex(_total["CaseID"]).values
    _c = _uhc["OECD_cap"].reindex(_total["CaseID"]).values

    _per_hh = _total.iloc[:, 1:].div(_w, axis=0)
    _per_hh.insert(0, "CaseID", _total["CaseID"].values)

    _per_cap = _total.iloc[:, 1:].div(_c, axis=0)
    _per_cap.insert(0, "CaseID", _total["CaseID"].values)

    Total_EF_HHgrouped_TJ[y] = _total
    avg_EF_per_hh_TJ[y] = _per_hh
    avg_EF_per_OECDcap_TJ[y] = _per_cap

    # Per-row totals across all COICOP categories — one value per survey household.
    #   Total_aggr_EF_hh_wght = total TJ for the WeightD households the row represents
    #                           (sum across COICOP of Total_EF_HHgrouped_TJ).
    #                           A sum, not an average. Population totals sum to the
    #                           national EF total.
    #   avg_EF_per_hh         = total TJ per actual household for that segment
    #                           (sum across COICOP of avg_EF_per_hh_TJ).
    #   avg_EF_perOECD_cap    = total TJ per OECD-equivalent adult for that segment
    #                           (sum across COICOP of avg_EF_per_OECDcap_TJ).
    Total_aggr_EF_hh_wght[y] = pd.DataFrame({
        "CaseID":                _total["CaseID"].values,
        "Total_aggr_EF_hh_wght": _total.iloc[:, 1:].sum(axis=1).values,
    })
    avg_EF_per_hh[y] = pd.DataFrame({
        "CaseID":        _per_hh["CaseID"].values,
        "avg_EF_per_hh": _per_hh.iloc[:, 1:].sum(axis=1).values,
    })
    avg_EF_perOECD_cap[y] = pd.DataFrame({
        "CaseID":             _per_cap["CaseID"].values,
        "avg_EF_perOECD_cap": _per_cap.iloc[:, 1:].sum(axis=1).values,
    })

    print(f"\n[{y}] Total_EF_HHgrouped_TJ shape: {_total.shape}, "
          f"total = {_total.iloc[:, 1:].values.sum():,.2f} TJ")
    print(f"[{y}] avg_EF_per_hh_TJ      shape: {_per_hh.shape}")
    print(f"[{y}] avg_EF_per_OECDcap_TJ shape: {_per_cap.shape}")
    print(f"[{y}] avg_EF_per_hh      mean = {avg_EF_per_hh[y]['avg_EF_per_hh'].mean():.4f} TJ/hh")
    print(f"[{y}] avg_EF_perOECD_cap mean = {avg_EF_perOECD_cap[y]['avg_EF_perOECD_cap'].mean():.4f} TJ/OECD-cap")

    if coicop_to_coicop_matching == 75:
        _dir = _shares.mul(EF_COICOP_TJ_indir_dir[y]["EF_dir_COICOP"],   axis=1)
        _ind = _shares.mul(EF_COICOP_TJ_indir_dir[y]["EF_indir_COICOP"], axis=1)
        _dir.insert(0, "CaseID", SHS_spend_shares[y]["CaseID"].values)
        _ind.insert(0, "CaseID", SHS_spend_shares[y]["CaseID"].values)
        Dir_EF_hh_TJ[y], Indir_EF_hh_TJ[y] = _dir, _ind
        print(f"[{y}] Dir_EF_hh_TJ   total = {_dir.iloc[:, 1:].values.sum():,.2f} TJ")
        print(f"[{y}] Indir_EF_hh_TJ total = {_ind.iloc[:, 1:].values.sum():,.2f} TJ")

# %% Cell 5 - Stacked bar chart: per OECD-cap and per household EF, segmented by COICOP category
# Two bars (per OECD-cap, per household) per year, one panel per year. Each
# bar is stacked by COICOP category, largest contributors at the bottom. Long
# category labels are truncated for the legend.
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


def _short(label, n=25):
    s = str(label).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _plot_stacked_by_year(tables, title, ylabel, xlabel, rot=0, panel_width=None, height=7):
    """
    One stacked-bar panel per year (shared y axis). `tables` maps year ->
    DataFrame (rows = bars, columns = COICOP categories).

    When every year has the same categories (12 or 32 level), the columns are
    put in one order (largest total across years at the bottom) so a category
    has the same colour in every panel, and one legend is drawn. At the 75
    level the code sets differ per year, so each panel keeps its own order
    and legend.
    """
    years = list(tables)
    same_cols = all(set(t.columns) == set(tables[years[0]].columns) for t in tables.values())
    if same_cols:
        order = sum(t.sum(axis=0) for t in tables.values()).sort_values(ascending=False).index
        tables = {y: t[order] for y, t in tables.items()}
    else:
        tables = {y: t[t.sum(axis=0).sort_values(ascending=False).index] for y, t in tables.items()}

    if panel_width is None:
        panel_width = max(4.0, 0.9 * max(len(t.index) for t in tables.values()))
    fig, axes = plt.subplots(
        1, len(years), figsize=(panel_width * len(years) + 4, height),
        sharey=True, squeeze=False,
    )
    axes = axes[0]
    for ax, y in zip(axes, years):
        tables[y].rename(columns=_short).plot.bar(
            ax=ax, stacked=True, color=_COLORBLIND_PALETTE, edgecolor="white",
            linewidth=0.4, legend=not same_cols,
        )
        ax.set_title(str(y))
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="x", labelrotation=rot)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if not same_cols:
            ax.legend(title="COICOP category", fontsize=6, loc="upper left")
        _add_hatching(ax)
    axes[0].set_ylabel(ylabel)
    if same_cols:
        axes[-1].legend(title="COICOP category", bbox_to_anchor=(1.02, 1),
                        loc="upper left", fontsize=8)
        _add_hatching(axes[-1])
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


avg_EF_weighted_OECDcap_per_COICOP = {}
avg_EF_weighted_hh_per_COICOP = {}
_plot_tables = {}
for y in YEARS:
    _sum_OECD_cap = Unit_hh_cap[y]["OECD_cap"].sum()
    _sum_WeightD  = Unit_hh_cap[y]["WeightD"].sum()

    avg_EF_weighted_OECDcap_per_COICOP[y] = (EF_tot_aggr[y] / _sum_OECD_cap).sort_values(ascending=False)
    avg_EF_weighted_hh_per_COICOP[y] = (
        EF_tot_aggr[y].reindex(avg_EF_weighted_OECDcap_per_COICOP[y].index) / _sum_WeightD
    )

    print(f"[{y}] sum(OECD_cap) = {_sum_OECD_cap:,.0f}, sum(WeightD) = {_sum_WeightD:,.0f}")
    print(f"[{y}] EF total: {EF_tot_aggr[y].sum():,.0f} TJ "
          f"-> {avg_EF_weighted_OECDcap_per_COICOP[y].sum():.3f} TJ/OECD-cap, "
          f"{avg_EF_weighted_hh_per_COICOP[y].sum():.3f} TJ/hh")

    _plot_tables[y] = pd.DataFrame({
        "Per OECD-cap": avg_EF_weighted_OECDcap_per_COICOP[y],
        "Per household": avg_EF_weighted_hh_per_COICOP[y],
    }).T

_plot_stacked_by_year(
    _plot_tables,
    title=f"Energy footprint by COICOP-{coicop_to_coicop_matching} category",
    ylabel="TJ per unit", xlabel="", rot=0, panel_width=4,
)

# %% Cell 6 - Stacked bar: average household EF by province (PROV)
# Group households by Prov, then compute the weight-average TJ per household
# per COICOP category:
#   avg_TJ_per_hh[prov, cat] = sum(Total_EF_hh_TJ[h, cat] for h in prov)
#                              / sum(WeightD[h] for h in prov)
# Each bar = one province, stacked by COICOP category. The same `_short`
# label-truncation helper from Cell 5 is reused.

_PROV_NAMES = {
    "10": "NL", "11": "PE", "12": "NS", "13": "NB",
    "14": "Atlantic",      # 2017 PUMF bundles NL+PE+NS+NB
    "24": "QC", "35": "ON",
    "46": "MB", "47": "SK", "48": "AB", "59": "BC",
    "63": "Territories",   # YT+NT+NU bundled in PUMF for confidentiality
}

EF_per_hh_by_PROV = {}
for y in YEARS:
    _total_df = Total_EF_HHgrouped_TJ[y]
    _meta_prov = SHS_raw[y][["CaseID", "Prov", "WeightD"]]
    _merged = _total_df.merge(_meta_prov, on="CaseID", how="inner")

    _cat_cols = [c for c in _total_df.columns if c != "CaseID"]
    _grp = _merged.groupby("Prov")
    _t = _grp[_cat_cols].sum().div(_grp["WeightD"].sum(), axis=0)

    # Pretty province labels (fall back to raw code if unknown).
    _t.index = [_PROV_NAMES.get(str(p).strip(), str(p).strip()) for p in _t.index]
    EF_per_hh_by_PROV[y] = _t

    print(f"[{y}] EF_per_hh_by_PROV shape: {_t.shape}")
    print(_t.sum(axis=1).round(4))

_plot_stacked_by_year(
    EF_per_hh_by_PROV,
    title=f"Average household EF by province (COICOP-{coicop_to_coicop_matching})",
    ylabel="TJ per household", xlabel="Province", rot=0,
)

# %% Cell 7 - Stacked bar: average per-OECDcap EF by household type (HHType6)
# Group households by HHType6, then compute the OECD_cap-weighted mean of
# EF_per_OECDcap_TJ per COICOP category:
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


def _ocap_weighted_mean_by(year, col):
    """OECD_cap-weighted mean of avg_EF_per_OECDcap_TJ per category of `col`."""
    per_cap_df = avg_EF_per_OECDcap_TJ[year]
    meta = SHS_raw[year][["CaseID", col]].merge(
        Unit_hh_cap[year][["CaseID", "OECD_cap"]], on="CaseID", how="inner"
    )
    merged = per_cap_df.merge(meta, on="CaseID", how="inner")
    cat_cols = [c for c in per_cap_df.columns if c != "CaseID"]
    weighted = merged[cat_cols].multiply(merged["OECD_cap"], axis=0)
    weighted[col] = merged[col].values
    return (
        weighted.groupby(col).sum()
        .div(merged.groupby(col)["OECD_cap"].sum(), axis=0)
    )


EF_per_OECDcap_by_HHType6 = {}
for y in YEARS:
    _t = _ocap_weighted_mean_by(y, "HHType6")
    _t.index = [_HHTYPE_NAMES.get(int(t), str(int(t))) for t in _t.index]
    EF_per_OECDcap_by_HHType6[y] = _t
    print(f"[{y}] EF_per_OECDcap_by_HHType6 shape: {_t.shape}")
    print(_t.sum(axis=1).round(4))

_plot_stacked_by_year(
    EF_per_OECDcap_by_HHType6,
    title=f"Average per-OECD-cap EF by household type (COICOP-{coicop_to_coicop_matching})",
    ylabel="TJ per OECD-cap", xlabel="Household type", rot=15,
)

# %% Cell 8 - Stacked bar: average per-OECDcap EF by reference-person age group
# Same OECD_cap-weighted-mean method as Cell 7, but grouped by RP_AgeGrp
# (age of the household reference person). RP_AgeGrp is stored as a string
# code "01"..."06" (harmonised across years in year_config).

_RP_AGE_NAMES = {
    "01": "<30",
    "02": "30-39",
    "03": "40-54",
    "04": "55-64",
    "05": "65-74",
    "06": "75+",
}

EF_per_OECDcap_by_RPAgeGrp = {}
for y in YEARS:
    # Preserve numeric order on the x-axis.
    _t = _ocap_weighted_mean_by(y, "RP_AgeGrp").sort_index()
    _t.index = [_RP_AGE_NAMES.get(str(a).strip(), str(a).strip()) for a in _t.index]
    EF_per_OECDcap_by_RPAgeGrp[y] = _t
    print(f"[{y}] EF_per_OECDcap_by_RPAgeGrp shape: {_t.shape}")
    print(_t.sum(axis=1).round(4))

_plot_stacked_by_year(
    EF_per_OECDcap_by_RPAgeGrp,
    title=(f"Average per-OECD-cap EF by reference-person age group "
           f"(COICOP-{coicop_to_coicop_matching})"),
    ylabel="TJ per OECD-cap", xlabel="Age of reference person", rot=0,
)

# %% Cell 9 - Per-OECD-cap EF by additional SHS variables (one chart per var)
# Same OECD_cap-weighted-mean method as Cells 7 / 8, swept through several
# socio-demographic variables. Each chart has one panel per year.
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

_results_by_var = {}
for _col, (_labels, _xlabel, _sort_idx) in _VAR_SPECS.items():
    _tables = {}
    for y in YEARS:
        if _col not in SHS_raw[y].columns:
            print(f"Skipping {_col} for {y} (not in SHS_raw[{y}])")
            continue
        _t = _ocap_weighted_mean_by(y, _col)
        if _sort_idx:
            _t = _t.sort_index()
        # Pretty x-axis labels.
        _t.index = [_labels.get(str(v).strip(), str(v).strip()) for v in _t.index]
        _tables[y] = _t
    if not _tables:
        continue
    _results_by_var[_col] = _tables
    _plot_stacked_by_year(
        _tables,
        title=f"Avg per-OECD-cap EF by {_xlabel.lower()} (COICOP-{coicop_to_coicop_matching})",
        ylabel="TJ per OECD-cap", xlabel=_xlabel, rot=15, height=6,
    )

# %% Cell 10 - Stacked bar: average household EF by weighted income decile
# Use Total_EF_HHgrouped_TJ (which holds the energy footprint of every
# population household represented by each survey row) and aggregate by
# Inc_hh_dec (built in Cell 1d, per year). Per decile:
#   avg_TJ_per_hh[dec, c] = sum_h(Total_EF[h, c]) / sum_h(WeightD[h])
# Each bar = one income decile (D1 = lowest income, D10 = highest), stacked
# by COICOP category, largest contributors at the bottom.

EF_per_hh_by_IncDec = {}
mean_inc_per_IncDec = {}
for y in YEARS:
    _total_df = Total_EF_HHgrouped_TJ[y]
    _meta_d = Unit_hh_cap[y][["CaseID", "Inc_hh_dec", "WeightD"]]
    _merged = _total_df.merge(_meta_d, on="CaseID", how="inner")
    _cat_cols = [c for c in _total_df.columns if c != "CaseID"]

    _grp = _merged.groupby("Inc_hh_dec")
    _t = _grp[_cat_cols].sum().div(_grp["WeightD"].sum(), axis=0).sort_index()

    # Weighted mean income (HH_TotInc, CAD) per decile, for x-tick annotation:
    #   mean_inc[dec] = sum_h(HH_TotInc[h] * WeightD[h]) / sum_h(WeightD[h])
    _inc_meta = SHS_raw[y][["CaseID", "HH_TotInc"]].merge(
        Unit_hh_cap[y][["CaseID", "Inc_hh_dec", "WeightD"]], on="CaseID", how="inner"
    )
    mean_inc_per_IncDec[y] = (
        (_inc_meta["HH_TotInc"] * _inc_meta["WeightD"])
        .groupby(_inc_meta["Inc_hh_dec"]).sum()
        .div(_inc_meta.groupby("Inc_hh_dec")["WeightD"].sum())
    ).sort_index()

    # Two-line x-tick labels: "D1\n$24k" etc.
    _t.index = [f"D{int(d)}\n${mean_inc_per_IncDec[y].loc[d] / 1000:.0f}k" for d in _t.index]
    EF_per_hh_by_IncDec[y] = _t

    print(f"[{y}] Weighted mean HH_TotInc per decile (CAD):")
    print(mean_inc_per_IncDec[y].round(0))

_plot_stacked_by_year(
    EF_per_hh_by_IncDec,
    title=f"Average household EF by income decile (COICOP-{coicop_to_coicop_matching})",
    ylabel="TJ per household",
    xlabel="Income decile (D1 = lowest) — mean HH income below",
    rot=0,
)

# %% Cell 11 - Stacked bar: average TJ per OECD-cap by per-OECD-cap income decile
# Uses Inc_OECDcap_dec (built in Cell 1d) — deciles of equivalised income
# (HH_TotInc / OECD_scale) weighted by OECD_cap. Each decile holds ~10% of
# OECD-equivalent adults.
# Aggregation:
#   avg_TJ_per_cap[dec, c] = sum_h(Total_EF[h, c]) / sum_h(OECD_cap[h])
# X-tick labels show the weighted-mean equivalised income per decile.

EF_per_OECDcap_by_IncOECDDec = {}
mean_eq_inc_per_dec = {}
for y in YEARS:
    _total_df = Total_EF_HHgrouped_TJ[y]
    _meta_o = Unit_hh_cap[y][["CaseID", "Inc_OECDcap_dec", "OECD_cap"]]
    _merged = _total_df.merge(_meta_o, on="CaseID", how="inner")
    _cat_cols = [c for c in _total_df.columns if c != "CaseID"]

    _grp = _merged.groupby("Inc_OECDcap_dec")
    _t = _grp[_cat_cols].sum().div(_grp["OECD_cap"].sum(), axis=0).sort_index()

    # Weighted-mean equivalised income (CAD per OECD-cap) per decile:
    #   mean_eq_inc[dec] = sum_h((HH_TotInc/OECD_scale) * OECD_cap) / sum_h(OECD_cap)
    _uhc = Unit_hh_cap[y].set_index("CaseID")
    _eq_inc_meta = pd.DataFrame({
        "CaseID":   SHS_raw[y]["CaseID"].values,
        "EqIncome": SHS_raw[y]["HH_TotInc"].values
                    / _uhc["OECD_scale"].reindex(SHS_raw[y]["CaseID"].values).values,
    }).merge(Unit_hh_cap[y][["CaseID", "Inc_OECDcap_dec", "OECD_cap"]], on="CaseID", how="inner")

    mean_eq_inc_per_dec[y] = (
        (_eq_inc_meta["EqIncome"] * _eq_inc_meta["OECD_cap"])
        .groupby(_eq_inc_meta["Inc_OECDcap_dec"]).sum()
        .div(_eq_inc_meta.groupby("Inc_OECDcap_dec")["OECD_cap"].sum())
    ).sort_index()

    _t.index = [f"D{int(d)}\n${mean_eq_inc_per_dec[y].loc[d] / 1000:.0f}k" for d in _t.index]
    EF_per_OECDcap_by_IncOECDDec[y] = _t

    print(f"[{y}] Weighted-mean equivalised income (CAD / OECD-cap) per decile:")
    print(mean_eq_inc_per_dec[y].round(0))

_plot_stacked_by_year(
    EF_per_OECDcap_by_IncOECDDec,
    title=f"Average EF per OECD-cap by per-OECD-cap income decile (COICOP-{coicop_to_coicop_matching})",
    ylabel="TJ per OECD-cap",
    xlabel="Per-OECD-cap income decile — mean equivalised income below",
    rot=0,
)

# %% Cell 12 - Assemble data for statistical analysis
# Build a single DataFrame `var_for_stat_analysis` with one row per survey
# household per year (a `year` column identifies the survey), joining
# everything needed for downstream regression / inequality work. Save to disk
# so a separate notebook can `pd.read_pickle(...)` it without re-running this
# script. CaseID is only unique within a year: use (year, CaseID) as the key.
# At the 75 level the COICOP columns differ between years; codes that do not
# exist in a year are NaN for that year's rows.
import re

_socio_vars = [
    "Prov", "HHType6", "RP_AgeGrp",
    "RP_Gender", "RP_MarStat", "RP_Educ",
    "DwelTyp", "Tenure", "NumBedr",
    "SecResYN", "OthPropYN", "InternetYN", "VehicleYN",
    "HH_MajIncSrc", "HH_TotInc",
]


def _sanitize_colname(s, prefix="C", max_len=50):
    """Convert a COICOP-like label into a safe Python identifier."""
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if s and s[0].isdigit():
        s = f"{prefix}_{s}"
    return (s[:max_len] or prefix).rstrip("_")


def _safe_labels_for(labels):
    """Sanitized, de-duplicated column names for a list of COICOP labels."""
    safe = [_sanitize_colname(c) for c in labels]
    if len(set(safe)) != len(safe):
        # Disambiguate any duplicates by appending a suffix.
        seen = {}
        deduped = []
        for s in safe:
            n = seen.get(s, 0)
            seen[s] = n + 1
            deduped.append(s if n == 0 else f"{s}_{n}")
        safe = deduped
    return safe


coicop_label_map = {}
_frames = []
for y in YEARS:
    _existing_socio = [c for c in _socio_vars if c in SHS_raw[y].columns]
    _missing_socio  = [c for c in _socio_vars if c not in SHS_raw[y].columns]
    if _missing_socio:
        print(f"Note: missing in SHS_raw[{y}], skipped: {_missing_socio}")

    # 1. Start with Unit_hh_cap (CaseID, WeightD, OECD_scale, OECD_cap, deciles)
    _df = Unit_hh_cap[y].copy()
    _df.insert(0, "year", y)

    # 2. Join socio-demographic + income vars from SHS_raw
    _df = _df.merge(SHS_raw[y][["CaseID"] + _existing_socio], on="CaseID", how="left")

    # 3. Join EF row-totals (each is CaseID + one column)
    _df = _df.merge(Total_aggr_EF_hh_wght[y], on="CaseID", how="left")
    _df = _df.merge(avg_EF_per_hh[y],         on="CaseID", how="left")
    _df = _df.merge(avg_EF_perOECD_cap[y],    on="CaseID", how="left")

    # 4. Join per-COICOP per-hh and per-OECD-cap tables, prefixing column names
    #    so per-hh and per-cap variants of the same COICOP don't collide. Each
    #    COICOP label is also converted into a safe Python identifier so the
    #    columns work with statsmodels.formula and similar tools.
    _per_hh  = avg_EF_per_hh_TJ[y].copy()
    _per_cap = avg_EF_per_OECDcap_TJ[y].copy()
    _orig_labels = list(_per_hh.columns[1:])
    _safe = _safe_labels_for(_orig_labels)

    _hh_cols, _cap_cols = [], []
    for orig, safe in zip(_orig_labels, _safe):
        hh_name  = f"avg_EF_per_hh__{safe}"
        cap_name = f"avg_EF_per_cap__{safe}"
        _hh_cols.append(hh_name)
        _cap_cols.append(cap_name)
        coicop_label_map[hh_name]  = orig
        coicop_label_map[cap_name] = orig

    _per_hh.columns  = ["CaseID"] + _hh_cols
    _per_cap.columns = ["CaseID"] + _cap_cols
    _df = _df.merge(_per_hh,  on="CaseID", how="left")
    _df = _df.merge(_per_cap, on="CaseID", how="left")
    _frames.append(_df)

var_for_stat_analysis = pd.concat(_frames, ignore_index=True, sort=False)

print(f"var_for_stat_analysis shape: {var_for_stat_analysis.shape}")
print("Rows per year:")
print(var_for_stat_analysis["year"].value_counts().sort_index())
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
