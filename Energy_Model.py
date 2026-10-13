"""
Energy Model — main script.

Define all file paths and imports once in the Setup cell, choose the survey
years in YEARS (any one, two or all three of 2017, 2019, 2021), then run the
numbered cells in order with Ctrl+Enter.

The model itself (Cells 1-10 of the original single-year script) lives in
`energy_model_year(year)`, which runs the whole pipeline for one year and
returns its results as a dict. The results for all chosen years are stored
in `EM`, keyed by year:

    EM[2021]["e_COICOP_TJperMlnEur"]
    EM[2017]["EF_COICOP_TJ_indir_dir"]

Year-specific inputs (SHS file, bridge file and sheet names, IEA file,
exchange rate) come from year_config.YEAR_CONFIG.
"""

# %% Setup
import os
import numpy as np
import pandas as pd

from Figaro_economic_data import figaro_economic_data
from SHS_data import shs_data
from NACE_COICOP_mask import nace_to_coicop_mask
from RAS_balancing import ras_balancing
from Indir_E_e import indir_E_e
from year_config import YEAR_CONFIG, check_years, path_for

# --- File paths and config (defaults below; can be overridden by the caller ---
# --- when running this script via runpy.run_path(init_globals={...})). ---
# When run directly, the defaults apply. When run from Household_EF.py (or any
# other driver that injects globals), those injected values take precedence.
DATA_DIR = globals().get(
    "DATA_DIR",
    r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents\additional_projects\Canada_hh_energy\Data",
)
FIGARO_FOLDER = globals().get(
    "FIGARO_FOLDER",
    r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents - 403001-01 Utvikling og vedlikehold klimakost\General\Utslippsfaktorer\Kryssløpsanalyse\FIGARO data",
)
COUNTRY_CODE = globals().get("COUNTRY_CODE", "CA")
HH_CATEGORY = globals().get("HH_CATEGORY", "P3_S14")
FD_CATEGORIES = globals().get(
    "FD_CATEGORIES", ["P3_S13", "P3_S14", "P3_S15", "P51G", "P5M"]
)

# === Years to run: any subset of [2017, 2019, 2021] ===
# (Household_EF.py passes its own list via _YEARS_OVERRIDE; a separate name is
# used so that editing the list below always takes effect in an open console.)
YEARS = check_years(globals().get("_YEARS_OVERRIDE") or [2017, 2019, 2021])
print(f"Energy Model years: {YEARS}")

# Shared across years.
NACE_IEA_PATH = os.path.join(DATA_DIR, "IEA_NACE_mapping.xlsx")


# %% Cell 0 — NACE-to-IEA bridge (same for every year)
# NACE-to-IEA bridge matrix (many-to-many).
# File: IEA_NACE_mapping.xlsx, tab: NACE_IEA_Bridge.
# Row 1 = column names; columns A and B = NACE row labels; D2:Z63 = 0/1 matrix.
NACE_to_IEA_mask = pd.read_excel(
    NACE_IEA_PATH,
    sheet_name="NACE_IEA_Bridge",
    usecols="A:B,D:Z",
    header=0,
    nrows=62,
)
print(f"NACE_to_IEA_mask shape:       {NACE_to_IEA_mask.shape}")


# %% Model for one year
def energy_model_year(year):
    """
    Run the full Energy Model for one survey year and return its results.

    Steps are the numbered cells of the original 2019 script; only the input
    files, FIGARO year and exchange rate change with `year`.
    """
    cfg = YEAR_CONFIG[year]
    BRIDGE_PATH = path_for(DATA_DIR, year, "bridge_file")
    IEA_PATH = path_for(DATA_DIR, year, "iea_file")

    # ---- Cell 1 — Load FIGARO economic IO data ----
    # The household FD vector for COUNTRY_CODE is also computed inside the
    # function and returned under the key f"{COUNTRY_CODE}_Y_hh".
    result = figaro_economic_data(
        figaro_folder=FIGARO_FOLDER,
        first_year=year,
        last_year=year,
        edition_economic_data=2501,
        country_code=COUNTRY_CODE,
        hh_category=HH_CATEGORY,
        fd_categories=FD_CATEGORIES,
    )
    X = result["X"]
    Y = result["Y"]
    Z = result["Z"]
    meta = result["meta"]   # access as meta.regions, meta.sectors, meta.FDcategories

    regions = meta.regions
    sectors = meta.sectors

    # Pull out the country household FD vector. Key in the result dict follows
    # the country code (e.g. "CA_Y_hh"), but we use a generic local name so the
    # rest of the code is country-agnostic.
    Y_hh = result[f"{COUNTRY_CODE}_Y_hh"]

    # ---- Cell 2 — Load IEA energy data (from IEA_data_<year>.xlsx) ----
    # IEA_FC_tot: total final consumption.
    # Includes road indirect and international bunkers.
    IEA_FC_tot = pd.read_excel(
        IEA_PATH,
        sheet_name="IEA_to_Figaro_data",
        usecols="A:F,M",
    )
    print(f"IEA_FC_tot shape:             {IEA_FC_tot.shape}")

    # ---- Cell 3 — Indirect energy intensity + household indirect footprint ----
    m = result["m"]   # 46 regions
    n = result["n"]   # 62 sectors

    indir = indir_E_e(
        Z=Z, X=X, Y_hh=Y_hh,
        sectors=sectors,
        NACE_to_IEA_mask=NACE_to_IEA_mask,
        IEA_FC_tot=IEA_FC_tot,
        m=m, n=n,
    )
    E_TJ = indir["E_TJ"]
    e_NACE = indir["e_NACE"]
    hh_EF_NACE_indir = indir["hh_EF_NACE_indir"]

    # ---- Cell 4 — Load SHS household spends ----
    # The loader returns a dict with two flavours of the same data, both with
    # expenditure values multiplied by the household weight (WeightD):
    #   - CAD_WGHT     : weighted spends in CAD per household
    #   - Mln_EUR_WGHT : same spends converted to million EUR using the
    #                    year's exchange rate (YEAR_CONFIG[year]["exchange_rate"]),
    #                    i.e. CAD value ÷ rate ÷ 1,000,000.
    # Each row is one household; columns are CaseID, WeightD, then the
    # expenditures listed in the year's bridge (75 for 2019/2021, 77 for 2017).
    shs_dict = shs_data(data_dir=DATA_DIR, year=year)
    SHS_data_CAD_WGHT     = shs_dict["CAD_WGHT"]
    SHS_data_Mln_EUR_WGHT = shs_dict["Mln_EUR_WGHT"]

    # Strip the "_wght" suffix from expenditure columns so we have clean COICOP
    # codes throughout the rest of the pipeline. CaseID and WeightD stay as-is.
    SHS_data_CAD_WGHT.columns     = [c.removesuffix("_wght") for c in SHS_data_CAD_WGHT.columns]
    SHS_data_Mln_EUR_WGHT.columns = [c.removesuffix("_wght") for c in SHS_data_Mln_EUR_WGHT.columns]

    # COICOP_code: canonical list of the COICOP codes (no _wght suffix).
    COICOP_code = SHS_data_CAD_WGHT.columns[2:].tolist()   # skip CaseID + WeightD
    print(f"COICOP_code ({len(COICOP_code)} categories): first 5 = {COICOP_code[:5]}")

    # ---- Cell 5 — Load NACE-to-COICOP bridge matrix ----
    NACE_to_COICOP_mask = nace_to_coicop_mask(
        bridge_path=BRIDGE_PATH,
        sectors=sectors,
        shs_data=SHS_data_CAD_WGHT,    # column names only — values not used
        sheet_name=cfg["bridge_sheet"],
        last_col=cfg["last_col"],
    )

    # ---- Cell 6 — RAS-balance SHS spends against NACE sectors, then distribute
    # the household indirect energy footprint across COICOP categories ----

    # SHS expenditure totals (million EUR) per COICOP. Skip CaseID and WeightD
    # (first two columns) and sum across all households.
    SHS_COICOP_totals = SHS_data_Mln_EUR_WGHT.iloc[:, 2:].sum().to_numpy(dtype=float)
    print(f"SHS_COICOP_totals shape: {SHS_COICOP_totals.shape}, "
          f"sum (mln EUR) = {SHS_COICOP_totals.sum():,.2f}")

    # RAS balancing — feed the already-EUR-converted SHS values.
    ras_result = ras_balancing(
        nace_coicop_mask=NACE_to_COICOP_mask,
        shs_eur=SHS_data_Mln_EUR_WGHT,
        Y=Y,
        regions=regions,
        m=result["m"],
        n=result["n"],
        k=result["k"],
        year_idx=0,
        country_code=COUNTRY_CODE,
        fd_categories=FD_CATEGORIES,
        hh_category=HH_CATEGORY,
        mode="rows",         # row-only distribution (columns are best-effort)
    )
    RAS_NACE_COICOP_spends = ras_result["balanced"]   # units: million EUR
    RAS_nace_to_coicop_shares = ras_result["RAS_nace_to_coicop_shares"]
    print(f"\nBalanced matrix shape: {RAS_NACE_COICOP_spends.shape}")
    print(f"Converged: {ras_result['converged']}, iterations: {ras_result['iterations']}")

    # Aggregate household energy footprint to 62 NACE sectors.
    # hh_EF_NACE_indir has shape (m*n,) organised as country blocks of n=62 sectors.
    # Sum across all 46 countries for each sector (MATLAB equivalent: 1:62:end).
    n_sec = result["n"]
    hh_EF_NACE = pd.Series(
        np.array([hh_EF_NACE_indir[s::n_sec].sum() for s in range(n_sec)]),
        index=sectors.iloc[:, 1].astype(str).str.strip().tolist(),
        name="hh_EF_NACE",
    )
    print(f"hh_EF_NACE shape: {hh_EF_NACE.shape}, total = {hh_EF_NACE.sum():,.2f} TJ")

    # Distribute the per-NACE energy footprint across COICOP categories using
    # RAS_nace_to_coicop_shares. Each row of the (62 x n_COICOP) shares matrix
    # is multiplied by the corresponding scalar from hh_EF_NACE.
    EF_indir_COICOP = RAS_nace_to_coicop_shares.multiply(hh_EF_NACE, axis=0)
    print(f"EF_indir_COICOP shape: {EF_indir_COICOP.shape}, "
          f"total = {EF_indir_COICOP.values.sum():,.2f} TJ")

    # Indirect Energy intensity per COICOP category:
    #   e_COICOP[j] = sum_i EF_indir_COICOP[i, j]  /  sum_i RAS_NACE_COICOP_spends[i, j]
    #   units: TJ per million EUR
    e_COICOP_indir = EF_indir_COICOP.sum(axis=0) / RAS_NACE_COICOP_spends.sum(axis=0)
    e_COICOP_indir.name = "e_COICOP_indir"
    print(f"e_COICOP_indir shape: {e_COICOP_indir.shape} (units: TJ / million EUR)")

    # ---- Cell 9 — Direct energy intensity per COICOP ----
    # Import direct energy values per COICOP from IEA_data_<year>.xlsx,
    # tab "Resid_Road_split_to_COICOP", cells A28:C33.
    # Row 28 is the header; rows 29-33 are 5 data rows.
    # Layout: column A = description, column B = COICOP code, column C = value (TJ).
    dir_E_road_resid = pd.read_excel(
        IEA_PATH,
        sheet_name="Resid_Road_split_to_COICOP",
        usecols="A:C",
        header=0,
        skiprows=27,
        nrows=5,
    )
    print(f"dir_E_road_resid shape: {dir_E_road_resid.shape}")
    print(dir_E_road_resid)

    # e_COICOP_dir[code] = direct_energy_value / column_sum_in_RAS_NACE_COICOP_spends
    # Match COICOP codes (column 2 = index 1) to RAS_NACE_COICOP_spends column names.
    e_COICOP_dir = pd.Series(dtype=float, name="e_COICOP_dir")
    for _, row in dir_E_road_resid.iterrows():
        code = str(row.iloc[1]).strip()       # COICOP code
        val = float(row.iloc[2])              # direct energy value (TJ)
        if code not in RAS_NACE_COICOP_spends.columns:
            raise ValueError(
                f"COICOP code {code!r} not found in RAS_NACE_COICOP_spends columns"
            )
        col_sum = RAS_NACE_COICOP_spends[code].sum()
        e_COICOP_dir[code] = val / col_sum

    print(f"\ne_COICOP_dir ({len(e_COICOP_dir)} categories):")
    print(e_COICOP_dir)

    # ---- Cell 10 — Combine indirect + direct COICOP intensities ----
    # Build a DataFrame with two columns ('indir', 'dir'). Categories that don't
    # appear in e_COICOP_dir get 0 in the 'dir' column.
    e_COICOP_TJperMlnEur = pd.DataFrame({
        "indir": e_COICOP_indir,
        "dir":   e_COICOP_dir.reindex(e_COICOP_indir.index, fill_value=0.0),
    })
    print(f"e_COICOP_TJperMlnEur shape: {e_COICOP_TJperMlnEur.shape}")
    print(e_COICOP_TJperMlnEur.head())

    # Energy footprint (TJ) per COICOP — same logic as above but with absolute
    # TJ values instead of intensities.
    #   EF_indir_COICOP = column sums of EF_indir_COICOP (TJ)
    #   EF_dir_COICOP   = direct TJ values from dir_E_road_resid, matched to COICOP
    #                 codes; categories not in dir_E_road_resid get 0.
    EF_dir_COICOP = pd.Series(
        {str(row.iloc[1]).strip(): float(row.iloc[2]) for _, row in dir_E_road_resid.iterrows()},
        name="EF_dir_COICOP",
    )
    EF_COICOP_TJ_indir_dir = pd.DataFrame({
        "EF_indir_COICOP": EF_indir_COICOP.sum(axis=0),
        "EF_dir_COICOP":   EF_dir_COICOP.reindex(EF_indir_COICOP.columns, fill_value=0.0),
    })
    print(f"\nEF_COICOP_TJ_indir_dir shape: {EF_COICOP_TJ_indir_dir.shape} (units: TJ)")
    print(EF_COICOP_TJ_indir_dir.head())

    # Keep everything small enough to cache; the big FIGARO matrices (Z, Y, X)
    # and the Leontief inverse are dropped after each year.
    return {
        "year": year,
        "regions": regions,
        "sectors": sectors,
        "Y_hh": Y_hh,
        "IEA_FC_tot": IEA_FC_tot,
        "E_TJ": E_TJ,
        "e_NACE": e_NACE,
        "hh_EF_NACE_indir": hh_EF_NACE_indir,
        "SHS_data_CAD_WGHT": SHS_data_CAD_WGHT,
        "SHS_data_Mln_EUR_WGHT": SHS_data_Mln_EUR_WGHT,
        "COICOP_code": COICOP_code,
        "NACE_to_COICOP_mask": NACE_to_COICOP_mask,
        "RAS_NACE_COICOP_spends": RAS_NACE_COICOP_spends,
        "RAS_nace_to_coicop_shares": RAS_nace_to_coicop_shares,
        "hh_EF_NACE": hh_EF_NACE,
        "EF_indir_COICOP": EF_indir_COICOP,
        "e_COICOP_indir": e_COICOP_indir,
        "dir_E_road_resid": dir_E_road_resid,
        "e_COICOP_dir": e_COICOP_dir,
        "e_COICOP_TJperMlnEur": e_COICOP_TJperMlnEur,
        "EF_COICOP_TJ_indir_dir": EF_COICOP_TJ_indir_dir,
    }


# %% Run the model for every year in YEARS
EM = {}
for _year in YEARS:
    print("=" * 70)
    print(f"Energy Model — {_year}")
    print("=" * 70)
    EM[_year] = energy_model_year(_year)


# %% Summary across years
# Totals per year (no new calculation, only sums of the results above):
#   SHS spends (mln EUR), FIGARO household FD (mln EUR), indirect / direct /
#   total household energy footprint (TJ).
EM_summary = pd.DataFrame({
    year: {
        "SHS spends (mln EUR)":    r["SHS_data_Mln_EUR_WGHT"].iloc[:, 2:].values.sum(),
        "FIGARO Y_hh (mln EUR)":   r["Y_hh"].sum(),
        "EF indirect (TJ)":        r["EF_COICOP_TJ_indir_dir"]["EF_indir_COICOP"].sum(),
        "EF direct (TJ)":          r["EF_COICOP_TJ_indir_dir"]["EF_dir_COICOP"].sum(),
        "EF total (TJ)":           r["EF_COICOP_TJ_indir_dir"].values.sum(),
        "Number of COICOP codes":  len(r["COICOP_code"]),
    }
    for year, r in EM.items()
})
print(EM_summary.round(1))


# %% Unpack one year into top-level variables (for the Variables panel)
# Pick which year's results show up as plain variables (e_COICOP_TJperMlnEur,
# EF_COICOP_TJ_indir_dir, ...). All years stay available in EM[year].
SHOW_YEAR = YEARS[-1]
globals().update({k: v for k, v in EM[SHOW_YEAR].items() if k != "year"})
print(f"Top-level variables now hold the {SHOW_YEAR} results.")

# %%
