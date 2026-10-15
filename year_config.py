"""
Per-year configuration for the multi-year Energy Model (2017, 2019, 2021).

Everything that differs between survey years lives here, so Energy_Model.py
and Household_EF.py can loop over any subset of YEARS without year-specific
code:

  - which SHS microdata file to read (2017 = Diary CSV, 2019/2021 = SAS),
  - the COICOP_to_FIGARO_Bridge_<year>.xlsx file, its sheet names and last
    COICOP column (2017 has 77 SHS codes -> D:CB, 2019/2021 have 75 -> D:BZ),
  - the IEA_data_<year>.xlsx file,
  - the CAD -> EUR exchange rate,
  - renames of SHS socio-demographic variables back to the 2019 names.

Example usage:

    from year_config import YEAR_CONFIG, load_shs_raw, bridge_coicop_codes

    raw_2021 = load_shs_raw(DATA_DIR, 2021)          # 2019-style column names
    codes_2021 = bridge_coicop_codes(DATA_DIR, 2021)  # 75 SHS codes, bridge order
"""

# %%
import os
import pandas as pd

AVAILABLE_YEARS = [2017, 2019, 2021]

# Relative to DATA_DIR. Windows-style separators to match the rest of the code.
_SHS_ROOT = r"OneDrive_2026-03-09\Survey of Household Spending"

YEAR_CONFIG = {
    2017: {
        # 2017 PUMF comes as two files. The Diary file (4,012 households,
        # weight WeightD) also carries every interview item, so it covers all
        # 77 bridge codes; the Interview file (WeightC) has no diary items.
        "shs_path": _SHS_ROOT + r"\SHS_EDM_2017\Data - Données\CSV\SHS-62M004X-E-2017-Diary_F1.csv",
        "shs_format": "csv",
        "bridge_file": "COICOP_to_FIGARO_Bridge_2017.xlsx",
        "bridge_sheet": "COICOP_to_FIGARO_Bridge_2017",
        "aggr_sheet": "COICOP_to_COICOPaggr_2017",
        "last_col": "CB",           # 77 COICOP columns, D:CB
        "iea_file": "IEA_data_2017.xlsx",
        "exchange_rate": 1.4647,    # CAD per EUR, ECB 2017 (bridge sheet SHS_PUMF_2017!R5)
        # 1000 bootstrap weights for the Diary file (fixed-width text, 12 chars per weight).
        "bsw_path": _SHS_ROOT + r"\SHS_EDM_2017\Data - Données\TXT\pumf_shs2017_diary_bsw_flatfile.txt",
        "bsw_format": "txt",
    },
    2019: {
        "shs_path": _SHS_ROOT + r"\SHS_EDM_2019\Data\SAS\pumf_shs2019.sas7bdat",
        "shs_format": "sas",
        "bridge_file": "COICOP_to_FIGARO_Bridge_2019.xlsx",
        "bridge_sheet": "COICOP_to_FIGARO_Bridge",
        "aggr_sheet": "COICOP_to_COICOPaggr",
        "last_col": "BZ",           # 75 COICOP columns, D:BZ
        "iea_file": "IEA_data_2019.xlsx",
        "exchange_rate": 1.49,      # CAD per EUR, as in the original 2019 model
        "bsw_path": _SHS_ROOT + r"\SHS_EDM_2019\Data\SAS\pumf_shs2019_bsw.sas7bdat",
        "bsw_format": "sas",
    },
    2021: {
        "shs_path": _SHS_ROOT + r"\SHS_EDM_2021\RY2021\Data\SAS\pumf_shs2021.sas7bdat",
        "shs_format": "sas",
        "bridge_file": "COICOP_to_FIGARO_Bridge_2021.xlsx",
        "bridge_sheet": "COICOP_to_FIGARO_Bridge_2021",
        "aggr_sheet": "COICOP_to_COICOPaggr",
        "last_col": "BZ",           # 75 COICOP columns, D:BZ (different set than 2019)
        "iea_file": "IEA_data_2021.xlsx",
        "exchange_rate": 1.4826,    # CAD per EUR (bridge sheet SHS_PUMF_2021!O5)
        "bsw_path": _SHS_ROOT + r"\SHS_EDM_2021\RY2021\Data\SAS\pumf_2021_final_bsw.sas7bdat",
        "bsw_format": "sas",
    },
}

# Socio-demographic variables renamed to the 2019 PUMF names.
SOCIO_RENAME = {
    2017: {
        "RP_Sex": "RP_Gender",
    },
    2019: {},
    2021: {
        "P0to4": "P0to4YN",
        "P5to15": "P5to15YN",
        "RP_AgeGp": "RP_AgeGrp",
        "RP_Gen": "RP_Gender",
        "RP_MaSt": "RP_MarStat",
        "SecRes": "SecResYN",
        "OthProp": "OthPropYN",
        "Internet": "InternetYN",
        "OwnVeh": "VehicleYN",
        "MAJINS": "HH_MajIncSrc",
        "HHTOTINC": "HH_TotInc",
    },
}

# Categorical socio-demographic columns stored as strings in the 2019 PUMF.
_CATEGORICAL = [
    "CaseID", "Prov", "HHType6", "HHSize", "P0to4YN", "P5to15YN", "P5to17YN",
    "RP_AgeGrp", "RP_Gender", "RP_MarStat", "RP_Educ", "DwelTyp", "Tenure",
    "NumBedr", "SecResYN", "OthPropYN", "InternetYN", "VehicleYN",
    "HH_MajIncSrc",
]


def check_years(years):
    """Validate a list of years against AVAILABLE_YEARS and return it sorted."""
    years = sorted(set(int(y) for y in years))
    bad = [y for y in years if y not in YEAR_CONFIG]
    if not years or bad:
        raise ValueError(
            f"YEARS must be a non-empty subset of {AVAILABLE_YEARS}, got {years}"
        )
    return years


def path_for(data_dir, year, key):
    """Full path of a per-year input file (key: 'shs_path', 'bridge_file', 'iea_file')."""
    return os.path.join(data_dir, YEAR_CONFIG[year][key])


def harmonise_shs(df, year):
    """
    Rename socio-demographic columns to the 2019 names and store categorical
    codes in the 2019 string format, so the same downstream code works for
    every year. Expenditure columns are left untouched.

      - CaseID      : zero-padded 6-digit string ("000010")
      - RP_AgeGrp   : 2-digit string ("01".."06"); 2017 stores 1..6
      - HHType6     : no leading zero ("1".."6"); 2021 stores "01".."06"
      - other codes : plain strings ("1", "2", ...)
    """
    df = df.rename(columns=SOCIO_RENAME[year])
    for col in _CATEGORICAL:
        if col not in df.columns:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            s = s.astype("Int64").astype(str).replace("<NA>", pd.NA)
        s = s.astype("string").str.strip()
        if col == "CaseID":
            s = s.str.zfill(6)
        elif col == "RP_AgeGrp":
            s = s.str.zfill(2)
        elif col == "HHType6":
            s = s.str.lstrip("0")
        df[col] = s
    return df


def load_shs_raw(data_dir, year):
    """
    Load the full SHS PUMF file for `year` (every column, raw values, no
    weighting) with harmonised socio-demographic columns.
    """
    cfg = YEAR_CONFIG[year]
    path = path_for(data_dir, year, "shs_path")
    if cfg["shs_format"] == "csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_sas(path, format="sas7bdat", encoding="latin1")
    return harmonise_shs(df, year)


def bridge_coicop_codes(data_dir, year):
    """
    SHS expenditure codes used in the bridge for `year`, in bridge column
    order (row 8 of the bridge sheet, from column D to last_col).
    """
    cfg = YEAR_CONFIG[year]
    row = pd.read_excel(
        path_for(data_dir, year, "bridge_file"),
        sheet_name=cfg["bridge_sheet"],
        usecols=f"D:{cfg['last_col']}",
        header=None,
        skiprows=7,
        nrows=1,
    )
    return row.iloc[0].astype(str).str.strip().tolist()


def load_bsw(data_dir, year, n_boot=None):
    """
    Statistics Canada bootstrap weights for `year` (1000 replicates per
    household). Returns a DataFrame indexed by CaseID (6-digit string) with
    columns BSW1..BSW<n_boot>. Each column is a full set of household weights,
    on the same scale as WeightD.
    """
    cfg = YEAR_CONFIG[year]
    path = path_for(data_dir, year, "bsw_path")
    if not os.path.exists(path):
        path = path.replace("\\", os.sep)
    if cfg["bsw_format"] == "sas":
        df = pd.read_sas(path, format="sas7bdat", encoding="latin1")
        df = df.rename(columns={"caseid": "CaseID"})
        df["CaseID"] = df["CaseID"].astype(str).str.strip().str.zfill(6)
        df = df.set_index("CaseID")
    else:
        # Fixed width: CaseID in chars 1-6, then 12 chars per weight. Each
        # record is wrapped over two physical lines, so join the text first.
        with open(path, "rb") as f:
            text = f.read().decode("latin1").replace("\r", "")
        lines = [ln for ln in text.split("\n") if ln.strip()]
        records, cur = [], ""
        for ln in lines:
            cur += ln
            if len(cur) >= 6 + 12 * 1000:
                records.append(cur)
                cur = ""
        ids = [r[:6] for r in records]
        vals = [[float(r[6 + 12 * i: 18 + 12 * i]) for i in range(1000)] for r in records]
        df = pd.DataFrame(vals, index=pd.Index(ids, name="CaseID"),
                          columns=[f"BSW{i}" for i in range(1, 1001)])
    cols = [f"BSW{i}" for i in range(1, 1001)]
    if n_boot is not None:
        cols = cols[:n_boot]
    return df[cols].astype(float)
