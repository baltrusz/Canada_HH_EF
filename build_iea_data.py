"""
Build IEA_data_<year>.xlsx for the Energy Model from an IEA World Energy
Balances (WBAL) csv export.

The output copies the layout of an existing IEA_data_<year>.xlsx (the
template, e.g. IEA_data_2021.xlsx): same sheets, same rows and columns. The
Excel formulas of the template (FILTER, UNIQUE, XLOOKUP, MMULT) are
recomputed in Python and written as values, so the Energy Model can read the
file directly, without opening and saving it in Excel first.

What is computed (same logic as the formulas in IEA_data_2021.xlsx):

  IEA_Raw       : all WBAL rows for `year` (UNIT .. Value, 11 columns).
  IEA_filtered  : rows of IEA_Raw whose PRODUCT contains "TOTAL"; column N =
                  unique flow names.
  RoW           : rows 3-26 = total final consumption per IEA sector (23
                  sectors + Residential at row 22) for the World (col C) and
                  each FIGARO country (cols D:AV); col B = FIGW1 = World minus
                  the 45 FIGARO countries.
  meta          : FIGARO country x IEA sector, with IEA country and flow codes.
  IEA_to_Figaro_data :
                  G = final energy of the sector (FIGW1: RoW col B),
                  H = international aviation bunkers added to Domestic aviation,
                  I = international marine bunkers added to Domestic navigation,
                  J = G + H + I, K = 40 % of J for Road (direct, households),
                  L = J - K for Road, M = J - K (read by the Energy Model).
  Direct_resid_road_FIGAROctrs :
                  Residential and Road direct (K) per FIGARO country.
  Resid_Road_split_to_COICOP :
                  Canada residential energy by product (oil, gas, biofuels,
                  electricity) + Road direct, split over SH034, SH030_C,
                  SH030_B, SH062 and TR036. Electricity is split between
                  SH030_B and SH062 by their weighted SHS spends (in K EUR).

Usage (from the repo folder):

    python build_iea_data.py --wbal "WBAL-....csv" --year 2023 \
        --template IEA_data_2021.xlsx --out IEA_data_2023.xlsx \
        --shs-spends SHS_2023_weighted_expenditure.xlsx --exchange-rate 1.4595

`--wbal` can also be an already-filtered csv or xlsx with the 11 WBAL columns
(UNIT, Unit, COUNTRY, Country, PRODUCT, Product, FLOW, Flow, TIME, Time,
Value). `--shs-spends` is a sheet with expenditure_code / weighted_sum in CAD
(the weight_exp_<year> layout). Without it, the template's electricity split
is kept.
"""

import argparse

import numpy as np
import openpyxl
import pandas as pd

WBAL_COLS = ["UNIT", "Unit", "COUNTRY", "Country", "PRODUCT", "Product",
             "FLOW", "Flow", "TIME", "Time", "Value"]

ROW_FIRST, ROW_LAST = 3, 26          # RoW sheet: sector rows (24 incl. Residential)
ROW_CTRY_FIRST, ROW_CTRY_LAST = 29, 74  # RoW sheet: FIGARO <-> IEA country table


def read_wbal(path, year, chunksize=2_000_000):
    """Rows of the WBAL export for `year`, in TJ. Reads big csv files in chunks."""
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
        df = df[WBAL_COLS]
    else:
        parts = []
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False,
                                 encoding="latin1"):
            chunk.columns = [c.strip() for c in chunk.columns]
            parts.append(chunk[pd.to_numeric(chunk["TIME"], errors="coerce") == year])
        df = pd.concat(parts, ignore_index=True)[WBAL_COLS]
    df = df[pd.to_numeric(df["TIME"], errors="coerce") == year]
    if "UNIT" in df and df["UNIT"].nunique() > 1:
        df = df[df["UNIT"] == "TJ"]
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0.0)
    return df.reset_index(drop=True)


def _first(df, **conds):
    m = np.ones(len(df), dtype=bool)
    for col, val in conds.items():
        m &= (df[col] == val).to_numpy()
    hit = df.loc[m, "Value"]
    return None if hit.empty else float(hit.iloc[0])


class Lookup:
    """XLOOKUP on (COUNTRY, Flow) in a table, also trying the quoted flow name."""

    def __init__(self, df):
        d = df.drop_duplicates(["COUNTRY", "Flow"], keep="first")
        self.v = dict(zip(zip(d["COUNTRY"], d["Flow"]), d["Value"]))
        d = df.drop_duplicates(["COUNTRY", "FLOW"], keep="first")
        self.v_code = dict(zip(zip(d["COUNTRY"], d["FLOW"]), d["Value"]))
        self.missing = []

    def __call__(self, country, flow):
        for key in ((country, flow), (country, f'"{flow}"')):
            if key in self.v:
                return float(self.v[key])
        self.missing.append((country, flow))
        return 0.0

    def code(self, country, flow_code):
        return float(self.v_code.get((country, flow_code), 0.0))


def build(raw, year, template, out, spends_keur=None):
    raw = raw.copy()
    filt = raw[raw["PRODUCT"].astype(str).str.contains("TOTAL", case=False)].reset_index(drop=True)
    look = Lookup(filt)

    wb = openpyxl.load_workbook(template, data_only=True)

    # ---- IEA_Raw ----
    ws = wb["IEA_Raw"]
    ws.delete_rows(2, ws.max_row)
    for r in raw[WBAL_COLS].itertuples(index=False):
        ws.append(list(r))

    # ---- IEA_filtered ----
    ws = wb["IEA_filtered"]
    ws.delete_rows(2, ws.max_row)
    flows = list(dict.fromkeys(raw["Flow"].tolist()))
    for i, r in enumerate(filt[WBAL_COLS].itertuples(index=False)):
        ws.append(list(r) + [None, None, flows[i] if i < len(flows) else None])
    for j in range(len(filt), len(flows)):
        ws.cell(j + 2, 14, flows[j])

    # ---- RoW ----
    ws = wb["RoW"]
    ws.cell(1, 3, f"World value {year} TJ")
    ieacodes = [ws.cell(2, c).value for c in range(3, 49)]   # C2:AV2 (WLD + 45)
    sectors = [ws.cell(r, 1).value for r in range(ROW_FIRST, ROW_LAST + 1)]
    row_vals = {}
    for r, sect in zip(range(ROW_FIRST, ROW_LAST + 1), sectors):
        vals = [look(code, sect) for code in ieacodes]
        for c, v in zip(range(3, 49), vals):
            ws.cell(r, c, v)
        figw1 = vals[0] - sum(vals[1:])
        ws.cell(r, 2, figw1)
        row_vals[sect] = {"FIGW1": figw1, **dict(zip(ieacodes, vals))}
    fig2iea = {ws.cell(r, 1).value: ws.cell(r, 3).value
               for r in range(ROW_CTRY_FIRST, ROW_CTRY_LAST + 1)}

    # ---- meta ----
    flow_code = dict(zip(filt["Flow"].str.strip('"'), filt["FLOW"]))
    ws = wb["meta"]
    meta = []
    for r in range(2, ws.max_row + 1):
        fig, sect = ws.cell(r, 1).value, ws.cell(r, 3).value
        if fig is None:
            continue
        iea = fig2iea.get(fig)
        ws.cell(r, 2, iea)
        ws.cell(r, 4, flow_code.get(sect))
        meta.append((r, fig, iea, sect))

    # ---- IEA_to_Figaro_data ----
    ws = wb["IEA_to_Figaro_data"]
    others = [c for c in fig2iea.values() if c != "FIGW1"]
    road_K = {}
    for r, fig, iea, sect in meta:
        if iea == "FIGW1":
            g = row_vals[sect]["FIGW1"]
            h = (look.code("WLD", "WORLDAV")
                 + sum(look(c, "International aviation bunkers") for c in others)
                 if sect == "Domestic aviation" else 0.0)
            i = (look.code("WLD", "WORLDMAR")
                 + sum(look(c, "International marine bunkers") for c in others)
                 if sect == "Domestic navigation" else 0.0)
        else:
            g = look(iea, sect)
            h = -look(iea, "International aviation bunkers") if sect == "Domestic aviation" else 0.0
            i = -look(iea, "International marine bunkers") if sect == "Domestic navigation" else 0.0
        j = g + h + i
        k = j * 0.4 if sect == "Road" else 0.0
        l = j - k if k > 0 else 0.0
        for col, v in zip(range(1, 14), [fig, iea, sect, flow_code.get(sect), year, "TJ",
                                         g, h, i, j, k, l, j - k]):
            ws.cell(r, col, v)
        if sect == "Road":
            road_K[fig] = k

    # ---- Direct_resid_road_FIGAROctrs ----
    ws = wb["Direct_resid_road_FIGAROctrs"]
    resid = row_vals["Residential"]
    for r in range(2, ws.max_row + 1):
        fig = ws.cell(r, 2).value
        if fig is None:
            continue
        ws.cell(r, 3, fig2iea.get(fig))
        ws.cell(r, 4, year)
        ws.cell(r, 6, resid.get(fig2iea.get(fig), 0.0))
        ws.cell(r, 7, road_K.get(fig, 0.0))
    for n, code in enumerate(["FIGW1"] + ieacodes):          # J:L helper columns
        ws.cell(n + 3, 10, resid.get(code))
        ws.cell(n + 3, 11, code)
        ws.cell(n + 3, 12, next((f for f, c in fig2iea.items() if c == code), None))
    can_road = road_K.get("CA", 0.0)

    # ---- Resid_Road_split_to_COICOP ----
    ws = wb["Resid_Road_split_to_COICOP"]
    raw_res = raw[(raw["Country"] == "Canada") & (raw["Flow"] == "Residential")]
    b = []
    for r in range(2, 6):
        prod = ws.cell(r, 1).value
        v = _first(raw_res, Product=prod)
        if v is None:
            v = _first(raw_res, Product=f'"{prod}"') or 0.0
        b.append(v)
        ws.cell(r, 2, v)
        ws.cell(r, 6, year)
    b.append(can_road)
    ws.cell(6, 2, can_road)
    ws.cell(6, 6, year)
    tot = sum(b)
    ws.cell(8, 2, tot)
    ws.cell(9, 2, can_road)
    for r, v in zip(range(2, 7), b):
        ws.cell(r, 3, v / tot if tot else 0.0)
    ws.cell(2, 4, b[0] / (b[0] + b[2]) if (b[0] + b[2]) else 0.0)
    ws.cell(4, 4, b[2] / (b[0] + b[2]) if (b[0] + b[2]) else 0.0)

    # weighted spends (K EUR) and the electricity split
    if spends_keur is not None:
        for r in range(12, 17):
            code = ws.cell(r, 2).value
            ws.cell(r, 3, spends_keur.get(code))
            ws.cell(r, 4, spends_keur.get(code))
    e1, e2 = ws.cell(14, 4).value or 0.0, ws.cell(15, 4).value or 0.0
    s1 = e1 / (e1 + e2) if (e1 + e2) else 0.0
    ws.cell(14, 6, s1)
    ws.cell(15, 6, 1 - s1)
    ws.cell(24, 4, s1)
    ws.cell(24, 5, 1 - s1)
    for r in range(21, 26):
        ws.cell(r, 7, sum(ws.cell(r, c).value or 0.0 for c in range(2, 7)))
    # MMULT: B2:B6 (5 IEA rows) x B21:F25 (5 x 5) -> 5 COICOP values
    M = np.array([[ws.cell(r, c).value or 0.0 for c in range(2, 7)] for r in range(21, 26)])
    direct = np.array(b) @ M
    for r, v in zip(range(29, 34), direct):
        ws.cell(r, 3, float(v))
    ws.cell(34, 4, float(direct.sum() - tot))

    wb.save(out)
    return {"missing_lookups": sorted(set(look.missing)), "direct_TJ": direct,
            "n_raw": len(raw), "n_filtered": len(filt)}


def spends_keur_from_sheet(path, exchange_rate, sheet=0):
    df = pd.read_excel(path, sheet_name=sheet)
    return dict(zip(df["expenditure_code"], df["weighted_sum"] / exchange_rate / 1000))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wbal", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shs-spends")
    ap.add_argument("--exchange-rate", type=float)
    a = ap.parse_args()
    if a.out == a.template:
        raise SystemExit("--out must differ from --template")
    raw = read_wbal(a.wbal, a.year)
    if raw.empty:
        raise SystemExit(f"No rows for {a.year} in {a.wbal}")
    sp = spends_keur_from_sheet(a.shs_spends, a.exchange_rate) if a.shs_spends else None
    res = build(raw, a.year, a.template, a.out, sp)
    print(f"{res['n_raw']} raw rows, {res['n_filtered']} TOTAL rows")
    print("Direct energy (TJ) SH034, SH030_C, SH030_B, SH062, TR036:", res["direct_TJ"])
    if res["missing_lookups"]:
        print("Not found (set to 0):", res["missing_lookups"])
