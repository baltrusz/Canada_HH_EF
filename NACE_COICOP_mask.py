"""
NACE-to-COICOP bridge matrix loader.

Loads the many-to-many bridge matrix from the COICOP_to_FIGARO_Bridge_2019.xlsx
file and optionally validates it against `sectors` and the SHS DataFrame.

Example usage from another script:

    from NACE_COICOP_mask import nace_to_coicop_mask

    NACE_to_COICOP_mask = nace_to_coicop_mask(
        bridge_path=r"C:\\path\\to\\COICOP_to_FIGARO_Bridge_2019.xlsx",
        sectors=meta.sectors,        # optional: enables sector-order check
        shs_data=SHS_2019_data,      # optional: enables COICOP-name check
    )
"""

# %%
import pandas as pd


def nace_to_coicop_mask(bridge_path, sectors=None, shs_data=None,
                        sheet_name="COICOP_to_FIGARO_Bridge", last_col="BZ"):
    """
    Load the NACE-to-COICOP bridge matrix from `bridge_path`.

    The matrix is read from tab `sheet_name`, range B8:<last_col>70
    (2019 defaults: "COICOP_to_FIGARO_Bridge", B8:BZ70; the per-year values
    are in year_config.YEAR_CONFIG).
    Layout:
        column 0  = sector codes (NACE)
        column 1  = "FIGARO sector name / COICOP CODE -->" label
        columns 2+ = COICOP codes (1/0 bridge values below)

    Parameters
    ----------
    bridge_path : str
        Full path to COICOP_to_FIGARO_Bridge_2019.xlsx.
    sectors : pd.DataFrame, optional
        If provided, checks that column 0 of the mask matches `sectors.iloc[:, 1]`
        in the same order.
    shs_data : pd.DataFrame, optional
        If provided, checks that the mask's COICOP column names (columns 2+)
        match the SHS column names with the "_wght" suffix stripped
        (excluding CaseID and WeightD).
    sheet_name : str
        Bridge tab name (differs per year, see year_config).
    last_col : str
        Last COICOP column letter ("BZ" for 75 codes, "CB" for 77 in 2017).

    Returns
    -------
    pd.DataFrame
        62 × (2 + n_COICOP) bridge matrix (62 × 77 for 2019).
    """
    mask = pd.read_excel(
        bridge_path,
        sheet_name=sheet_name,
        header=0,           # row 8 contains column names
        skiprows=7,         # skip rows 1-7, so row 8 becomes the header
        nrows=62,           # rows 9 to 70 inclusive = 62 data rows
        usecols=f"B:{last_col}",  # B:BZ = 77 columns for 2019
    )
    print(f"NACE_to_COICOP_mask shape: {mask.shape}")

    # ---- Optional validation: sector order ----
    if sectors is not None:
        mask_sectors = mask.iloc[:, 0].astype(str).str.strip().tolist()
        expected_sectors = sectors.iloc[:, 1].astype(str).str.strip().tolist()
        if mask_sectors != expected_sectors:
            mismatches = [
                (i, m, e)
                for i, (m, e) in enumerate(zip(mask_sectors, expected_sectors))
                if m != e
            ]
            raise ValueError(
                "Sector order mismatch in NACE_to_COICOP_mask first column.\n"
                f"  First mismatch at row {mismatches[0][0]}: "
                f"mask={mismatches[0][1]!r} vs sectors[:,1]={mismatches[0][2]!r}"
            )
        print(f"OK: first column matches sectors order ({len(mask_sectors)} sectors)")

    # ---- Optional validation: COICOP column names vs SHS columns ----
    if shs_data is not None:
        mask_coicop = [str(c) for c in mask.columns[2:]]
        shs_stripped = [
            c.removesuffix("_wght") for c in shs_data.columns
            if c not in ("CaseID", "WeightD")
        ]
        missing_in_mask = [c for c in shs_stripped if c not in mask_coicop]
        extra_in_mask = [c for c in mask_coicop if c not in shs_stripped]
        if missing_in_mask or extra_in_mask:
            raise ValueError(
                "COICOP column mismatch between mask and SHS data.\n"
                f"  Missing in mask: {missing_in_mask}\n"
                f"  Extra in mask:   {extra_in_mask}"
            )
        print(f"OK: mask column names match SHS columns ({len(mask_coicop)} categories)")

    return mask


def load_coicop_aggregation_masks(bridge_path):
    """
    Load the four COICOP-aggregation masks from tab "COICOP_to_COICOPaggr"
    in COICOP_to_FIGARO_Bridge_2019.xlsx.

    Ranges:
        COICOP75_to_COICOP32  : D25:BZ56  (32 x 75)  - aggregates 75 -> 32
        COICOP75_to_COICOP12  : D59:BZ70  (12 x 75)  - aggregates 75 -> 12
        NACE_to_COICOP32      : D76:AI137 (62 x 32)  - bridge NACE -> COICOP32
        NACE_to_COICOP12      : D144:O205 (62 x 12)  - bridge NACE -> COICOP12

    Empty cells are read as NaN; we fill with 0 so the matrices are pure 0/1.

    Parameters
    ----------
    bridge_path : str
        Full path to COICOP_to_FIGARO_Bridge_2019.xlsx.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: 'COICOP75_to_COICOP32', 'COICOP75_to_COICOP12',
              'NACE_to_COICOP32', 'NACE_to_COICOP12'
    """
    sheet = "COICOP_to_COICOPaggr"

    def _read(usecols, skiprows, nrows):
        return pd.read_excel(
            bridge_path,
            sheet_name=sheet,
            header=None,
            usecols=usecols,
            skiprows=skiprows,
            nrows=nrows,
        ).fillna(0).to_numpy(dtype=float)

    masks = {
        "COICOP75_to_COICOP32": _read("D:BZ", skiprows=24, nrows=32),
        "COICOP75_to_COICOP12": _read("D:BZ", skiprows=58, nrows=12),
        "NACE_to_COICOP32":     _read("D:AI", skiprows=75, nrows=62),
        "NACE_to_COICOP12":     _read("D:O",  skiprows=143, nrows=62),
    }

    expected_shapes = {
        "COICOP75_to_COICOP32": (32, 75),
        "COICOP75_to_COICOP12": (12, 75),
        "NACE_to_COICOP32":     (62, 32),
        "NACE_to_COICOP12":     (62, 12),
    }
    for name, m in masks.items():
        if m.shape != expected_shapes[name]:
            raise ValueError(
                f"{name} expected shape {expected_shapes[name]}, got {m.shape}"
            )
        print(f"  {name}: shape {m.shape}")

    return masks


# %%
# ---------------------------------------------------------------------------
# Standalone usage — only runs when this file is executed directly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    BRIDGE_PATH = r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents\additional_projects\Canada_hh_energy\Data\COICOP_to_FIGARO_Bridge_2019.xlsx"

    NACE_to_COICOP_mask = nace_to_coicop_mask(bridge_path=BRIDGE_PATH)
    print(NACE_to_COICOP_mask.head())

    aggr_masks = load_coicop_aggregation_masks(BRIDGE_PATH)
    print("Loaded aggregation masks:", list(aggr_masks.keys()))
