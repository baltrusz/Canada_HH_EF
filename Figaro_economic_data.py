"""
FIGARO Economic Data loader.

Wraps the loading + processing of FIGARO economic data into a single
function that can be called from another script.

Example usage from another script:

    from figaro_economic_data import figaro_economic_data

    result = figaro_economic_data(
        figaro_folder=r"C:\\path\\to\\FIGARO data",
        first_year=2010,
        last_year=2023,
        edition_economic_data=2501,
    )
    X = result["X"]
    Y = result["Y"]
    Z = result["Z"]

NOTE: To import this from another script, the file name cannot contain
spaces. Rename this file to `figaro_economic_data.py` (no spaces) before
using the import line above.
"""

# %%
import os
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd


def figaro_economic_data(
    figaro_folder,
    first_year,
    last_year,
    country_code,
    edition_economic_data=2501,
    hh_category="P3_S14",
    fd_categories=None,
):
    """
    Load FIGARO economic data and build X, Y, Z 3D arrays.

    Parameters
    ----------
    figaro_folder : str
        Path to the locally-synced FIGARO data folder.
    first_year, last_year : int
        Inclusive year range, e.g. 2010 and 2023.
    country_code : str
        Country code (e.g. "CA"). The household FD vector for this country
        is always returned under the key `f"{country_code}_Y_hh"`.
    edition_economic_data : int
        FIGARO edition number (default 2501). Used to build the file prefix
        `matrix_eu-ic-io_ind-by-ind_<edition>ed_<year>.csv`.
    hh_category : str
        Household consumption FD category (default "P3_S14").
    fd_categories : list of str, optional
        Order of FD categories along the columns of Y. Defaults to
        ['P3_S13', 'P3_S14', 'P3_S15', 'P51G', 'P5M'].

    Returns
    -------
    dict with keys:
        X : np.ndarray, shape (m*n, 1, n_years)        total output
        Y : np.ndarray, shape (m*n, m*k, n_years)      final demand
        Z : np.ndarray, shape (m*n, m*n, n_years)      intermediate flows
        country_sector_combined : list[str]            row/col labels
        meta : SimpleNamespace                         meta.xlsx sheets
            .regions, .sectors, .FDcategories : pd.DataFrame
        m, n, mn, k : int                              dimension constants
        figaro : dict[int, pd.DataFrame]               raw yearly DataFrames
        {country_code}_Y_hh : np.ndarray, shape (m*n,)
            Household final demand for that country (year_idx=0).
    """
    # ---- 1. Validate folder ----
    if not os.path.isdir(figaro_folder):
        raise FileNotFoundError(
            f"FIGARO folder does not exist:\n  {figaro_folder}\n\n"
            "Sync the SharePoint folder to your computer, then pass the "
            "correct path as `figaro_folder`."
        )

    # File prefix is built from the edition so any edition works
    file_prefix = f"matrix_eu-ic-io_ind-by-ind_{edition_economic_data}ed_"
    years = range(first_year, last_year + 1)

    # ---- 2. Find matching CSV files (recursively under figaro_folder) ----
    matching = []
    for root, _, files in os.walk(figaro_folder):
        for f in files:
            if f.startswith(file_prefix):
                matching.append(os.path.join(root, f))

    print(f"Found {len(matching)} matching files:")
    for path in sorted(matching):
        print(" ", os.path.relpath(path, figaro_folder))

    # ---- 3. Load each year into a dict {year: DataFrame} ----
    figaro = {}
    for year in years:
        candidates = [p for p in matching if str(year) in os.path.basename(p)]
        if not candidates:
            print(f"  [skip] no file found for {year}")
            continue
        full_path = candidates[0]
        print(f"Loading {os.path.basename(full_path)} ...")
        figaro[year] = pd.read_csv(full_path)
    print(f"\nLoaded {len(figaro)} years: {sorted(figaro.keys())}")

    # ---- 4. Load meta.xlsx (search recursively) ----
    meta_path = None
    for root, _, files in os.walk(figaro_folder):
        if "meta.xlsx" in files:
            meta_path = os.path.join(root, "meta.xlsx")
            break
    if meta_path is None:
        raise FileNotFoundError("meta.xlsx not found anywhere under figaro_folder")

    regions = pd.read_excel(meta_path, sheet_name="country")
    sectors = pd.read_excel(meta_path, sheet_name="sector")
    FDcategories = pd.read_excel(meta_path, sheet_name="final_demand")

    # Bundle meta sheets into a single namespace (access as meta.regions, etc.)
    meta = SimpleNamespace(
        regions=regions,
        sectors=sectors,
        FDcategories=FDcategories,
    )

    # ---- 5. Dimension constants (computed from the meta sheets) ----
    m = len(regions)          # number of regions
    n = len(sectors)          # number of sectors
    mn = m * n                # total country-sector combinations
    k = 5                     # final demand categories (P3_S13, P3_S14, P3_S15, P51G, P5M)

    # ---- 6. Build X, Y, Z 3D arrays ----
    timer_start = time.time()

    # U and T sectors are NOT included in final demand
    final_demand_names = ['P3_S13', 'P3_S14', 'P3_S15', 'P51G', 'P5M']

    # Country and sector codes are in column 2 of the meta sheets
    country_names = list(regions.iloc[:, 1])
    sector_names = list(sectors.iloc[:, 1])

    country_sector_combined = [f"{c}_{s}" for c in country_names for s in sector_names]
    country_fd_combined = [f"{c}_{fd}" for c in country_names for fd in final_demand_names]

    years_list = list(years)
    n_years = len(years_list)
    n_cs = m * n
    n_fd = m * len(final_demand_names)

    # Preallocate
    Z = np.zeros((n_cs, n_cs, n_years))
    Y = np.zeros((n_cs, n_fd, n_years))
    X = np.zeros((n_cs, 1, n_years))

    for i, year in enumerate(years_list):
        print(f"  Importing FIGARO economic data for year: {year}")
        if year not in figaro:
            raise KeyError(f"Year {year} was not loaded (no matching file).")
        T = figaro[year]

        # First column = row labels; rest = data columns
        row_label_col = T.columns[0]
        data_col_set = set(T.columns[1:])

        # Validate columns
        missing_z = [c for c in country_sector_combined if c not in data_col_set]
        if missing_z:
            raise ValueError(
                f"Missing country-sector columns for year {year}: {missing_z[:5]} ..."
            )
        missing_y = [c for c in country_fd_combined if c not in data_col_set]
        if missing_y:
            raise ValueError(
                f"Missing country-final-demand columns for year {year}: {missing_y[:5]} ..."
            )

        # Filter and reorder rows to match country_sector_combined
        T_indexed = T.set_index(row_label_col)
        missing_rows = [r for r in country_sector_combined if r not in T_indexed.index]
        if missing_rows:
            raise ValueError(
                f"Missing country-sector rows for year {year}: {missing_rows[:5]} ..."
            )
        T_filtered = T_indexed.loc[country_sector_combined]

        Z[:, :, i] = T_filtered[country_sector_combined].to_numpy()
        Y[:, :, i] = T_filtered[country_fd_combined].to_numpy()
        # Total output = row sum of [Z | Y]
        X[:, 0, i] = Z[:, :, i].sum(axis=1) + Y[:, :, i].sum(axis=1)

    # Avoid zeros (used later when dividing by X)
    X[X == 0] = 1e-9

    elapsed = time.time() - timer_start
    print(f"Done. Imported FIGARO economic data for {n_years} years in {elapsed:.2f} seconds")

    result = {
        "X": X,
        "Y": Y,
        "Z": Z,
        "country_sector_combined": country_sector_combined,
        "meta": meta,
        "m": m,
        "n": n,
        "mn": mn,
        "k": k,
        "figaro": figaro,
    }

    # ---- Extract the household FD vector for the chosen country ----
    fd_cat_list = fd_categories if fd_categories is not None else final_demand_names
    country_codes_list = regions.iloc[:, 1].astype(str).str.strip().tolist()
    if country_code not in country_codes_list:
        raise ValueError(
            f"country_code {country_code!r} not found in regions.iloc[:, 1]"
        )
    if hh_category not in fd_cat_list:
        raise ValueError(
            f"hh_category {hh_category!r} not in fd_categories: {fd_cat_list}"
        )
    c_idx = country_codes_list.index(country_code)
    fd_off = fd_cat_list.index(hh_category)
    hh_col = c_idx * k + fd_off
    y_hh = Y[:, hh_col, 0].astype(float).copy()
    key = f"{country_code}_Y_hh"
    result[key] = y_hh
    print(f"{key} shape: {y_hh.shape}, total = {y_hh.sum():,.2f} (million EUR)")

    return result


# %%
# ---------------------------------------------------------------------------
# Standalone usage — only runs when this file is executed directly.
# When imported from another script, this block is skipped.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ===========================================================================
    # >>> REPLACE THIS PATH <<<
    # After syncing the SharePoint "FIGARO data" folder to your computer,
    # right-click the folder in File Explorer -> "Copy as path" and paste below.
    # ===========================================================================
    FIGARO_FOLDER = r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents - 403001-01 Utvikling og vedlikehold klimakost\General\Utslippsfaktorer\Kryssløpsanalyse\FIGARO data"

    result = figaro_economic_data(
        figaro_folder=FIGARO_FOLDER,
        first_year=2010,
        last_year=2023,
        country_code="CA",
        edition_economic_data=2501,
    )

    X = result["X"]
    Y = result["Y"]
    Z = result["Z"]
    print(f"\nX shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Z shape: {Z.shape}")
    print(f"m={result['m']}, n={result['n']}, mn={result['mn']}, k={result['k']}")
