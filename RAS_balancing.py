"""
RAS balancing of Canada's household consumption against SHS expenditure data.

Pipeline:
  1. Sum the weighted SHS expenditures by COICOP category (CAD).
  2. Convert from CAD to EUR using the 2019 exchange rate (1 EUR = 1.49 CAD).
  3. Extract Canada's household consumption column (P3_S14) from Y.
  4. Distribute the difference (CA_Y_hh_total - SHS_total_EUR) across SHS
     categories proportionally so the SHS column totals match CA_Y_hh_total.
  5. Aggregate CA_Y_hh across the 46 countries to get one value per of the
     62 NACE sectors -> CA_Y_hh_aggr (length 62).
  6. RAS-balance a 62 x 75 matrix:
        - initial structure  = mask (1/0) starting at column index 2
        - target row sums    = CA_Y_hh_aggr
        - target column sums = adjusted SHS totals (EUR)

Example usage from another script:

    from RAS_balancing import ras_balancing

    balanced = ras_balancing(
        mask=NACE_to_COICOP_mask,
        shs_data=SHS_2019_data,
        Y=Y,
        regions=meta.regions,
        m=46, n=62, k=5,
        year_idx=0,
        exchange_rate=1.49,
        country_code="CA",
    )
"""

# %%
import numpy as np
import pandas as pd


def ras_balancing(
    nace_coicop_mask,
    shs_totals_cad,
    Y,
    regions,
    m,
    n,
    k=5,
    sector_labels=None,
    coicop_labels=None,
    year_idx=0,
    exchange_rate=1.49,
    country_code="CA",
    fd_categories=None,
    hh_category="P3_S14",
    max_iter=1000,
    tol=1e-9,
    mode="ras",
):
    """
    Run the RAS balancing pipeline described above.

    Parameters
    ----------
    nace_coicop_mask : np.ndarray or pd.DataFrame
        Bridge mask. Either:
          - np.ndarray of shape (n_sectors, n_coicop) with 0/1 values
          - DataFrame in the original 75-cat format (col 0 = sector code,
            col 1 = label, cols 2+ = COICOP). The function extracts the data.
    shs_totals_cad : np.ndarray, pd.Series, or pd.DataFrame
        SHS expenditure totals in CAD. Either:
          - 1D np.ndarray / pd.Series of length n_coicop (already summed)
          - DataFrame with CaseID, WeightD, and expenditure columns; this
            function will sum the expenditure columns automatically.
    Y : np.ndarray, shape (m*n, m*k, n_years)
        FIGARO final demand matrix.
    regions : pd.DataFrame
        Meta sheet 'country'. Column 1 holds the country codes.
    m, n : int
        Number of regions and sectors.
    k : int
        Number of final-demand categories per country (default 5).
    sector_labels : list of str, optional
        Labels for the rows (n_sectors). Inferred from a DataFrame mask
        if not provided.
    coicop_labels : list of str, optional
        Labels for the columns (n_coicop). Inferred from a DataFrame
        mask/SHS DataFrame if not provided.
    year_idx : int
        Year slice of Y to use (default 0).
    exchange_rate : float
        CAD per EUR. 2019 = 1.49.
    country_code : str
        Country code for Canada in regions.iloc[:, 1] (default "CA").
    fd_categories : list of str
        Order of FD categories along the columns of Y. Defaults to
        ['P3_S13', 'P3_S14', 'P3_S15', 'P51G', 'P5M'].
    hh_category : str
        Household consumption category to extract (default "P3_S14").
    max_iter : int
        Max RAS iterations (default 1000).
    tol : float
        Convergence tolerance on max abs(row/col error) (default 1e-9).
    mode : str
        'ras' (default) - both row and column constraints enforced.
        'columns' - one-pass column-only distribution.
        'rows' - one-pass row-only distribution.

    Returns
    -------
    dict with keys:
        balanced : pd.DataFrame
            Balanced 62 x 75 matrix. Rows = NACE sectors, cols = COICOP.
        CA_Y_hh : np.ndarray, shape (m*n,)
            Canada's P3_S14 column extracted from Y (in million EUR).
        CA_Y_hh_aggr : pd.Series
            CA_Y_hh aggregated by sector across the 46 countries (length 62).
        shs_totals_eur : pd.Series
            SHS expenditure totals in million EUR by COICOP (length 75).
        shs_total_eur : float
            Sum of shs_totals_eur (million EUR).
        shs_totals_adj : pd.Series
            SHS totals after distributing the difference (length 75).
        ca_y_hh_total : float
            Sum of CA_Y_hh (million EUR).
        diff : float
            ca_y_hh_total - shs_total_eur (distributed across SHS shares).
        converged : bool
            Whether RAS converged within `max_iter` iterations.
        iterations : int
            Number of RAS iterations actually run.
    """
    if fd_categories is None:
        fd_categories = ['P3_S13', 'P3_S14', 'P3_S15', 'P51G', 'P5M']

    # ---- Normalise mask input ----
    if isinstance(nace_coicop_mask, pd.DataFrame):
        # Old format: column 0 = sector code, column 1 = label, columns 2+ = data
        if sector_labels is None:
            sector_labels = nace_coicop_mask.iloc[:, 0].astype(str).str.strip().tolist()
        if coicop_labels is None:
            coicop_labels = [str(c) for c in nace_coicop_mask.columns[2:]]
        A0 = nace_coicop_mask.iloc[:, 2:].fillna(0).to_numpy(dtype=float)
    else:
        A0 = np.asarray(nace_coicop_mask, dtype=float)
        A0 = np.nan_to_num(A0, nan=0.0)

    # ---- Normalise SHS totals input ----
    if isinstance(shs_totals_cad, pd.DataFrame):
        # Full SHS DataFrame with CaseID, WeightD, then expenditure cols
        if coicop_labels is None:
            coicop_labels = [str(c) for c in shs_totals_cad.columns[2:]]
        shs_totals_cad = shs_totals_cad.iloc[:, 2:].sum().to_numpy(dtype=float)
    elif isinstance(shs_totals_cad, pd.Series):
        if coicop_labels is None:
            coicop_labels = [str(c) for c in shs_totals_cad.index]
        shs_totals_cad = shs_totals_cad.to_numpy(dtype=float)
    else:
        shs_totals_cad = np.asarray(shs_totals_cad, dtype=float)

    # Default labels if still missing
    if sector_labels is None:
        sector_labels = [f"sector_{i}" for i in range(A0.shape[0])]
    if coicop_labels is None:
        coicop_labels = [f"cat_{j}" for j in range(A0.shape[1])]

    # ---- 2. Convert to million EUR ----
    # SHS is in CAD per household; CA_Y_hh is in million EUR.
    # Divide by exchange rate to get EUR, then by 1,000,000 to get million EUR.
    shs_totals_eur = shs_totals_cad / exchange_rate / 1_000_000

    # ---- 3. Extract Canada's household consumption from Y ----
    countries = regions.iloc[:, 1].astype(str).str.strip().tolist()
    if country_code not in countries:
        raise ValueError(f"Country code {country_code!r} not found in regions.iloc[:, 1]")
    ca_idx = countries.index(country_code)

    if hh_category not in fd_categories:
        raise ValueError(f"hh_category {hh_category!r} not in fd_categories")
    fd_offset = fd_categories.index(hh_category)

    # Column in Y for Canada's P3_S14
    ca_hh_col = ca_idx * k + fd_offset
    CA_Y_hh = Y[:, ca_hh_col, year_idx].astype(float).copy()  # shape (m*n,)

    # ---- 4. Adjust SHS so its sum matches CA_Y_hh sum ----
    ca_y_hh_total = CA_Y_hh.sum()
    shs_total_eur = shs_totals_eur.sum()
    diff = ca_y_hh_total - shs_total_eur
    shs_shares = shs_totals_eur / shs_total_eur
    shs_totals_adj = shs_totals_eur + diff * shs_shares

    print(f"CA_Y_hh sum (EUR):       {ca_y_hh_total:,.2f}")
    print(f"SHS sum (EUR, original): {shs_total_eur:,.2f}")
    print(f"Difference distributed:  {diff:,.2f}")
    print(f"SHS sum (EUR, adjusted): {shs_totals_adj.sum():,.2f}")

    # ---- 5. Aggregate CA_Y_hh across countries for each of the 62 sectors ----
    # Layout: blocks of n sectors per country, repeated for m countries.
    # Sector s lives at indices s, s+n, s+2n, ..., s+(m-1)n.
    CA_Y_hh_aggr = np.array([CA_Y_hh[s::n].sum() for s in range(n)], dtype=float)
    # Sanity: aggregating across countries preserves the total
    assert np.isclose(CA_Y_hh_aggr.sum(), ca_y_hh_total), "Aggregation total mismatch"

    # ---- 6. RAS balancing ----
    # A0 was already built from the mask above

    if A0.shape[0] != n:
        raise ValueError(f"Mask has {A0.shape[0]} rows but n={n}")
    if A0.shape[1] != len(shs_totals_adj):
        raise ValueError(
            f"Mask has {A0.shape[1]} COICOP columns but SHS has "
            f"{len(shs_totals_adj)} expenditure columns"
        )

    # Target sums
    u = CA_Y_hh_aggr        # row totals (62)
    v = shs_totals_adj      # column totals (75)

    # Sanity check: row and col totals must be equal for RAS to converge
    if not np.isclose(u.sum(), v.sum()):
        raise ValueError(f"Row total {u.sum()} != Column total {v.sum()}")

    # Check for empty rows/cols where target is non-zero (mask has no 1s)
    empty_rows = np.where((A0.sum(axis=1) == 0) & (u != 0))[0]
    if len(empty_rows) > 0:
        raise ValueError(f"Mask has empty rows where row total is non-zero: {empty_rows.tolist()}")
    empty_cols = np.where((A0.sum(axis=0) == 0) & (v != 0))[0]
    if len(empty_cols) > 0:
        raise ValueError(f"Mask has empty columns where column total is non-zero: {empty_cols.tolist()}")

    # Subtler infeasibility: a row has u[i] > 0 but every cell where mask[i,j]=1
    # has v[j] = 0. The row can never reach its target.
    stuck_rows = []
    for i in np.where(u > 0)[0]:
        allowed_cols = np.where(A0[i, :] > 0)[0]
        if len(allowed_cols) > 0 and v[allowed_cols].sum() == 0:
            stuck_rows.append((i, sector_labels[i], u[i]))
    if stuck_rows:
        print("WARNING: structural infeasibility — rows whose mask only points to zero-target columns:")
        for i, name, ui in stuck_rows:
            allowed = [coicop_labels[j] for j in np.where(A0[i, :] > 0)[0]]
            print(f"  row {i} ({name}): target={ui:,.2f}, allowed cols (all v=0): {allowed}")

    stuck_cols = []
    for j in np.where(v > 0)[0]:
        allowed_rows = np.where(A0[:, j] > 0)[0]
        if len(allowed_rows) > 0 and u[allowed_rows].sum() == 0:
            stuck_cols.append((j, coicop_labels[j], v[j]))
    if stuck_cols:
        print("WARNING: structural infeasibility — cols whose mask only points to zero-target rows:")
        for j, name, vj in stuck_cols:
            print(f"  col {j} ({name}): target={vj:,.2f}")

    A = A0.copy()
    converged = False
    iterations = 0
    row_err = np.inf
    col_err = np.inf

    if mode == "columns":
        # One-pass column-only distribution.
        # Each column sums exactly to v[j]; mass is split across the rows where
        # mask=1, weighted by the row target u[i] so larger sectors get more.
        weights = A0 * u[:, None]                # shape (n, n_cols)
        col_w_sums = weights.sum(axis=0)
        col_w_sums = np.where(col_w_sums == 0, 1.0, col_w_sums)
        A = weights * (v / col_w_sums)[None, :]

        col_err = np.max(np.abs(A.sum(axis=0) - v))
        row_err = np.max(np.abs(A.sum(axis=1) - u))
        converged = col_err < tol               # only columns are required
        iterations = 1
        print(f"Columns-only mode: col err {col_err:.2e}, row err {row_err:.2e} "
              "(rows are best-effort)")

    elif mode == "rows":
        # One-pass row-only distribution.
        # Each row sums exactly to u[i]; mass is split across the columns where
        # mask=1, weighted by the column target v[j] so larger COICOP categories
        # get more.
        weights = A0 * v[None, :]                # shape (n, n_cols)
        row_w_sums = weights.sum(axis=1)
        row_w_sums = np.where(row_w_sums == 0, 1.0, row_w_sums)
        A = weights * (u / row_w_sums)[:, None]

        row_err = np.max(np.abs(A.sum(axis=1) - u))
        col_err = np.max(np.abs(A.sum(axis=0) - v))
        converged = row_err < tol               # only rows are required
        iterations = 1
        print(f"Rows-only mode: row err {row_err:.2e}, col err {col_err:.2e} "
              "(columns are best-effort)")

    elif mode == "ras":
        for it in range(max_iter):
            # Row scaling
            row_sums = A.sum(axis=1)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            A = A * (u / row_sums)[:, None]

            # Column scaling
            col_sums = A.sum(axis=0)
            col_sums = np.where(col_sums == 0, 1.0, col_sums)
            A = A * (v / col_sums)[None, :]

            # Convergence check
            row_err = np.max(np.abs(A.sum(axis=1) - u))
            col_err = np.max(np.abs(A.sum(axis=0) - v))

            # Progress log every 100 iterations
            if (it + 1) % 100 == 0:
                print(f"  iter {it+1:5d}: row err {row_err:.3e}, col err {col_err:.3e}")

            if row_err < tol and col_err < tol:
                converged = True
                iterations = it + 1
                print(f"RAS converged in {iterations} iterations "
                      f"(row err {row_err:.2e}, col err {col_err:.2e})")
                break
        else:
            iterations = max_iter
    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'ras', 'columns', or 'rows'.")

    if mode == "ras" and not converged:
        print(f"WARNING: RAS did not converge after {max_iter} iterations "
              f"(row err {row_err:.2e}, col err {col_err:.2e})")
        # Show which rows are most off so the user can see structural issues
        actual_row_sums = A.sum(axis=1)
        residuals = u - actual_row_sums
        order = np.argsort(-np.abs(residuals))
        print("\nTop 10 rows with largest residual (target vs achieved):")
        print(f"  {'sector':<10}  {'target':>12}  {'achieved':>12}  {'residual':>12}  cols-with-1s (and v[j])")
        for i in order[:10]:
            allowed = np.where(A0[i, :] > 0)[0]
            cols_info = ", ".join(f"{coicop_labels[j]}(v={v[j]:.1f})" for j in allowed)
            print(f"  {sector_labels[i]:<10}  {u[i]:>12,.2f}  "
                  f"{actual_row_sums[i]:>12,.2f}  {residuals[i]:>12,.2f}  {cols_info}")

    # Wrap as DataFrame with sector codes as rows, COICOP codes as columns
    balanced = pd.DataFrame(A, index=sector_labels, columns=coicop_labels)

    return {
        "balanced": balanced,
        "CA_Y_hh": CA_Y_hh,
        "CA_Y_hh_aggr": pd.Series(CA_Y_hh_aggr, index=sector_labels, name="CA_Y_hh_aggr"),
        "shs_totals_eur": pd.Series(shs_totals_eur, index=coicop_labels, name="shs_totals_eur"),
        "shs_total_eur": shs_total_eur,
        "shs_totals_adj": pd.Series(shs_totals_adj, index=coicop_labels, name="shs_totals_adj"),
        "ca_y_hh_total": ca_y_hh_total,
        "diff": diff,
        "converged": converged,
        "iterations": iterations,
    }


# %%
# ---------------------------------------------------------------------------
# Standalone usage (only runs when this file is executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Pull data the same way Energy_Model.py does
    from Figaro_economic_data import figaro_economic_data
    from SHS_EDM_2019_data import shs_2019_data
    from NACE_COICOP_mask import nace_to_coicop_mask

    figaro_result = figaro_economic_data(
        figaro_folder=r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents - 403001-01 Utvikling og vedlikehold klimakost\General\Utslippsfaktorer\Kryssløpsanalyse\FIGARO data",
        first_year=2019,
        last_year=2019,
        edition_economic_data=2501,
    )

    SHS_2019_data = shs_2019_data(
        sas_path=r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents\additional_projects\Canada_hh_energy\Data\OneDrive_2026-03-09\Survey of Household Spending\SHS_EDM_2019\Data\SAS\pumf_shs2019.sas7bdat"
    )

    NACE_to_COICOP_mask = nace_to_coicop_mask(
        bridge_path=r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents\additional_projects\Canada_hh_energy\Data\COICOP_to_FIGARO_Bridge.xlsx",
        sectors=figaro_result["meta"].sectors,
        shs_data=SHS_2019_data,
    )

    result = ras_balancing(
        nace_coicop_mask=NACE_to_COICOP_mask,
        shs_totals_cad=SHS_2019_data,
        Y=figaro_result["Y"],
        regions=figaro_result["meta"].regions,
        m=figaro_result["m"],
        n=figaro_result["n"],
        k=figaro_result["k"],
        year_idx=0,
        exchange_rate=1.49,
        country_code="CA",
    )

    print(f"\nBalanced matrix shape: {result['balanced'].shape}")
    print(result["balanced"].head())
