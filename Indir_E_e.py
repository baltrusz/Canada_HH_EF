"""
Indirect energy intensities and household indirect energy footprint.

Builds (in order):
  1. sum_Z_sup_E             — column sums of Z over the energy-supplying
                               NACE rows (B, C19, D35) across all countries.
  2. sum_Z_sup_E_IEAsect     — per-country distribution of those supplies
                               across the 23 IEA sectors via NACE_to_IEA_mask.
  3. shares_sup_IEA_NACE     — row-normalised shares so each per-country
                               IEA row sums to 1.
  4. E_TJ                    — IEA total final consumption distributed over
                               the m*n country-sector grid.
  5. e_NACE                  — energy intensity = E_TJ / X (TJ per million EUR).
  6. L                       — Leontief inverse = (I - A)^-1, A = Z/X.
  7. hh_EF_NACE_indir        — household indirect energy footprint
                               = e_NACE * (L @ Y_hh)  (TJ per country-sector).

Example usage:

    from Indir_E_e import indir_E_e

    res = indir_E_e(
        Z=Z, X=X, Y_hh=Y_hh,
        sectors=sectors,
        NACE_to_IEA_mask=NACE_to_IEA_mask,
        IEA_FC_tot=IEA_FC_tot,
        m=m, n=n,
    )
    E_TJ, e_NACE, L, hh_EF_NACE_indir = (
        res["E_TJ"], res["e_NACE"], res["L"], res["hh_EF_NACE_indir"]
    )
"""

# %%
import numpy as np


def indir_E_e(
    Z,
    X,
    Y_hh,
    sectors,
    NACE_to_IEA_mask,
    IEA_FC_tot,
    m,
    n,
    sup_codes=("B", "C19", "D35"),
    year_idx=0,
):
    """
    Compute indirect energy extension, intensity, Leontief inverse, and
    household indirect energy footprint.

    Parameters
    ----------
    Z : np.ndarray, shape (m*n, m*n, n_years)
        FIGARO intermediate-flows matrix.
    X : np.ndarray, shape (m*n, 1, n_years)
        FIGARO total-output vector.
    Y_hh : np.ndarray, shape (m*n,)
        Household final-demand vector for the chosen country.
    sectors : pd.DataFrame
        Meta sheet 'sector'. Column 1 (index 1) holds the NACE codes.
    NACE_to_IEA_mask : pd.DataFrame
        Bridge matrix from IEA_NACE_mapping.xlsx. Column 0 = NACE code,
        column 1 = NACE name, columns 2+ = 0/1 mapping to 23 IEA sectors.
    IEA_FC_tot : pd.DataFrame
        IEA total final consumption. Last column = value (TJ). Rows ordered
        as (country, IEA_sector) pairs: m * n_iea total rows.
    m, n : int
        Number of regions (m=46) and NACE sectors (n=62).
    sup_codes : tuple of str
        NACE sector codes for the "energy-supplying" rows. Default
        ("B", "C19", "D35").
    year_idx : int
        Year slice of Z/X to use (default 0).

    Returns
    -------
    dict with keys: E_TJ, e_NACE, L, hh_EF_NACE_indir
    """
    # ---- 1. sum_Z_sup_E: column sums of Z over the energy-supply rows ----
    sector_codes = sectors.iloc[:, 1].astype(str).str.strip().tolist()
    sup_indices = np.array([sector_codes.index(code) for code in sup_codes])

    country_offsets = np.arange(m) * n
    all_rows = (country_offsets[:, None] + sup_indices[None, :]).ravel()
    sum_Z_sup_E = Z[all_rows, :, year_idx].sum(axis=0)             # (m*n,)

    # ---- 2. Distribute across IEA sectors per country ----
    NACE_to_IEA_data = NACE_to_IEA_mask.iloc[:, 2:].fillna(0).to_numpy(dtype=float)  # (n, 23)
    NACE_to_IEA_T = NACE_to_IEA_data.T                              # (23, n)
    n_iea = NACE_to_IEA_T.shape[0]

    sum_Z_2d = sum_Z_sup_E.reshape(m, n)                           # (m, n)
    sum_Z_sup_E_IEAsect = (
        NACE_to_IEA_T[:, None, :] * sum_Z_2d[None, :, :]           # (23, m, n)
    ).reshape(n_iea, m * n)

    # ---- 3. Row-normalised shares (per IEA-sector per country) ----
    ZIE = sum_Z_sup_E_IEAsect.reshape(n_iea, m, n)
    row_sums_per_country = ZIE.sum(axis=2)
    row_sums_safe = np.where(row_sums_per_country == 0, 1.0, row_sums_per_country)
    shares = ZIE / row_sums_safe[:, :, None]                       # (23, m, n)

    # ---- 4. E_TJ — energy extension in TJ over the m*n grid ----
    ifc_values = IEA_FC_tot.iloc[:, -1].to_numpy(dtype=float)
    if ifc_values.shape[0] != m * n_iea:
        raise ValueError(
            f"Expected {m*n_iea} rows in IEA_FC_tot, got {ifc_values.shape[0]}"
        )
    IEA_FC_tot_per_country = ifc_values.reshape(m, n_iea)
    E_TJ_2d = np.einsum("ci,icn->cn", IEA_FC_tot_per_country, shares)
    E_TJ = E_TJ_2d.reshape(-1)                                      # (m*n,)

    # Sum check: total of E_TJ should equal total of IEA_FC_tot
    ifc_total = ifc_values.sum()
    big_total = E_TJ.sum()
    diff = abs(ifc_total - big_total)
    print(f"E_TJ vs IEA_FC_tot total: {big_total:,.2f} vs {ifc_total:,.2f} "
          f"(|diff|={diff:,.4f})")
    if diff > 1e-3 * abs(ifc_total):
        print("  WARNING: totals do not match — likely some rows have empty mask coverage.")

    # ---- 5. e_NACE — energy intensity (TJ per million EUR) ----
    X_vec = X[:, 0, year_idx]                                       # (m*n,)
    e_NACE = E_TJ / X_vec

    # ---- 6. Leontief inverse L = (I - A)^-1 with A = Z * diag(1/X) ----
    A_mat = Z[:, :, year_idx] / X_vec[None, :]
    L = np.linalg.inv(np.eye(A_mat.shape[0]) - A_mat)

    # ---- 7. Household indirect energy footprint = e_NACE * (L @ Y_hh) ----
    hh_EF_NACE_indir = e_NACE * (L @ Y_hh)
    print(f"hh_EF_NACE_indir shape: {hh_EF_NACE_indir.shape}, "
          f"total = {hh_EF_NACE_indir.sum():,.2f} TJ")

    return {
        "E_TJ": E_TJ,
        "e_NACE": e_NACE,
        "L": L,
        "hh_EF_NACE_indir": hh_EF_NACE_indir,
    }