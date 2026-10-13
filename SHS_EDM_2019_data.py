"""
SHS 2019 data loader.

Loads the Survey of Household Spending (SHS) 2019 SAS file and produces two
weighted-expenditure DataFrames:

  - SHS_2019_data_CAD_WGHT     : raw CAD values per household, each weighted
                                  by the household weight (WeightD).
  - SHS_2019_data_Mln_EUR_WGHT : the same data converted to million EUR using
                                  the supplied exchange rate.

Each row is one survey household; CaseID and WeightD are kept on the left,
followed by the 75 expenditure columns (already weighted).

Example usage from another script:

    from SHS_EDM_2019_data import shs_2019_data

    shs = shs_2019_data(
        sas_path=r"C:\\path\\to\\pumf_shs2019.sas7bdat",
        exchange_rate=1.49,
    )
    SHS_2019_data_CAD_WGHT     = shs["CAD_WGHT"]
    SHS_2019_data_Mln_EUR_WGHT = shs["Mln_EUR_WGHT"]
"""

# %%
import time
import pandas as pd


def shs_2019_data(sas_path, exchange_rate=1.49):
    """
    Load the SHS 2019 PUMF SAS file and return weighted-expenditure DataFrames
    in two currencies (CAD and million EUR).

    Each expenditure column is multiplied by the household weight (WeightD).
    The result is returned in two flavours:
      - CAD_WGHT     : values in CAD per household, weighted.
      - Mln_EUR_WGHT : the CAD values divided by `exchange_rate` and by
                       1,000,000 to give million EUR per household, weighted.

    Parameters
    ----------
    sas_path : str
        Full path to the SHS .sas7bdat file.
    exchange_rate : float
        CAD per EUR (default 1.49, the 2019 Bank of Canada annual average).

    Returns
    -------
    dict with keys:
        "CAD_WGHT"     : pd.DataFrame
            CaseID, WeightD, then 75 weighted expenditure columns (in CAD).
        "Mln_EUR_WGHT" : pd.DataFrame
            CaseID, WeightD, then 75 weighted expenditure columns (in million EUR).
    """
    timer_start = time.time()

    # Load the SAS file
    df = pd.read_sas(sas_path, format="sas7bdat", encoding="latin1")

    # Expenditure columns to weight (see COICOP_to_FIGARO_Bridge_2019.xlsx, tab
    # COICOP_to_FIGARO_Bridge, row 8)
    cols = [
        'FD003', 'FD990', 'SH003', 'SH011', 'SH082', 'SH991', 'SH015', 'SH016',
        'SH031', 'SH032', 'SH033', 'SH034', 'SH042', 'SH061', 'SH044', 'SH062',
        'SH060', 'SH046', 'SH047', 'SH050', 'CS003', 'CS007', 'CS008', 'CS020',
        'HO002', 'HO003', 'HO010', 'HO014', 'HO018', 'HO022', 'CC001', 'HF002',
        'HE001', 'HE017', 'HE020', 'CL030', 'TR003', 'TR020', 'TR031', 'TR085',
        'TR033', 'TR034', 'TR071', 'TR036', 'TR038', 'TR039', 'TR070', 'HC002_D',
        'HC002_C', 'HC022', 'PC001', 'RE124', 'RE990', 'RE006', 'RE007', 'RE010_D',
        'RE010_C', 'RE016', 'RE022', 'RE032', 'RE040', 'RE062', 'RE063', 'RE066',
        'RE067', 'RE090', 'RE074', 'RE140', 'RV001', 'ED002', 'RO001', 'TA018',
        'GC001', 'ME039', 'ME040',
    ]

    # Multiply each column by WeightD and tag with _wght suffix
    weighted = df[cols].multiply(df['WeightD'], axis=0)
    weighted.columns = [c + '_wght' for c in cols]

    # CAD version: CaseID, WeightD, then weighted expenditures (in CAD)
    SHS_CAD = pd.concat([df[['CaseID', 'WeightD']], weighted], axis=1)

    # Million-EUR version: same shape, expenditures divided by exchange rate
    # and 1,000,000. CaseID and WeightD are NOT converted.
    weighted_eur = weighted / exchange_rate / 1_000_000
    SHS_EUR = pd.concat([df[['CaseID', 'WeightD']], weighted_eur], axis=1)

    elapsed = time.time() - timer_start
    print(f"Imported SHS_2019_data in {elapsed:.2f} seconds "
          f"(exchange rate {exchange_rate} CAD/EUR)")
    return {"CAD_WGHT": SHS_CAD, "Mln_EUR_WGHT": SHS_EUR}


# %%
# ---------------------------------------------------------------------------
# Standalone usage — only runs when this file is executed directly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SAS_PATH = r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents\additional_projects\Canada_hh_energy\Data\OneDrive_2026-03-09\Survey of Household Spending\SHS_EDM_2019\Data\SAS\pumf_shs2019.sas7bdat"

    shs = shs_2019_data(sas_path=SAS_PATH, exchange_rate=1.49)
    print(shs["CAD_WGHT"].head())
    print(f"CAD shape:     {shs['CAD_WGHT'].shape}")
    print(f"Mln EUR shape: {shs['Mln_EUR_WGHT'].shape}")