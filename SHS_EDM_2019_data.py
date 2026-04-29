"""
SHS 2019 data loader.

Loads the Survey of Household Spending (SHS) 2019 SAS file, multiplies the
selected expenditure columns by the household weight (WeightD), and returns
the resulting DataFrame.

Example usage from another script:

    from SHS_EDM_2019_data import shs_2019_data

    SHS_2019_data = shs_2019_data(
        sas_path=r"C:\\path\\to\\pumf_shs2019.sas7bdat"
    )
"""

# %%
import time
import pandas as pd


def shs_2019_data(sas_path):
    """
    Load the SHS 2019 PUMF SAS file and return a weighted-expenditure DataFrame.

    Each selected expenditure column is multiplied by the WeightD column,
    and the resulting DataFrame contains CaseID, WeightD, and all weighted
    columns (suffixed with `_wght`).

    Parameters
    ----------
    sas_path : str
        Full path to the SHS .sas7bdat file.

    Returns
    -------
    pd.DataFrame
        Columns: CaseID, WeightD, <col>_wght for each expenditure column.
    """
    timer_start = time.time()

    # Load the SAS file
    df = pd.read_sas(sas_path, format="sas7bdat", encoding="latin1")

    # Expenditure columns to weight (see COICOP_to_FIGARO_Bridge.xlsx, tab
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

    # Multiply each column by WeightD and rename with _wght suffix
    weighted = df[cols].multiply(df['WeightD'], axis=0)
    weighted.columns = [c + '_wght' for c in cols]

    # Build final DataFrame with CaseID, WeightD, and weighted columns
    result = pd.concat([df[['CaseID', 'WeightD']], weighted], axis=1)

    elapsed = time.time() - timer_start
    print(f"Imported SHS_2019_data in {elapsed:.2f} seconds")
    return result


# %%
# ---------------------------------------------------------------------------
# Standalone usage — only runs when this file is executed directly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SAS_PATH = r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents\additional_projects\Canada_hh_energy\Data\OneDrive_2026-03-09\Survey of Household Spending\SHS_EDM_2019\Data\SAS\pumf_shs2019.sas7bdat"

    SHS_2019_data = shs_2019_data(sas_path=SAS_PATH)
    print(SHS_2019_data.head())
    print(f"Shape: {SHS_2019_data.shape}")
