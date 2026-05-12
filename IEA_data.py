"""
IEA World Energy Balances data — load only what we need.

Filters applied while streaming through the 5 GB file:
  - Unit (column 0 OR column 1) == "TJ"
  - TIME == 2019
"""

# %%
import pandas as pd

CSV_PATH = r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents\additional_projects\Canada_hh_energy\Data\WBAL-2023-1-EN-20231013T115337.csv"

# Read just the header to find the column names
columns = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
unit_col_0 = columns[0]
unit_col_1 = columns[1]
print(f"Unit columns: {unit_col_0!r}, {unit_col_1!r}")


# %%
# Stream the file in chunks and keep only rows where Unit is TJ and TIME is 2019
chunks = []
for chunk in pd.read_csv(CSV_PATH, chunksize=500_000):
    mask = (
        ((chunk[unit_col_0].astype(str) == "TJ") | (chunk[unit_col_1].astype(str) == "TJ"))
        & (chunk["TIME"] == 2019)
    )
    chunks.append(chunk[mask])

IEA_2019_TJ = pd.concat(chunks, ignore_index=True)
print(f"IEA_2019_TJ shape: {IEA_2019_TJ.shape}")


# %%
# Show the result so it appears in the Variables panel for browsing
IEA_2019_TJ
# %%
