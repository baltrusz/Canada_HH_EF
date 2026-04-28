#%%
import pandas as pd

# Local path to the SAS file synced via OneDrive
SAS_PATH = r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents\additional_projects\Canada_hh_energy\Data\OneDrive_2026-03-09\Survey of Household Spending\SHS_EDM_2019\Data\SAS\pumf_shs2019.sas7bdat"

# Load the SAS file directly into a DataFrame
df = pd.read_sas(SAS_PATH, format="sas7bdat", encoding="latin1")

# Print the first 5 rows to verify the data loaded correctly
print(df.head())

# %%
# Column-wise sums for all numeric columns
sum_exp = df.select_dtypes(include='number').sum()
print(sum_exp)
sum_exp.to_frame().to_clipboard()

# %%
# Summary statistics (count, mean, std, min, max, etc.) for all columns
df.describe()

# %%
# Columns to multiply by WeightID
cols = [
    'FD001', 'SH003', 'SH010', 'SH031', 'SH032', 'SH033', 'SH034', 'SH042',
    'SH061', 'SH044', 'SH062', 'SH060', 'SH046', 'SH047', 'SH050', 'CS003',
    'CS007', 'CS008', 'CS020', 'HO002', 'HO003', 'HO010', 'HO014', 'HO018',
    'HO022', 'CC001', 'HF002', 'HE001', 'HE017', 'HE020', 'CL030', 'TR003',
    'TR020', 'TR031', 'TR085', 'TR033', 'TR034', 'TR071', 'TR036', 'TR038',
    'TR039', 'TR070', 'HC002_D', 'HC002_C', 'HC022', 'PC001', 'RE124', 'RE990',
    'RE006', 'RE007', 'RE010_D', 'RE010_C', 'RE016', 'RE022', 'RE032', 'RE040',
    'RE062', 'RE063', 'RE066', 'RE067', 'RE090', 'RE074', 'RE140', 'RV001',
    'ED002', 'RO001', 'TA018', 'GC001', 'ME039', 'ME040',
]

# Multiply each column by WeightID
weighted = df[cols].multiply(df['WeightD'], axis=0)
weighted.columns = [c + '_wght' for c in cols]

# Build new dataframe with CaseID, WeightID and weighted columns
df_wght = pd.concat([df[['CaseID', 'WeightD']], weighted], axis=1)
df_wght.head()


# %%
# Column-wise sums for all numeric columns
sum_exp_selectedwght = df_wght.select_dtypes(include='number').sum()
print(sum_exp_selectedwght)
sum_exp_selectedwght.to_frame().to_clipboard()
# %%
