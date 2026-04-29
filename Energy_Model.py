# Load economic IO data from FIGARO
from Figaro_economic_data import figaro_economic_data

result = figaro_economic_data(
        figaro_folder=r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents - 403001-01 Utvikling og vedlikehold klimakost\General\Utslippsfaktorer\Kryssløpsanalyse\FIGARO data",
        first_year=2019,
        last_year=2019,
        edition_economic_data=2501,
    )
X = result["X"]
Y = result["Y"]
Z = result["Z"]
meta = result["meta"]   # access as meta.regions, meta.sectors, meta.FDcategories

# Pull meta sheets out as top-level DataFrames so they show up in the
# Variables panel with a grid icon for browsing
regions = meta.regions
sectors = meta.sectors
FDcategories = meta.FDcategories

#Load SHS 2019 household spends
from SHS_EDM_2019_data import shs_2019_data

SHS_2019_data = shs_2019_data(
        sas_path=r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents\additional_projects\Canada_hh_energy\Data\OneDrive_2026-03-09\Survey of Household Spending\SHS_EDM_2019\Data\SAS\pumf_shs2019.sas7bdat"
    )



# %%
# Load NACE-to-COICOP bridge matrix (many-to-many, ones and zeros)
# Range B8:BZ70 in tab COICOP_to_FIGARO_Bridge
import pandas as pd

BRIDGE_PATH = r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents\additional_projects\Canada_hh_energy\Data\COICOP_to_FIGARO_Bridge.xlsx"

NACE_to_COICOP_mask = pd.read_excel(
    BRIDGE_PATH,
    sheet_name="COICOP_to_FIGARO_Bridge",
    header=0,           # row 8 contains column names
    skiprows=7,         # skip rows 1-7, so row 8 becomes the header
    nrows=62,           # rows 9 to 70 inclusive = 62 data rows
    usecols="B:BZ",     # columns B through BZ = 77 columns
)

print(f"NACE_to_COICOP_mask shape: {NACE_to_COICOP_mask.shape}")
