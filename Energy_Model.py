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
# Load NACE-to-COICOP bridge matrices (75-cat plus aggregation masks)
from NACE_COICOP_mask import nace_to_coicop_mask, load_coicop_aggregation_masks

BRIDGE_PATH = r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents\additional_projects\Canada_hh_energy\Data\COICOP_to_FIGARO_Bridge.xlsx"

NACE_to_COICOP_mask = nace_to_coicop_mask(
    bridge_path=BRIDGE_PATH,
    sectors=sectors,
    shs_data=SHS_2019_data,
)

# 4 aggregation masks from tab COICOP_to_COICOPaggr:
#   COICOP75_to_COICOP32 (32 x 75) - aggregates 75 -> 32 COICOP categories
#   COICOP75_to_COICOP12 (12 x 75) - aggregates 75 -> 12 COICOP categories
#   NACE_to_COICOP32     (62 x 32) - bridge NACE sectors -> 32 COICOP
#   NACE_to_COICOP12     (62 x 12) - bridge NACE sectors -> 12 COICOP
aggr_masks = load_coicop_aggregation_masks(BRIDGE_PATH)
COICOP75_to_COICOP32_mask = aggr_masks["COICOP75_to_COICOP32"]
COICOP75_to_COICOP12_mask = aggr_masks["COICOP75_to_COICOP12"]
NACE_to_COICOP32_mask = aggr_masks["NACE_to_COICOP32"]
NACE_to_COICOP12_mask = aggr_masks["NACE_to_COICOP12"]


# %%
# Compute SHS expenditure totals (CAD) at each aggregation level
# Skip CaseID and WeightD (first two columns) and sum across all households.
SHS_2019_75cat = SHS_2019_data.iloc[:, 2:].sum().to_numpy(dtype=float)   # shape (75,)

# Aggregate to 32 and 12 categories using the COICOP aggregation masks.
# Each row of the aggregation mask is a 0/1 indicator selecting which 75-cat
# entries roll up into that aggregated category.
SHS_2019_32cat = COICOP75_to_COICOP32_mask @ SHS_2019_75cat              # shape (32,)
SHS_2019_12cat = COICOP75_to_COICOP12_mask @ SHS_2019_75cat              # shape (12,)

print(f"SHS_2019_75cat shape: {SHS_2019_75cat.shape}, sum = {SHS_2019_75cat.sum():,.2f}")
print(f"SHS_2019_32cat shape: {SHS_2019_32cat.shape}, sum = {SHS_2019_32cat.sum():,.2f}")
print(f"SHS_2019_12cat shape: {SHS_2019_12cat.shape}, sum = {SHS_2019_12cat.sum():,.2f}")


# %%
# RAS balancing — choose which COICOP aggregation level to use
from RAS_balancing import ras_balancing

# Switch: 75, 32, or 12
RAS_LEVEL = 75

if RAS_LEVEL == 75:
    nace_mask = NACE_to_COICOP_mask
    shs_totals = SHS_2019_data
elif RAS_LEVEL == 32:
    nace_mask = NACE_to_COICOP32_mask
    shs_totals = SHS_2019_32cat
elif RAS_LEVEL == 12:
    nace_mask = NACE_to_COICOP12_mask
    shs_totals = SHS_2019_12cat
else:
    raise ValueError(f"RAS_LEVEL must be 75, 32, or 12 — got {RAS_LEVEL}")

print(f"\nRunning RAS at COICOP-{RAS_LEVEL} level")

ras_result = ras_balancing(
    nace_coicop_mask=nace_mask,
    shs_totals_cad=shs_totals,
    Y=Y,
    regions=regions,
    m=result["m"],
    n=result["n"],
    k=result["k"],
    year_idx=0,
    exchange_rate=1.49,  # 1 EUR = 1.49 CAD in 2019 (Bank of Canada annual avg)
    country_code="CA",
    mode="rows",         # row-only distribution (columns are best-effort)
)

# Pull intermediate values out for inspection in the Variables panel
balanced_hh = ras_result["balanced"]
CA_Y_hh = ras_result["CA_Y_hh"]
CA_Y_hh_aggr = ras_result["CA_Y_hh_aggr"]
shs_totals_eur = ras_result["shs_totals_eur"]
shs_total_eur = ras_result["shs_total_eur"]
shs_totals_adj = ras_result["shs_totals_adj"]

print(f"\nBalanced matrix shape: {balanced_hh.shape}")
print(f"Converged: {ras_result['converged']}, iterations: {ras_result['iterations']}")
# %%
