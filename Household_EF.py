"""
Household_EF - household energy footprint downstream of Energy_Model.

Runs Energy_Model.py once and caches the four needed objects to a pickle
on disk. On subsequent runs, if none of the source files have changed,
the cache is loaded instantly instead of re-running the full model.

Objects exposed:
  - e_COICOP_TJperMlnEur       : combined indir + dir intensity (TJ / mln EUR)
  - E_COICOP_EF_TJ             : combined indir + dir energy footprint (TJ)
  - SHS_2019_data_CAD_WGHT     : weighted SHS spends in CAD
  - SHS_2019_data_Mln_EUR_WGHT : weighted SHS spends in million EUR
  - NACE_to_COICOP_mask        : 0/1 bridge mask (62 NACE by 75 COICOP)

Delete the cache file (printed at startup) to force a rebuild.
"""

# %% Cell 1 - Load Energy_Model results (cached)
import os
import pickle
import runpy

_HERE = os.path.dirname(__file__)
_ENERGY_MODEL_PATH = os.path.join(_HERE, "Energy_Model.py")
_CACHE_PATH = os.path.join(_HERE, "_cache_household_ef.pkl")

# Files whose mtime invalidates the cache (Energy_Model.py + its local imports).
_SOURCE_FILES = [
    "Energy_Model.py",
    "Figaro_economic_data.py",
    "SHS_EDM_2019_data.py",
    "NACE_COICOP_mask.py",
    "RAS_balancing.py",
    "Indir_E_e.py",
]
_KEYS = [
    "e_COICOP_TJperMlnEur",
    "E_COICOP_EF_TJ",
    "SHS_2019_data_CAD_WGHT",
    "SHS_2019_data_Mln_EUR_WGHT",
    "NACE_to_COICOP_mask",
]


def _cache_is_fresh() -> bool:
    if not os.path.exists(_CACHE_PATH):
        return False
    cache_mtime = os.path.getmtime(_CACHE_PATH)
    for name in _SOURCE_FILES:
        path = os.path.join(_HERE, name)
        if os.path.exists(path) and os.path.getmtime(path) > cache_mtime:
            return False
    return True


print(f"Cache file: {_CACHE_PATH}")

if _cache_is_fresh():
    print("Loading cached Energy_Model outputs...")
    with open(_CACHE_PATH, "rb") as f:
        _data = pickle.load(f)
    print("Cache loaded.")
else:
    print("Cache missing or stale -- running Energy_Model.py (progress below)")
    print("-" * 70)
    _ns = runpy.run_path(_ENERGY_MODEL_PATH, run_name="__energy_model__")
    print("-" * 70)
    _data = {k: _ns[k] for k in _KEYS}
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cached {len(_KEYS)} objects to disk for future runs.")
    del _ns

e_COICOP_TJperMlnEur       = _data["e_COICOP_TJperMlnEur"]
E_COICOP_EF_TJ             = _data["E_COICOP_EF_TJ"]
SHS_2019_data_CAD_WGHT     = _data["SHS_2019_data_CAD_WGHT"]
SHS_2019_data_Mln_EUR_WGHT = _data["SHS_2019_data_Mln_EUR_WGHT"]
NACE_to_COICOP_mask        = _data["NACE_to_COICOP_mask"]
del _data

print(
    f"Ready: e_COICOP_TJperMlnEur {e_COICOP_TJperMlnEur.shape}, "
    f"E_COICOP_EF_TJ {E_COICOP_EF_TJ.shape}, "
    f"SHS_CAD {SHS_2019_data_CAD_WGHT.shape}, "
    f"SHS_EUR {SHS_2019_data_Mln_EUR_WGHT.shape}, "
    f"NACE_to_COICOP_mask {NACE_to_COICOP_mask.shape}"
)

# %%
