"""
SHS data loader for any survey year (2017, 2019, 2021).

Generalises SHS_EDM_2019_data.shs_2019_data: the same weighting and currency
conversion, but the expenditure columns come from the year's bridge file
(row 8 of COICOP_to_FIGARO_Bridge_<year>.xlsx) instead of a hard-coded list,
and the raw file is read via year_config.load_shs_raw.

Produces two weighted-expenditure DataFrames:

  - CAD_WGHT     : raw CAD values per household, each weighted by the
                   household weight (WeightD).
  - Mln_EUR_WGHT : the same data converted to million EUR using the
                   year's exchange rate.

Each row is one survey household; CaseID and WeightD are kept on the left,
followed by the expenditure columns (already weighted, "_wght" suffix).

Example usage from another script:

    from SHS_data import shs_data

    shs = shs_data(data_dir=DATA_DIR, year=2021)
    SHS_data_CAD_WGHT     = shs["CAD_WGHT"]
    SHS_data_Mln_EUR_WGHT = shs["Mln_EUR_WGHT"]
"""

# %%
import time
import pandas as pd

from year_config import YEAR_CONFIG, load_shs_raw, bridge_coicop_codes


def shs_data(data_dir, year, raw=None, cols=None, exchange_rate=None):
    """
    Load the SHS PUMF for `year` and return weighted-expenditure DataFrames
    in two currencies (CAD and million EUR).

    Parameters
    ----------
    data_dir : str
        DATA_DIR holding the bridge files and the SHS folder.
    year : int
        Survey year (key of YEAR_CONFIG).
    raw : pd.DataFrame, optional
        Already-loaded output of load_shs_raw(data_dir, year). Loaded if None.
    cols : list of str, optional
        Expenditure columns to weight. Defaults to the bridge codes for `year`.
    exchange_rate : float, optional
        CAD per EUR. Defaults to YEAR_CONFIG[year]["exchange_rate"].

    Returns
    -------
    dict with keys "CAD_WGHT" and "Mln_EUR_WGHT" (same layout as
    shs_2019_data): CaseID, WeightD, then weighted expenditure columns.
    """
    timer_start = time.time()
    if raw is None:
        raw = load_shs_raw(data_dir, year)
    if cols is None:
        cols = bridge_coicop_codes(data_dir, year)
    if exchange_rate is None:
        exchange_rate = YEAR_CONFIG[year]["exchange_rate"]

    missing = [c for c in cols if c not in raw.columns]
    if missing:
        raise ValueError(f"SHS {year}: bridge codes missing in microdata: {missing}")

    # Multiply each column by WeightD and tag with _wght suffix
    weighted = raw[cols].astype(float).multiply(raw["WeightD"], axis=0)
    weighted.columns = [c + "_wght" for c in cols]

    # CAD version: CaseID, WeightD, then weighted expenditures (in CAD)
    SHS_CAD = pd.concat([raw[["CaseID", "WeightD"]], weighted], axis=1)

    # Million-EUR version: same shape, expenditures divided by exchange rate
    # and 1,000,000. CaseID and WeightD are NOT converted.
    weighted_eur = weighted / exchange_rate / 1_000_000
    SHS_EUR = pd.concat([raw[["CaseID", "WeightD"]], weighted_eur], axis=1)

    elapsed = time.time() - timer_start
    print(f"Imported SHS_{year}_data in {elapsed:.2f} seconds "
          f"({len(cols)} expenditure columns, exchange rate {exchange_rate} CAD/EUR)")
    return {"CAD_WGHT": SHS_CAD, "Mln_EUR_WGHT": SHS_EUR}
