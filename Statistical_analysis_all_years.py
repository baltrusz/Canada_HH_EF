"""
Compares the household energy footprint (EF) across the survey years 2017, 2019 and 2021.
It answers three questions:

- **A. Total footprint:** how much did it change between the years?
- **B. Categories (COICOP 12):** how did each category change, which changed most and least, and why (more spending or a different energy intensity)?
- **C. Socioeconomic drivers:** which household characteristics does the change depend on, and is that statistically significant?

The single-year analysis is in `Statistical_analysis.ipynb`.

**Before you run.** Run `Household_EF` for all three years at COICOP level 12. It saves the files this notebook reads: `var_for_stat_analysis_12COICOP.pkl`, `..._labels.json` and `..._category_totals.csv`. Then set `DATA_DIR` below. It is needed for Statistics Canada's bootstrap weights.

**Two things to keep in mind when reading the results**

1. **The survey interviews different households each year.** We cannot follow one household over time. "How the change depends on a characteristic" therefore means comparing the same *groups* (for example renters, or the top income decile) across years.
2. **Category totals come from the energy model, not from the survey.** Each category's national footprint is fixed by FIGARO/IEA. The survey only shares it out over households in proportion to what they spend. So the change in the *average* footprint per category is largely set by the model, and the significance tests mainly tell us how the change is spread across households and groups. Section B3 shows how much of each category's change comes from spending and how much from energy intensity.

**Uncertainty.** Every confidence interval and p-value uses Statistics Canada's 1000 bootstrap weights per year. Each weight set mimics a slightly different draw of the sample, and the spread of the results across the 1000 sets is the uncertainty. This respects how the survey was sampled. Section D compares it with the simpler HC1 method.

**Reading p-values.** p < 0.05 means a difference this large would be unlikely if there were truly no difference. Stars: `*` p<0.05, `**` p<0.01, `***` p<0.001. Where many tests run at once (12 categories, or 11 characteristics), the p-values are **Holm-adjusted**: they are made stricter so that, across the whole set, the chance of any false "significant" result stays below 5%.
"""

# %% Setup
import os
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except ImportError:          # running as a plain script
    display = print

import survey_stats as ss
from survey_stats import Est, pct_change, holm, stars
from year_config import load_bsw

# Same DATA_DIR as in Household_EF (only needed for the bootstrap weights).
DATA_DIR = r"C:\Users\marta.baltruszewicz\OneDrive - Asplan Viak\Documents\additional_projects\Canada_hh_energy\Data"

COICOP_LEVEL = 12        # must match coicop_to_coicop_matching in Household_EF
BASE_YEAR = 2017         # changes are measured from this year
N_BOOT = 1000            # bootstrap weights to use (max 1000; 200 runs ~5x faster for a quick look)
FOCUS_CATEGORY = "Transport"   # category followed through every section
UNIT, UNIT_SCALE = "GJ", 1000  # the pickle holds TJ; results are shown in GJ

# Optional price deflator for the spending vs intensity split (B3). None keeps
# nominal CAD. To use it, enter an index per year, e.g. Statistics Canada's
# annual all-items CPI for Canada, 2002 = 100: Table 18-10-0005-01,
# "Consumer Price Index, annual average, not seasonally adjusted",
# https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000501
# Format: {2017: ..., 2019: ..., 2021: ...}. Copy the values from that table.
CPI = None

_HERE = os.path.abspath(os.getcwd())
_stem = os.path.join(_HERE, f"var_for_stat_analysis_{COICOP_LEVEL}COICOP")
df = pd.read_pickle(_stem + ".pkl")
with open(_stem + "_labels.json", encoding="utf-8") as f:
    coicop_label_map = json.load(f)
category_totals = (pd.read_csv(_stem + "_category_totals.csv")
                   if os.path.exists(_stem + "_category_totals.csv") else None)

if "year" not in df.columns:
    raise ValueError("This pickle holds one year only. Rerun Household_EF with "
                     "YEARS = [2017, 2019, 2021].")
YEARS = sorted(int(y) for y in df["year"].unique())
if len(YEARS) < 2:
    raise ValueError(f"Need at least two years in the pickle, found {YEARS}.")
if BASE_YEAR not in YEARS:
    raise ValueError(f"BASE_YEAR={BASE_YEAR} is not in the pickle (years: {YEARS}).")
LAST_YEAR = YEARS[-1]
MAIN_PAIR = (BASE_YEAR, LAST_YEAR)
PAIRS = [(a, b) for a, b in zip(YEARS[:-1], YEARS[1:])]
if MAIN_PAIR not in PAIRS:
    PAIRS.append(MAIN_PAIR)

# --- Harmonise codes so groups mean the same thing in every year ----------
# 2017 only has "Atlantic" (14) for NL, PE, NS and NB; pool them in 2019/2021 too.
df["Prov"] = df["Prov"].astype(str).replace({"10": "14", "11": "14", "12": "14", "13": "14"})
# Major income source 5 ("all sources <= 0") has 2 households in 2021 only: merge into 4 (other).
df["HH_MajIncSrc"] = df["HH_MajIncSrc"].astype(str).replace({"5": "4"})
df["Inc_per_OECDcap"] = df["HH_TotInc"] / df["OECD_scale"]

# --- Outcome definitions ----------------------------------------------------
# per_cap: footprint per OECD adult equivalent, weighted by OECD_cap (people).
# per_hh : footprint per household, weighted by WeightD (households).
UNITS = {
    "per_cap": {"label": "per OECD adult equivalent", "total": "avg_EF_perOECD_cap",
                "prefix": "avg_EF_per_cap__", "weight": "OECD_cap",
                "income_decile": "Inc_OECDcap_dec", "pop": "n_OECD_cap"},
    "per_hh":  {"label": "per household", "total": "avg_EF_per_hh",
                "prefix": "avg_EF_per_hh__", "weight": "WeightD",
                "income_decile": "Inc_hh_dec", "pop": "n_households"},
}
CATS = [c[len("avg_EF_per_cap__"):] for c in df.columns if c.startswith("avg_EF_per_cap__")]
CAT_LABEL = {c: coicop_label_map.get("avg_EF_per_cap__" + c, c) for c in CATS}


def short(label, n=32):
    """Shortened category label for chart axes."""
    return label if len(label) <= n else label[: n - 1].rstrip(" ,") + "…"
if FOCUS_CATEGORY not in CATS:
    raise ValueError(f"FOCUS_CATEGORY must be one of {CATS}")

# --- Bootstrap weights, aligned with the pickle rows -----------------------
_t0 = time.time()
D, W, WB = {}, {}, {}
for y in YEARS:
    d = df[df["year"] == y].reset_index(drop=True)
    bsw = load_bsw(DATA_DIR, y, n_boot=N_BOOT).reindex(d["CaseID"].astype(str).str.zfill(6))
    if bsw.isna().any().any():
        raise ValueError(f"{y}: bootstrap weights missing for some households.")
    bsw = bsw.to_numpy()
    D[y] = d
    W[y] = {"per_hh": d["WeightD"].to_numpy(float), "per_cap": d["OECD_cap"].to_numpy(float)}
    # OECD_cap = WeightD x OECD_scale, so the per-person replicate weights are BSW x OECD_scale.
    WB[y] = {"per_hh": bsw, "per_cap": bsw * d["OECD_scale"].to_numpy(float)[:, None]}

print(f"Years: {YEARS}; base year {BASE_YEAR}; comparisons: {PAIRS}")
print("Households per year:", {y: len(D[y]) for y in YEARS})
print(f"Bootstrap weights: {N_BOOT} per year (loaded in {time.time() - _t0:.1f} s)")
print(f"{len(CATS)} COICOP categories; focus category: {CAT_LABEL[FOCUS_CATEGORY]}")
if category_totals is None:
    print("NOTE: category totals file not found; rerun Household_EF to enable section B3.")

# %% Helpers for tables and charts
YEAR_COLORS = dict(zip(YEARS, ["#2a78d6", "#eb6834", "#1baf7a"]))
LEVEL_NAMES = {
    "Tenure":     {"1": "Owner w/ mortgage", "2": "Owner w/o mortgage", "3": "Renter"},
    "HHType6":    {"1": "One person", "2": "Couple no kids", "3": "Couple w/ kids",
                   "4": "Couple + others", "5": "Lone parent", "6": "Other"},
    "RP_AgeGrp":  {"01": "<30", "02": "30-39", "03": "40-54",
                   "04": "55-64", "05": "65-74", "06": "75+"},
    "RP_Gender":  {"1": "Male", "2": "Female"},
    "RP_Educ":    {"1": "<High school", "2": "High school", "3": "Trade/college",
                   "4": "University", "9": "Not stated"},
    "DwelTyp":    {"1": "Single detached", "2": "Single attached", "3": "Apartment"},
    "SecResYN":   {"1": "Yes", "2": "No"},
    "VehicleYN":  {"1": "Yes", "2": "No"},
    "HH_MajIncSrc": {"1": "Employment income", "2": "Investment income",
                     "3": "Govt transfers", "4": "Other / none"},
    "Prov":       {"14": "Atlantic", "24": "QC", "35": "ON", "46": "MB", "47": "SK",
                   "48": "AB", "59": "BC", "63": "Territories"},
}
DRIVER_NAMES = {
    "INC_DEC": "Income decile", "HHType6": "Household type", "RP_AgeGrp": "Age of reference person",
    "RP_Gender": "Gender of reference person", "RP_Educ": "Education of reference person",
    "DwelTyp": "Dwelling type", "Tenure": "Tenure", "Prov": "Province / region",
    "VehicleYN": "Owns a vehicle", "SecResYN": "Second residence", "HH_MajIncSrc": "Main income source",
}


def level_name(v, lev):
    if v in ("INC_DEC", "Inc_OECDcap_dec", "Inc_hh_dec"):
        return f"D{lev}"
    return LEVEL_NAMES.get(v, {}).get(str(lev), str(lev))


def sorted_levels(values):
    vals = sorted(set(str(v) for v in values))
    try:
        return sorted(vals, key=lambda s: float(s))
    except ValueError:
        return vals


def fmt(x, nd=1):
    return "" if x is None or np.isnan(x) else f"{x:,.{nd}f}"


def fmt_ci(est, nd=1):
    lo, hi = est.ci()
    return f"[{fmt(float(lo), nd)}, {fmt(float(hi), nd)}]"


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def signif_words(p):
    return "statistically significant" if p < 0.05 else "not statistically significant"


def unit_y(y, u, col):
    """Outcome column for unit u in year y, in display units."""
    return D[y][col].to_numpy(float) * UNIT_SCALE


# %% [markdown]
# ## A. How much did the total footprint change?
#
# ### A1. Average footprint by year and the change between years
#
# **What it shows.** The table gives the average (mean) and typical (median) household footprint for each year. The mean is pulled up by a few very large footprints; the median is the household in the middle. Both are shown per OECD adult equivalent and per household. The second table gives the change between years in GJ and in %.
#
# **What the test does.** For each pair of years it asks whether the change in the mean is larger than what sampling chance alone could produce. The change is recomputed with each of the 1000 bootstrap weight sets. The spread of those 1000 values gives the 95% confidence interval and the p-value. This is equivalent to a weighted regression of the footprint on year dummies, which is what "WLS on C(year)" in the plan referred to. The *joint test* asks whether all years have the same mean.
#
# **How to read it.** If the 95% interval for the change does not include 0 (and p < 0.05), the footprint really changed between those years. If the interval includes 0, the survey cannot tell the years apart. Because the model fixes the national totals (see the introduction), the change in the per-person mean mostly mirrors the model's national total divided by the population. The interval says how precisely the survey pins down that average.

# %%
A_level, A_change = [], []
A_est = {}
for u, U in UNITS.items():
    for y in YEARS:
        yv = unit_y(y, u, U["total"])
        A_est[(u, y, "mean")] = ss.wmean(yv, W[y][u], WB[y][u])
        A_est[(u, y, "median")] = ss.wquantiles(yv, W[y][u], WB[y][u], [0.5])[0]
        m, md_ = A_est[(u, y, "mean")], A_est[(u, y, "median")]
        A_level.append({"unit": U["label"], "year": y,
                        f"mean ({UNIT})": fmt(float(m.value)), "mean 95% CI": fmt_ci(m),
                        f"median ({UNIT})": fmt(float(md_.value)), "median 95% CI": fmt_ci(md_)})
    pv = []
    rows = []
    for a, b in PAIRS:
        dm = A_est[(u, b, "mean")] - A_est[(u, a, "mean")]
        pm = pct_change(A_est[(u, b, "mean")], A_est[(u, a, "mean")])
        dmed = pct_change(A_est[(u, b, "median")], A_est[(u, a, "median")])
        A_est[(u, a, b)] = (dm, pm, dmed)
        pv.append(float(dm.pvalue()))
        rows.append({"unit": U["label"], "change": f"{a} -> {b}",
                     f"change in mean ({UNIT})": fmt(float(dm.value)), "95% CI": fmt_ci(dm),
                     "change in mean (%)": fmt(float(pm.value)), "% 95% CI": fmt_ci(pm),
                     "change in median (%)": fmt(float(dmed.value)),
                     "p": fmt_p(pv[-1])})
    for r, p_adj in zip(rows, holm(pv)):
        r["p (Holm)"] = fmt_p(p_adj) + " " + stars(p_adj)
    # Joint test: are all years' means equal?
    diffs = [A_est[(u, y, "mean")] - A_est[(u, BASE_YEAR, "mean")] for y in YEARS if y != BASE_YEAR]
    joint = Est(np.array([float(d.value) for d in diffs]), np.column_stack([d.reps for d in diffs]))
    chi2, dof, pj = ss.wald_test(joint)
    rows.append({"unit": U["label"], "change": f"joint: all {len(YEARS)} years equal",
                 "p": fmt_p(pj), "p (Holm)": f"chi2={chi2:.1f}, df={dof}"})
    A_change.extend(rows)

print("Average footprint by year")
display(pd.DataFrame(A_level))
print("Change between years")
display(pd.DataFrame(A_change).fillna(""))

# Chart: mean with 95% CI per year, one panel per unit.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (u, U) in zip(axes, UNITS.items()):
    for i, y in enumerate(YEARS):
        m = A_est[(u, y, "mean")]
        lo, hi = m.ci()
        ax.bar(i, m.value, color=YEAR_COLORS[y], width=0.6)
        ax.errorbar(i, m.value, yerr=[[m.value - lo], [hi - m.value]], color="#333", capsize=5)
        ax.text(i, hi, f" {m.value:,.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(YEARS)), [str(y) for y in YEARS])
    ax.set_ylabel(f"{UNIT} {U['label']}")
    ax.set_title(f"Mean footprint {U['label']}")
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Mean total footprint by year (bars: 95% confidence interval)")
plt.tight_layout()
plt.show()

for u, U in UNITS.items():
    dm, pm, _ = A_est[(u,) + MAIN_PAIR]
    lo, hi = pm.ci()
    print(f"Reading ({U['label']}): the mean footprint went from "
          f"{A_est[(u, BASE_YEAR, 'mean')].value:,.1f} {UNIT} in {BASE_YEAR} to "
          f"{A_est[(u, LAST_YEAR, 'mean')].value:,.1f} {UNIT} in {LAST_YEAR}, a change of "
          f"{pm.value:+.1f}% (95% CI {lo:+.1f}% to {hi:+.1f}%). This is {signif_words(dm.pvalue())} "
          f"(p = {fmt_p(dm.pvalue())}).")

# %% [markdown]
# ### A2. Did the whole distribution shift, or only the average?
#
# **What it shows.** The footprint at five points of the distribution (percentiles), by year, per OECD adult equivalent. P10 is the household with a lower footprint than 90% of households, P50 is the median and P90 is the high end. The chart overlays the three years' distributions.
#
# **What the test does.** For each percentile, the change from the base year is tested with the bootstrap, as in A1. This is the same as a quantile regression on year. The *KS distance* is the largest vertical gap between two years' cumulative distributions. 0 means identical distributions; 0.1 means that at some footprint level the share of households below it differs by 10 percentage points.
#
# **How to read it.** Similar % changes at every percentile mean the distribution moved as a whole. A larger change at P90 than at P10 means high-footprint households changed more, so the distribution widened (or narrowed, if the reverse).

# %%
QS = [0.10, 0.25, 0.50, 0.75, 0.90]
u = "per_cap"
U = UNITS[u]
Q = {y: ss.wquantiles(unit_y(y, u, U["total"]), W[y][u], WB[y][u], QS) for y in YEARS}
rows = []
for j, q in enumerate(QS):
    r = {"percentile": f"P{int(q * 100)}"}
    for y in YEARS:
        r[f"{y} ({UNIT})"] = fmt(float(Q[y].value[j]))
    for a, b in PAIRS:
        pc = pct_change(Q[b][j], Q[a][j])
        r[f"% {a}->{b}"] = f"{pc.value:+.1f} {stars(float(pc.pvalue()))}"
    rows.append(r)
print(f"Percentiles of the footprint {U['label']}")
display(pd.DataFrame(rows))

for a, b in PAIRS:
    ks = ss.weighted_ks(unit_y(a, u, U["total"]), W[a][u], unit_y(b, u, U["total"]), W[b][u])
    print(f"KS distance {a} vs {b}: {ks:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
hi_x = max(np.quantile(unit_y(y, u, U["total"]), 0.99) for y in YEARS)
bins = np.linspace(0, hi_x, 60)
for y in YEARS:
    axes[0].hist(unit_y(y, u, U["total"]), bins=bins, weights=W[y][u], density=True,
                 histtype="step", linewidth=2, color=YEAR_COLORS[y], label=str(y))
axes[0].set_xlabel(f"Footprint ({UNIT} {U['label']})")
axes[0].set_ylabel("Share of people (density)")
axes[0].set_title("Distribution by year (top 1% cut off)")
axes[0].legend(frameon=False)
x = np.arange(len(QS))
for k, (a, b) in enumerate([p for p in PAIRS if p != MAIN_PAIR] + [MAIN_PAIR]):
    pcs = [pct_change(Q[b][j], Q[a][j]) for j in range(len(QS))]
    vals = np.array([float(p.value) for p in pcs])
    err = np.array([[float(p.value - p.ci()[0]), float(p.ci()[1] - p.value)] for p in pcs]).T
    axes[1].errorbar(x + (k - 1) * 0.12, vals, yerr=err, fmt="o-", capsize=3,
                     color=YEAR_COLORS[b] if (a, b) != MAIN_PAIR else "#333", label=f"{a} -> {b}")
axes[1].axhline(0, color="#999", linewidth=1)
axes[1].set_xticks(x, [f"P{int(q * 100)}" for q in QS])
axes[1].set_ylabel("% change")
axes[1].set_title("Change at each percentile (95% CI)")
axes[1].legend(frameon=False)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

_p10 = pct_change(Q[LAST_YEAR][0], Q[BASE_YEAR][0])
_p90 = pct_change(Q[LAST_YEAR][4], Q[BASE_YEAR][4])
print(f"Reading: from {BASE_YEAR} to {LAST_YEAR} the low end (P10) changed by {_p10.value:+.1f}% "
      f"and the high end (P90) by {_p90.value:+.1f}%. "
      + ("The high end rose more (or fell less) than the low end, so the spread between low and high "
         "footprints widened in relative terms." if _p90.value > _p10.value else
         "The low end rose more (or fell less) than the high end, so the spread between low and high "
         "footprints narrowed in relative terms."))

# %% [markdown]
# ### A3. Inequality: is the footprint more or less evenly spread?
#
# **What it shows.**
# - **Gini coefficient of the footprint** per OECD adult equivalent: 0 means everyone has the same footprint, 1 means one person has all of it.
# - **Concentration index:** the same idea, but people are ranked by *income* (income per OECD adult equivalent). A positive value means the footprint is concentrated among higher-income people. The larger it is, the more the footprint follows income.
# - **Top/bottom ratio:** the mean footprint of the richest 10% (income decile 10) divided by that of the poorest 10% (decile 1).
#
# **What the test does.** Each measure's change from the base year is tested with the bootstrap.
#
# **How to read it.** A falling concentration index or top/bottom ratio means the footprint has become less tied to income. In the left chart, the further a curve sits below the diagonal, the more the footprint is concentrated among the higher-income or higher-footprint people.

# %%
u = "per_cap"
U = UNITS[u]
INEQ = {}
for y in YEARS:
    yv = unit_y(y, u, U["total"])
    dec = D[y][U["income_decile"]].astype(str).to_numpy()
    top = ss.wmean(np.column_stack([yv * (dec == "10"), dec == "10"]), W[y][u], WB[y][u])
    bot = ss.wmean(np.column_stack([yv * (dec == "1"), dec == "1"]), W[y][u], WB[y][u])
    INEQ[y] = {
        "Gini of footprint": ss.gini(yv, W[y][u], WB[y][u]),
        "Concentration index by income": ss.concentration_index(yv, D[y]["Inc_per_OECDcap"], W[y][u], WB[y][u]),
        "Top 10% / bottom 10% income: mean footprint ratio": (top[0] / top[1]) / (bot[0] / bot[1]),
    }
rows = []
for name in INEQ[YEARS[0]]:
    r = {"measure": name}
    for y in YEARS:
        e = INEQ[y][name]
        r[str(y)] = f"{float(e.value):.3f} {fmt_ci(e, 3)}"
    for a, b in PAIRS:
        d_ = INEQ[b][name] - INEQ[a][name]
        r[f"change {a}->{b}"] = f"{float(d_.value):+.3f} {stars(float(d_.pvalue()))} (p={fmt_p(float(d_.pvalue()))})"
    rows.append(r)
display(pd.DataFrame(rows))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for y in YEARS:
    p, L = ss.lorenz_points(unit_y(y, u, U["total"]), W[y][u], rank_by=D[y]["Inc_per_OECDcap"])
    axes[0].plot(p, L, color=YEAR_COLORS[y], linewidth=2, label=str(y))
    means = []
    for k in range(1, 11):
        m = D[y][U["income_decile"]].astype(str) == str(k)
        means.append(np.average(unit_y(y, u, U["total"])[m], weights=W[y][u][m]))
    axes[1].plot(range(1, 11), means, "o-", color=YEAR_COLORS[y], label=str(y))
axes[0].plot([0, 1], [0, 1], color="#999", linewidth=1, linestyle="--")
axes[0].set_xlabel("Cumulative share of people, ranked by income (poorest first)")
axes[0].set_ylabel("Cumulative share of the footprint")
axes[0].set_title("Concentration curves")
axes[1].set_xlabel("Income decile (per OECD adult equivalent)")
axes[1].set_ylabel(f"Mean footprint ({UNIT} {U['label']})")
axes[1].set_title("Footprint by income decile")
axes[1].set_xticks(range(1, 11))
for ax in axes:
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

_ci = INEQ[LAST_YEAR]["Concentration index by income"] - INEQ[BASE_YEAR]["Concentration index by income"]
print(f"Reading: the concentration index by income moved by {_ci.value:+.3f} from {BASE_YEAR} to {LAST_YEAR} "
      f"({signif_words(float(_ci.pvalue()))}, p = {fmt_p(float(_ci.pvalue()))}). "
      + ("A rise means the footprint became more concentrated among higher-income people."
         if _ci.value > 0 else "A fall means the footprint became less tied to income."))

# %% [markdown]
# ## B. How did the categories change?
#
# ### B1. Footprint per category by year, and the ranking of changes
#
# **What it shows.**
# - The mean footprint in each of the 12 categories, by year.
# - The category's share of the total footprint.
# - The change from the base year in GJ and in %.
# - Its **contribution to the total change**: the category's change in GJ divided by the change in the total. Contributions add up to 100%. A negative contribution means the category moved against the total.
#
# **What the test does.** For each category, the change is tested with the bootstrap, as in A1. Because 12 categories are tested at once, the p-values are Holm-adjusted.
#
# **How to read it.** Ranking by % change shows which categories changed most *relative to their size*. Ranking by GJ (or contribution) shows which categories drove the change in the total. A small category can change a lot in % and still barely move the total.

# %%
B_est = {}
B_tables = {}
for u, U in UNITS.items():
    for y in YEARS:
        Ymat = np.column_stack([unit_y(y, u, U["prefix"] + c) for c in CATS] + [unit_y(y, u, U["total"])])
        B_est[(u, y)] = ss.wmean(Ymat, W[y][u], WB[y][u])     # last column = total
    rows = []
    a, b = MAIN_PAIR
    d_all = B_est[(u, b)] - B_est[(u, a)]
    pc_all = pct_change(B_est[(u, b)], B_est[(u, a)])
    pvals = [float(d_all[i].pvalue()) for i in range(len(CATS))]
    p_holm = holm(pvals)
    tot_change = float(d_all.value[-1])
    for i, c in enumerate(CATS):
        r = {"category": CAT_LABEL[c]}
        for y in YEARS:
            r[f"{y} ({UNIT})"] = float(B_est[(u, y)].value[i])
        for y in YEARS:
            r[f"share {y} (%)"] = 100 * float(B_est[(u, y)].value[i] / B_est[(u, y)].value[-1])
        r[f"change {a}->{b} ({UNIT})"] = float(d_all.value[i])
        r[f"change {a}->{b} (%)"] = float(pc_all.value[i])
        lo, hi = pc_all[i].ci()
        r["% 95% CI"] = f"[{lo:+.1f}, {hi:+.1f}]"
        r["contribution to total change (%)"] = 100 * float(d_all.value[i]) / tot_change if tot_change else np.nan
        r["p (Holm)"] = p_holm[i]
        r["sig."] = stars(p_holm[i])
        for pa, pb in PAIRS:
            if (pa, pb) != MAIN_PAIR:
                r[f"% {pa}->{pb}"] = float(pct_change(B_est[(u, pb)], B_est[(u, pa)]).value[i])
        rows.append(r)
    t = pd.DataFrame(rows).sort_values(f"change {a}->{b} (%)", ascending=False)
    B_tables[u] = t
    print(f"\nCategories, {U['label']} (sorted by % change {a}->{b})")
    display(t.round(2))

# Charts: mean per category by year, and the ranking of the change.
u = "per_cap"
U = UNITS[u]
t = B_tables[u]
a, b = MAIN_PAIR
fig, axes = plt.subplots(1, 3, figsize=(17, 6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
order = t.sort_values(f"{BASE_YEAR} ({UNIT})")["category"].tolist()
ypos = np.arange(len(order))
h = 0.8 / len(YEARS)
for k, y in enumerate(YEARS):
    vals = t.set_index("category").loc[order, f"{y} ({UNIT})"]
    axes[0].barh(ypos + (k - (len(YEARS) - 1) / 2) * h, vals, height=h, color=YEAR_COLORS[y], label=str(y))
axes[0].set_yticks(ypos, [short(o) for o in order], fontsize=8)
axes[0].set_xlabel(f"{UNIT} {U['label']}")
axes[0].set_title("Footprint per category")
axes[0].legend(frameon=False)
for ax, col, title in [(axes[1], f"change {a}->{b} (%)", f"% change {a}->{b}"),
                       (axes[2], f"change {a}->{b} ({UNIT})", f"Change in {UNIT} {a}->{b}")]:
    s = t.sort_values(col)
    colors = ["#2a78d6" if v < 0 else "#eb6834" for v in s[col]]
    ax.barh(range(len(s)), s[col], color=colors)
    ax.set_yticks(range(len(s)), [short(c_) for c_ in s["category"]], fontsize=8)
    lo_, hi_ = min(s[col].min(), 0), max(s[col].max(), 0)
    ax.set_xlim(lo_ - 0.35 * (hi_ - lo_), hi_ + 0.35 * (hi_ - lo_))   # room for the value labels
    for j, (v, st) in enumerate(zip(s[col], s["sig."])):
        ax.text(v, j, f" {v:+.1f}{st}" if v >= 0 else f"{v:+.1f}{st} ", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    ax.axvline(0, color="#999", linewidth=1)
    ax.set_title(title)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

_t = B_tables["per_cap"]
_col = f"change {a}->{b} (%)"
_top, _bot = _t.iloc[0], _t.iloc[-1]
_focus = _t[_t["category"] == CAT_LABEL[FOCUS_CATEGORY]].iloc[0]
_drv = _t.sort_values(f"change {a}->{b} ({UNIT})", key=np.abs, ascending=False).iloc[0]
print(f"Reading ({UNITS['per_cap']['label']}, {a}->{b}): the largest % increase is in {_top['category']} "
      f"({_top[_col]:+.1f}%), the largest % decrease (or smallest increase) in {_bot['category']} ({_bot[_col]:+.1f}%). "
      f"{_drv['category']} moved the total most ({_drv[f'change {a}->{b} ({UNIT})']:+.1f} {UNIT}, "
      f"{_drv['contribution to total change (%)']:.0f}% of the total change). "
      f"{CAT_LABEL[FOCUS_CATEGORY]} changed by {_focus[_col]:+.1f}% "
      f"({'significant' if _focus['p (Holm)'] < 0.05 else 'not significant'} after Holm adjustment, p = {fmt_p(_focus['p (Holm)'])}).")

# %% [markdown]
# ### B2. Did the typical household's footprint mix change?
#
# **What it shows.** For each household, the share of its own footprint that falls in each category, averaged over households, by year. This differs from the shares in B1, which are each category's share of the *national* total. Here each household counts equally (per person, weighted by OECD_cap), so a few very large footprints do not dominate.
#
# **What the test does.** It tests the change in each average share from the base year with the bootstrap, Holm-adjusted across categories.
#
# **How to read it.** A significant rise in, say, the housing share means housing now makes up a larger part of the *typical* household's footprint, whatever happened to the total.

# %%
u = "per_cap"
U = UNITS[u]
SH = {}
for y in YEARS:
    tot = unit_y(y, u, U["total"])
    shares = np.column_stack([unit_y(y, u, U["prefix"] + c) / tot for c in CATS]) * 100
    SH[y] = ss.wmean(shares, W[y][u], WB[y][u])
a, b = MAIN_PAIR
dsh = SH[b] - SH[a]
p_h = holm([float(dsh[i].pvalue()) for i in range(len(CATS))])
rows = []
for i, c in enumerate(CATS):
    r = {"category": CAT_LABEL[c]}
    for y in YEARS:
        r[f"mean share {y} (%)"] = round(float(SH[y].value[i]), 2)
    r[f"change {a}->{b} (pp)"] = round(float(dsh.value[i]), 2)
    r["95% CI"] = fmt_ci(dsh[i], 2)
    r["p (Holm)"] = fmt_p(p_h[i]) + " " + stars(p_h[i])
    rows.append(r)
display(pd.DataFrame(rows).sort_values(f"change {a}->{b} (pp)", ascending=False))
_i = CATS.index(FOCUS_CATEGORY)
print(f"Reading: {CAT_LABEL[FOCUS_CATEGORY]} made up {SH[a].value[_i]:.1f}% of the typical household footprint in {a} "
      f"and {SH[b].value[_i]:.1f}% in {b} ({dsh.value[_i]:+.2f} percentage points, "
      f"{'significant' if p_h[_i] < 0.05 else 'not significant'} after Holm adjustment).")

# %% [markdown]
# ### B3. Why did each category change: more spending, or more energy per dollar?  *(new calculation)*
#
# **What it shows.** Each category's footprint per person can be written as
#
# footprint per person = spending per person (CAD) x energy intensity (energy per CAD)
#
# The change between two years is split into a **spending effect** (people spent more or less in the category) and an **intensity effect** (each dollar carried more or less energy). The split uses the LMDI method, where the two effects add up exactly to the total change. Spending comes from the survey. Intensity is the model's category footprint divided by survey spending, so it reflects FIGARO/IEA and the bridge.
#
# **What the test does.** There is no significance test: both parts are national totals from the model, not survey estimates. It is an accounting split.
#
# **How to read it.** If the intensity effect is negative while the spending effect is positive, people spent more but each dollar carried less energy. Spending is in nominal CAD, so inflation counts as "more spending" and as "lower intensity" by the same amount. Set `CPI` in the Setup cell to turn spending into real spending and remove inflation from both effects. The national-total table adds a third effect: the growth in population.

# %%
if category_totals is None:
    print("Category totals file not found: rerun Household_EF (Cell 12) to create it.")
else:
    ct = category_totals.set_index(["year", "safe_name"])
    deflator = {y: (CPI[y] / CPI[BASE_YEAR] if CPI else 1.0) for y in YEARS}
    price_word = "real CAD" if CPI else "nominal CAD"
    B3 = {}
    for u, U in UNITS.items():
        rows = []
        for a, b in PAIRS:
            for c in CATS + ["TOTAL"]:
                def _get(y):
                    sub = ct.loc[y]
                    if c == "TOTAL":
                        E, S = sub["EF_TJ"].sum(), sub["spend_CAD"].sum()
                    else:
                        E, S = sub.loc[c, "EF_TJ"], sub.loc[c, "spend_CAD"]
                    P = sub[U["pop"]].iloc[0]
                    S = S / deflator[y]
                    return E * UNIT_SCALE, S, P
                E0, S0, P0 = _get(a)
                E1, S1, P1 = _get(b)
                # Per-unit decomposition: e = (S/P) * (E/S)
                eff = ss.lmdi(E0 / P0, E1 / P1,
                              {"spending": S0 / P0, "intensity": E0 / S0},
                              {"spending": S1 / P1, "intensity": E1 / S1})
                # National-total decomposition: E = P * (S/P) * (E/S)
                effN = ss.lmdi(E0, E1,
                               {"population": P0, "spending": S0 / P0, "intensity": E0 / S0},
                               {"population": P1, "spending": S1 / P1, "intensity": E1 / S1})
                rows.append({
                    "change": f"{a}->{b}", "category": "TOTAL" if c == "TOTAL" else CAT_LABEL[c],
                    f"footprint {a} ({UNIT})": E0 / P0, f"footprint {b} ({UNIT})": E1 / P1,
                    f"change ({UNIT})": E1 / P1 - E0 / P0,
                    "spending effect": float(eff["spending"]), "intensity effect": float(eff["intensity"]),
                    f"spending per unit change (%, {price_word})": 100 * ((S1 / P1) / (S0 / P0) - 1),
                    "intensity change (%)": 100 * ((E1 / S1) / (E0 / S0) - 1),
                    "national: population effect (TJ)": float(effN["population"]) / UNIT_SCALE,
                    "national: spending effect (TJ)": float(effN["spending"]) / UNIT_SCALE,
                    "national: intensity effect (TJ)": float(effN["intensity"]) / UNIT_SCALE,
                })
        B3[u] = pd.DataFrame(rows)
        print(f"\nSpending vs intensity, {U['label']} (effects in {UNIT} {U['label']}; spending in {price_word})")
        display(B3[u][B3[u]["change"] == f"{MAIN_PAIR[0]}->{MAIN_PAIR[1]}"].round(2))

    u = "per_cap"
    t3 = B3[u][(B3[u]["change"] == f"{MAIN_PAIR[0]}->{MAIN_PAIR[1]}") & (B3[u]["category"] != "TOTAL")]
    t3 = t3.sort_values(f"change ({UNIT})")
    fig, ax = plt.subplots(figsize=(10, 6))
    ypos = np.arange(len(t3))
    sp, it = t3["spending effect"].to_numpy(), t3["intensity effect"].to_numpy()
    ax.barh(ypos - 0.2, sp, height=0.4, color="#2a78d6", label=f"Spending effect ({price_word})")
    ax.barh(ypos + 0.2, it, height=0.4, color="#eb6834", label="Intensity effect")
    ax.scatter(t3[f"change ({UNIT})"], ypos, color="#111", zorder=3, s=30, label="Net change")
    ax.axvline(0, color="#999", linewidth=1)
    ax.set_yticks(ypos, [short(c_) for c_ in t3["category"]], fontsize=8)
    ax.set_xlabel(f"{UNIT} {UNITS[u]['label']}")
    ax.set_title(f"What drove each category's change, {MAIN_PAIR[0]}->{MAIN_PAIR[1]}")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()

    _f = t3[t3["category"] == CAT_LABEL[FOCUS_CATEGORY]].iloc[0]
    print(f"Reading: {CAT_LABEL[FOCUS_CATEGORY]} changed by {_f[f'change ({UNIT})']:+.2f} {UNIT} per person. "
          f"Spending per person changed by {_f[f'spending per unit change (%, {price_word})']:+.1f}% "
          f"(effect {_f['spending effect']:+.2f} {UNIT}) and energy per CAD by {_f['intensity change (%)']:+.1f}% "
          f"(effect {_f['intensity effect']:+.2f} {UNIT}).")

# %% [markdown]
# ## C. Which socioeconomic characteristics does the change depend on?
#
# ### C1. Change by group
#
# **What it shows.** For each characteristic (income decile, household type, age, tenure, and so on), the mean footprint per person in each group and year. The table adds the % change from the base year with its p-value. The table and charts below cover the total footprint; the next cell repeats the table for `FOCUS_CATEGORY`.
#
# **What the test does.** It is the same bootstrap test as in A1, run within each group.
#
# **How to read it.** If one group's change is clearly larger than another's, the change depends on that characteristic. These are raw comparisons, though: renters are also younger and poorer, for example. C2 and C3 separate the characteristics from each other.

# %%
def drivers_for(u):
    """Characteristics used in section C; the income decile matches the unit."""
    return [UNITS[u]["income_decile"], "HHType6", "RP_AgeGrp", "RP_Gender", "RP_Educ",
            "DwelTyp", "Tenure", "Prov", "VehicleYN", "SecResYN", "HH_MajIncSrc"]


def driver_name(v):
    return DRIVER_NAMES["INC_DEC"] if v in ("Inc_OECDcap_dec", "Inc_hh_dec") else DRIVER_NAMES.get(v, v)


def group_means(u, col, v):
    """Est of group means (one per level) for every year, plus the levels."""
    levels = sorted_levels(pd.concat([D[y][v] for y in YEARS]).astype(str))
    out = {}
    for y in YEARS:
        g = D[y][v].astype(str).to_numpy()
        yv = unit_y(y, u, col)
        ind = np.column_stack([g == lev for lev in levels]).astype(float)
        m = ss.wmean(np.column_stack([ind * yv[:, None], ind]), W[y][u], WB[y][u])
        k = len(levels)
        out[y] = Est(m.value[:k] / m.value[k:], m.reps[:, :k] / m.reps[:, k:])
    return levels, out


def group_change_table(u, col):
    a, b = MAIN_PAIR
    rows = []
    for v in drivers_for(u):
        levels, gm = group_means(u, col, v)
        pc = pct_change(gm[b], gm[a])
        dd = gm[b] - gm[a]
        for i, lev in enumerate(levels):
            r = {"characteristic": driver_name(v), "group": level_name(v, lev)}
            for y in YEARS:
                r[f"{y} ({UNIT})"] = round(float(gm[y].value[i]), 2)
            r[f"% change {a}->{b}"] = round(float(pc.value[i]), 1)
            r["95% CI"] = fmt_ci(pc[i])
            r["p"] = fmt_p(float(dd[i].pvalue())) + " " + stars(float(dd[i].pvalue()))
            rows.append(r)
    return pd.DataFrame(rows)


u = "per_cap"
C1_total = group_change_table(u, UNITS[u]["total"])
print(f"Total footprint by group ({UNITS[u]['label']})")
display(C1_total)

# Small multiples: one panel per characteristic, bars per year.
drv = drivers_for(u)
ncol = 4
nrow = int(np.ceil(len(drv) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow))
for ax, v in zip(axes.flat, drv):
    levels, gm = group_means(u, UNITS[u]["total"], v)
    x = np.arange(len(levels))
    h = 0.8 / len(YEARS)
    for k, y in enumerate(YEARS):
        ax.bar(x + (k - (len(YEARS) - 1) / 2) * h, gm[y].value, width=h, color=YEAR_COLORS[y], label=str(y))
    ax.set_xticks(x, [level_name(v, l) for l in levels], rotation=45, ha="right", fontsize=7)
    ax.set_title(driver_name(v), fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
for ax in list(axes.flat)[len(drv):]:
    ax.axis("off")
axes.flat[0].legend(frameon=False, fontsize=8)
fig.suptitle(f"Mean footprint ({UNIT} {UNITS[u]['label']}) by group and year")
plt.tight_layout()
plt.show()

# %%
C1_focus = group_change_table("per_cap", UNITS["per_cap"]["prefix"] + FOCUS_CATEGORY)
print(f"{CAT_LABEL[FOCUS_CATEGORY]} footprint by group ({UNITS['per_cap']['label']})")
display(C1_focus)

# %% [markdown]
# ### C2. Regression for each year: effect of each characteristic, holding the others fixed
#
# **What it shows.** For each year, a weighted regression of the footprint on all the characteristics at once. Each coefficient is the difference in footprint (GJ) between a group and the *reference group* of that characteristic, holding all other characteristics fixed. The reference groups are: income decile 1, one-person household, age under 30, male, less than high school, single detached, owner with mortgage, Ontario, owns a vehicle, has a second residence, and employment income. The last column shows how much each coefficient changed from the base year, with its p-value.
#
# **What the test does.** The bootstrap gives the uncertainty of each coefficient and of each change between years. Estimating each year separately is the same as one pooled model with every characteristic interacted with year (year x characteristic). The change column is that interaction term.
#
# **How to read it.** Suppose the income-decile-10 coefficient is +40 GJ in 2017 and +25 GJ in 2021, and the change is significant. Then the gap between the richest and the poorest decile shrank, with the other characteristics held fixed. In other words, the footprint change depended on income.
#
# The same regressions are run for each of the 12 categories and for the log of the total (used in D). Only the total is printed here.

# %%
# Rows used in the regressions (C2-C5). Education is not published for the
# Territories in 2017 and 2019 (all their households are "not stated"), so
# Territories and "education not stated" cannot be told apart. Both are left
# out of the regressions; the descriptive results (A, B, C1) keep everyone.
REG_ROWS = {y: ((D[y]["Prov"].astype(str) != "63") & (D[y]["RP_Educ"].astype(str) != "9")).to_numpy()
            for y in YEARS}
for y in YEARS:
    _m = REG_ROWS[y]
    print(f"{y}: regressions use {_m.sum()} of {len(_m)} households "
          f"({100 * W[y]['per_cap'][_m].sum() / W[y]['per_cap'].sum():.1f}% of people)")


def build_designs(u):
    drv = drivers_for(u)
    DR = {y: D[y][REG_ROWS[y]] for y in YEARS}
    levels = {v: sorted_levels(pd.concat([DR[y][v] for y in YEARS]).astype(str)) for v in drv}
    reference = {"Prov": "35"}
    # A level missing in any year cannot be estimated for that year: merge it into the reference.
    for v in drv:
        for y in YEARS:
            present = set(DR[y][v].astype(str))
            missing = [lev for lev in levels[v] if lev not in present]
            if missing:
                print(f"Note: {v} levels {missing} missing in {y}; merged into the reference group.")
                levels[v] = [lev for lev in levels[v] if lev not in missing]
    X, groups = {}, None
    for y in YEARS:
        X[y], groups = ss.design_matrix(DR[y], drv, levels, reference)
    return X, groups, levels, reference


REG = {}
_t0 = time.time()
for u, U in UNITS.items():
    X, groups, levels, reference = build_designs(u)
    ycols = [U["prefix"] + c for c in CATS] + [U["total"]]
    names = CATS + ["TOTAL", "LOG_TOTAL"]
    beta, Ys, ws, Wbs = {}, {}, {}, {}
    for y in YEARS:
        m = REG_ROWS[y]
        Y = np.column_stack([unit_y(y, u, c)[m] for c in ycols])
        tot = Y[:, -1]
        Ys[y] = np.column_stack([Y, np.log(np.maximum(tot, tot[tot > 0].min()))])
        ws[y], Wbs[y] = W[y][u][m], WB[y][u][m]
        beta[y] = ss.wls_boot(X[y].to_numpy(), Ys[y], ws[y], Wbs[y])
    REG[u] = {"X": X, "groups": groups, "levels": levels, "beta": beta, "names": names,
              "cols": list(X[YEARS[0]].columns), "Y": Ys, "w": ws, "Wb": Wbs}
print(f"Regressions with {N_BOOT} bootstrap weights done in {time.time() - _t0:.0f} s")


def coef_table(u, outcome="TOTAL"):
    R = REG[u]
    k = R["names"].index(outcome)
    a, b = MAIN_PAIR
    rows = []
    for j, col in enumerate(R["cols"]):
        v, lev = (col.split("[")[0], col.split("[")[1].rstrip("]")) if "[" in col else ("Intercept", "")
        r = {"characteristic": driver_name(v) if v != "Intercept" else "Intercept (reference household)",
             "group": level_name(v, lev) if lev else ""}
        for y in YEARS:
            e = R["beta"][y][j, k]
            r[str(y)] = f"{float(e.value):+.2f}{stars(float(e.pvalue()))}"
        d_ = R["beta"][b][j, k] - R["beta"][a][j, k]
        r[f"change {a}->{b}"] = f"{float(d_.value):+.2f}{stars(float(d_.pvalue()))} (p={fmt_p(float(d_.pvalue()))})"
        rows.append(r)
    return pd.DataFrame(rows)


print(f"Coefficients for the total footprint ({UNIT} {UNITS['per_cap']['label']})")
display(coef_table("per_cap"))

# %% [markdown]
# ### C3. Which characteristics does the change depend on most? (joint tests)
#
# **What it shows.** For each characteristic, one joint test of whether *all* its coefficients stayed the same across the years. For example, for income: did the gaps between all ten deciles stay the same? A significant result means that group's footprint changed differently from the other groups', even after accounting for the other characteristics. The table ranks characteristics by the test statistic per degree of freedom (chi2/df), so the top rows are the characteristics whose effect changed most clearly.
#
# **What the test does.** It is a Wald test using the bootstrap covariance of the coefficient changes, Holm-adjusted across the 11 characteristics. The heatmap repeats the test for each category. Each cell is one characteristic in one category: darker means stronger evidence of a change, and stars mark Holm-adjusted significance within that category.
#
# **How to read it.** Characteristics at the top of the table, or with dark cells in the heatmap, are the ones the change in footprint depends on. "Intercept" is the change for the reference household. It captures the part of the change that is common to everyone.

# %%
def interaction_tests(u, outcome):
    R = REG[u]
    k = R["names"].index(outcome)
    rows = []
    later = [y for y in YEARS if y != BASE_YEAR]
    tests = list(R["groups"].items()) + [("Intercept", ["Intercept"])]
    for v, cols in tests:
        idx = [R["cols"].index(c) for c in cols]
        diffs = [R["beta"][y][idx, k] - R["beta"][BASE_YEAR][idx, k] for y in later]
        stacked = Est(np.concatenate([d.value for d in diffs]),
                      np.concatenate([d.reps for d in diffs], axis=1))
        chi2, dof, p = ss.wald_test(stacked)
        rows.append({"characteristic": driver_name(v) if v != "Intercept" else "Intercept (common change)",
                     "chi2": chi2, "df": dof, "chi2/df": chi2 / dof, "p": p})
    t = pd.DataFrame(rows)
    m = t["characteristic"] != "Intercept (common change)"
    t.loc[m, "p (Holm)"] = holm(t.loc[m, "p"])
    t["sig."] = [stars(p) for p in t["p (Holm)"].fillna(t["p"])]
    return t.sort_values("chi2/df", ascending=False).reset_index(drop=True)


C3 = {}
for u, U in UNITS.items():
    C3[(u, "TOTAL")] = interaction_tests(u, "TOTAL")
    print(f"\nDoes the effect of each characteristic change across {YEARS}? Total footprint, {U['label']}")
    display(C3[(u, "TOTAL")].round(4))

u = "per_cap"
heat = {}
for c in CATS + ["TOTAL"]:
    t = interaction_tests(u, c).set_index("characteristic")
    t = t.drop("Intercept (common change)")
    heat["Total" if c == "TOTAL" else CAT_LABEL[c]] = t
drv_order = C3[(u, "TOTAL")]["characteristic"].tolist()
drv_order = [d for d in drv_order if d != "Intercept (common change)"]
H = pd.DataFrame({cat: -np.log10(t.loc[drv_order, "p (Holm)"].clip(lower=1e-12)) for cat, t in heat.items()}).T
S = pd.DataFrame({cat: [stars(p) for p in t.loc[drv_order, "p (Holm)"]] for cat, t in heat.items()}).T
fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(H.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=max(4, np.nanmax(H.to_numpy())))
ax.set_xticks(range(len(drv_order)), drv_order, rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(len(H.index)), H.index, fontsize=8)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax.text(j, i, S.iloc[i, j], ha="center", va="center", fontsize=9,
                color="white" if H.iloc[i, j] > 3 else "#111")
fig.colorbar(im, ax=ax, label="-log10(Holm-adjusted p)   (1.3 = p 0.05, 3 = p 0.001)")
ax.set_title(f"Did the effect of each characteristic change across {YEARS}? ({UNITS[u]['label']})")
plt.tight_layout()
plt.show()

_t = C3[("per_cap", "TOTAL")]
_sig = _t[(_t["characteristic"] != "Intercept (common change)") & (_t["p (Holm)"] < 0.05)]
print("Reading (total footprint per OECD adult equivalent): "
      + (f"the effect changed significantly across years for {len(_sig)} characteristic(s): "
         + ", ".join(_sig["characteristic"]) + f". The clearest is {_sig.iloc[0]['characteristic']}."
         if len(_sig) else "no characteristic's effect changed significantly after Holm adjustment, "
                           "so the change was spread fairly evenly across groups."))

# %% [markdown]
# ### C4. Composition vs behaviour: Oaxaca-Blinder decomposition  *(new calculation)*
#
# **What it shows.** The change in the mean footprint between the base year and the last year, split into two parts.
# - **Composition effect:** the population's characteristics changed. For example, more renters, an older population or fewer vehicle owners. This is the change you would get if each group kept its base-year footprint and only the group sizes changed.
# - **Coefficient (behaviour) effect:** households with the *same* characteristics have a different footprint than before. This part also includes changes in the model's energy intensities.
#
# Each effect is broken down by characteristic. The intercept row is the common change for the reference household; it belongs to the coefficient effect.
#
# **What the test does.** The bootstrap gives a confidence interval and p-value for every part.
#
# **How to read it.** A large composition effect for, say, age means part of the change happened because the population got older, not because anyone changed behaviour. A note of caution: the split of the *coefficient* effect between characteristics depends on which group is the reference, so read it as indicative. The composition effects and the two totals do not have this problem.

# %%
def oaxaca(u, outcome, pair=MAIN_PAIR):
    a, b = pair
    R = REG[u]
    k = R["names"].index(outcome)
    xbar = {y: ss.wmean(R["X"][y].to_numpy(), R["w"][y], R["Wb"][y]) for y in (a, b)}
    ba, bb = R["beta"][a][:, k], R["beta"][b][:, k]
    comp = (xbar[b] - xbar[a]) * ba          # per column
    coef = xbar[b] * (bb - ba)
    rows = []
    groups = list(R["groups"].items())
    for v, cols in groups:
        idx = [R["cols"].index(c) for c in cols]
        ce = Est(comp.value[idx].sum(), comp.reps[:, idx].sum(axis=1))
        be = Est(coef.value[idx].sum(), coef.reps[:, idx].sum(axis=1))
        rows.append({"characteristic": driver_name(v),
                     f"composition ({UNIT})": float(ce.value), "comp. 95% CI": fmt_ci(ce, 2),
                     "comp. p": float(ce.pvalue()),
                     f"coefficient ({UNIT})": float(be.value), "coef. 95% CI": fmt_ci(be, 2),
                     "coef. p": float(be.pvalue())})
    i0 = R["cols"].index("Intercept")
    ie = coef[i0]
    rows.append({"characteristic": "Intercept (common change)", f"composition ({UNIT})": 0.0,
                 f"coefficient ({UNIT})": float(ie.value), "coef. 95% CI": fmt_ci(ie, 2),
                 "coef. p": float(ie.pvalue())})
    comp_tot = Est(comp.value.sum(), comp.reps.sum(axis=1))
    coef_tot = Est(coef.value.sum(), coef.reps.sum(axis=1))
    total = comp_tot + coef_tot
    summary = {"change": total, "composition": comp_tot, "coefficient": coef_tot}
    return pd.DataFrame(rows), summary


u = "per_cap"
oax_t, oax_s = oaxaca(u, "TOTAL")
print(f"Oaxaca-Blinder, total footprint {UNITS[u]['label']}, {MAIN_PAIR[0]}->{MAIN_PAIR[1]}")
for kname, e in oax_s.items():
    print(f"  {kname:12s} {float(e.value):+8.2f} {UNIT}  95% CI {fmt_ci(e, 2)}  p={fmt_p(float(e.pvalue()))}")
display(oax_t.round(3))

fig, ax = plt.subplots(figsize=(10, 5))
tt = oax_t.iloc[::-1]
ypos = np.arange(len(tt))
ax.barh(ypos - 0.2, tt[f"composition ({UNIT})"], height=0.4, color="#2a78d6", label="Composition")
ax.barh(ypos + 0.2, tt[f"coefficient ({UNIT})"], height=0.4, color="#eb6834", label="Coefficient (behaviour)")
ax.axvline(0, color="#999", linewidth=1)
ax.set_yticks(ypos, tt["characteristic"], fontsize=8)
ax.set_xlabel(f"{UNIT} {UNITS[u]['label']}")
ax.set_title(f"Oaxaca-Blinder decomposition of the change {MAIN_PAIR[0]}->{MAIN_PAIR[1]}")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

# Composition vs coefficient for every category and both units.
rows = []
for uu, UU in UNITS.items():
    for c in CATS + ["TOTAL"]:
        _, s = oaxaca(uu, c)
        rows.append({"unit": UU["label"], "category": "Total" if c == "TOTAL" else CAT_LABEL[c],
                     f"change ({UNIT})": float(s["change"].value),
                     f"composition ({UNIT})": float(s["composition"].value),
                     "comp. p": fmt_p(float(s["composition"].pvalue())),
                     f"coefficient ({UNIT})": float(s["coefficient"].value),
                     "coef. p": fmt_p(float(s["coefficient"].pvalue()))})
display(pd.DataFrame(rows).round(2))

_c, _b = oax_s["composition"], oax_s["coefficient"]
_big = oax_t.iloc[:-1].reindex(oax_t.iloc[:-1][f"composition ({UNIT})"].abs().sort_values(ascending=False).index).iloc[0]
print(f"Reading: of the {float(oax_s['change'].value):+.2f} {UNIT} change per person (regression sample), {float(_c.value):+.2f} {UNIT} "
      f"comes from changes in who the households are (composition, p = {fmt_p(float(_c.pvalue()))}) and "
      f"{float(_b.value):+.2f} {UNIT} from households with the same characteristics having a different footprint "
      f"(p = {fmt_p(float(_b.pvalue()))}). The largest composition effect is from {_big['characteristic']} "
      f"({_big[f'composition ({UNIT})']:+.2f} {UNIT}).")

# %% [markdown]
# ### C5. Random forest: does the importance of the characteristics shift between years?
#
# **What it shows.** A random forest is a flexible prediction model. It can pick up non-linear effects and interactions that the regression misses. It is fitted separately for each year, and each characteristic gets an *importance*: how much it helps predict the footprint. Dummies are summed back to their characteristic. The cross-validated R² is the share of the variation the model predicts on households it did not see.
#
# **What the test does.** None. This is descriptive and gives no p-values. Use it to check that the C2–C3 picture holds when the model is flexible.
#
# **How to read it.** A characteristic whose importance rises between years matters more for explaining footprints in the later year. If the random forest R² is much higher than the regression R², there are non-linear effects or interactions the regression misses.

# %%
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    u = "per_cap"
    R = REG[u]
    imp, r2 = {}, {}
    for y in YEARS:
        Xy = R["X"][y].drop(columns="Intercept")
        k = R["names"].index("TOTAL")
        yy, wy = R["Y"][y][:, k], R["w"][y]
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=42)
        r2[y] = cross_val_score(rf, Xy, yy, cv=5, scoring="r2").mean()
        rf.fit(Xy, yy, sample_weight=wy)
        s = pd.Series(rf.feature_importances_, index=Xy.columns)
        imp[y] = s.groupby(lambda c: driver_name(c.split("[")[0])).sum()
        # Weighted R² of the linear regression, for comparison.
        fit = R["X"][y].to_numpy() @ R["beta"][y].value[:, k]
        ybar = np.average(yy, weights=wy)
        r2_lin = 1 - np.sum(wy * (yy - fit) ** 2) / np.sum(wy * (yy - ybar) ** 2)
        print(f"{y}: random forest cross-validated R² = {r2[y]:.3f}; regression R² (in-sample) = {r2_lin:.3f}")
    IMP = pd.DataFrame(imp).sort_values(LAST_YEAR, ascending=False)
    display(IMP.round(3))

    fig, ax = plt.subplots(figsize=(10, 5))
    ypos = np.arange(len(IMP))[::-1]
    h = 0.8 / len(YEARS)
    for k, y in enumerate(YEARS):
        ax.barh(ypos + ((len(YEARS) - 1) / 2 - k) * h, IMP[y], height=h, color=YEAR_COLORS[y], label=str(y))
    ax.set_yticks(ypos, IMP.index, fontsize=8)
    ax.set_xlabel("Importance (share of the forest's predictive power)")
    ax.set_title("Random forest importance by year (total footprint per OECD adult equivalent)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()
    _d = (IMP[LAST_YEAR] - IMP[BASE_YEAR]).sort_values()
    print(f"Reading: from {BASE_YEAR} to {LAST_YEAR} the importance of {_d.index[-1]} rose most "
          f"({_d.iloc[-1]:+.3f}) and that of {_d.index[0]} fell most ({_d.iloc[0]:+.3f}).")
except ImportError:
    print("scikit-learn not installed. Run: pip install scikit-learn")

# %% [markdown]
# ## D. Robustness checks
#
# ### D1. Log footprint
#
# **What it shows.** The same year comparison and the same joint tests, but on the log of the footprint. A change in mean log footprint converts to a % change of the *geometric* mean, which is much less sensitive to the few very large footprints than the ordinary mean in A1.
#
# **How to read it.** If the % change and the ranking of characteristics look similar to A1 and C3, the results are not driven by a handful of extreme households.

# %%
u = "per_cap"
U = UNITS[u]
G = {}
for y in YEARS:
    yv = unit_y(y, u, U["total"])
    G[y] = ss.wmean(np.log(np.maximum(yv, yv[yv > 0].min())), W[y][u], WB[y][u])
rows = []
for a, b in PAIRS:
    d_ = G[b] - G[a]
    g = Est(100 * (np.exp(d_.value) - 1), 100 * (np.exp(d_.reps) - 1))
    lo, hi = g.ci()
    pm = A_est[(u, a, b)][1]
    rows.append({"change": f"{a}->{b}", "geometric mean change (%)": round(float(g.value), 2),
                 "95% CI": f"[{lo:+.2f}, {hi:+.2f}]", "p": fmt_p(float(d_.pvalue())),
                 "arithmetic mean change (%) from A1": round(float(pm.value), 2)})
display(pd.DataFrame(rows))
print("Joint tests on the log footprint (compare with C3):")
display(interaction_tests(u, "LOG_TOTAL").round(4))

# %% [markdown]
# ### D2. Bootstrap weights vs HC1 standard errors
#
# **What it shows.** The standard error (SE) of the main results computed two ways.
# - **HC1** treats the households as a simple random sample.
# - **Bootstrap** respects how Statistics Canada drew the sample (by province, in clusters).
#
# The ratio is the bootstrap SE divided by the HC1 SE.
#
# **How to read it.** A ratio above 1 means HC1 understates the uncertainty, so HC1 would call some results significant that are not. That is why the notebook uses the bootstrap throughout.

# %%
import statsmodels.formula.api as smf

u = "per_cap"
U = UNITS[u]
pool = pd.concat([D[y].assign(year=str(y)) for y in YEARS], ignore_index=True)
pool["y"] = pool[U["total"]] * UNIT_SCALE
fit = smf.wls(f"y ~ C(year, Treatment('{BASE_YEAR}'))", data=pool, weights=pool[U["weight"]]).fit(cov_type="HC1")
rows = []
for y in YEARS:
    if y == BASE_YEAR:
        continue
    name = f"C(year, Treatment('{BASE_YEAR}'))[T.{y}]"
    boot = A_est[(u, y, "mean")] - A_est[(u, BASE_YEAR, "mean")]
    rows.append({"estimate": f"change in mean {BASE_YEAR}->{y}", "value": round(float(boot.value), 3),
                 "SE HC1": round(float(fit.bse[name]), 3), "SE bootstrap": round(float(boot.se), 3),
                 "ratio": round(float(boot.se / fit.bse[name]), 2)})
R = REG[u]
k = R["names"].index("TOTAL")
for y in YEARS:
    Xy = R["X"][y].to_numpy()
    yy = R["Y"][y][:, k]
    se_hc1 = np.sqrt(np.diag(ss.hc1_cov(Xy, yy, R["w"][y], R["beta"][y].value[:, k])))
    se_boot = R["beta"][y][:, k].se
    ratio = se_boot[1:] / se_hc1[1:]
    rows.append({"estimate": f"{y} regression coefficients (C2), median over coefficients",
                 "SE HC1": round(float(np.median(se_hc1[1:])), 3),
                 "SE bootstrap": round(float(np.median(se_boot[1:])), 3),
                 "ratio": round(float(np.median(ratio)), 2)})
display(pd.DataFrame(rows).fillna(""))

# %% [markdown]
# ### D3. Notes on comparability
#
# - **Provinces.** 2017 only has "Atlantic" for NL, PE, NS and NB, so they are pooled in every year.
# - **Income.** Income enters as deciles *within* each year, which already removes inflation. Nominal income is only used to rank households for the concentration index (A3), and the ranking is unaffected by inflation.
# - **Main income source.** Code 5 ("all sources <= 0") has 2 households, in 2021 only. It is merged into "Other / none".
# - **Territories and "education not stated".** Statistics Canada does not publish education for the Territories in 2017 and 2019; all their households are coded "not stated". The two cannot be separated in a regression, so the regressions (C2–C5) leave out Territories households and households with education not stated. The cell above C2 prints how many households that is (Territories are about 0.3% of the population). The descriptive results (A, B, C1) include everyone.
# - **Number of bedrooms** is left out of the drivers. "Not stated" exists in 2017/2019 but not in 2021, and the number of bedrooms is largely a consequence of household type and income.
# - **Urban/rural** is not in the SHS public-use files, so it cannot be tested.
