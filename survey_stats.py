"""
Survey statistics helpers for Statistical_analysis_all_years.

Every estimator takes the full-sample weights plus an (n x B) matrix of
Statistics Canada bootstrap weights, and returns an `Est`: the full-sample
value together with its B bootstrap replicates. Standard errors follow the
SHS bootstrap method:

    SE = sqrt( mean_b (theta_b - theta)^2 )

where theta is the full-sample estimate and theta_b the estimate computed with
bootstrap weight b. Survey years are independent samples, so the difference
between two years' estimates is computed replicate by replicate (replicate b
of 2021 minus replicate b of 2017); its variance is then the sum of the two
years' variances, as it should be.
"""
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Estimate with bootstrap replicates
# ---------------------------------------------------------------------------
class Est:
    """A point estimate (any shape) with its bootstrap replicates (B, *shape)."""

    def __init__(self, value, reps):
        self.value = np.asarray(value, dtype=float)
        self.reps = np.asarray(reps, dtype=float)

    # Arithmetic works replicate by replicate, so differences, ratios and
    # % changes carry their own bootstrap distribution.
    def _op(self, other, f):
        if isinstance(other, Est):
            return Est(f(self.value, other.value), f(self.reps, other.reps))
        return Est(f(self.value, other), f(self.reps, other))

    def __add__(self, o): return self._op(o, np.add)
    def __sub__(self, o): return self._op(o, np.subtract)
    def __mul__(self, o): return self._op(o, np.multiply)
    def __truediv__(self, o): return self._op(o, np.divide)
    __radd__ = __add__
    __rmul__ = __mul__

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, tuple) else (idx,)
        return Est(self.value[idx], self.reps[(slice(None),) + idx])

    @property
    def se(self):
        return np.sqrt(np.mean((self.reps - self.value) ** 2, axis=0))

    def ci(self, level=0.95):
        z = stats.norm.ppf(0.5 + level / 2)
        return self.value - z * self.se, self.value + z * self.se

    def pvalue(self, null=0.0):
        """Two-sided p-value of value == null (normal approximation)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (self.value - null) / self.se
        return 2 * stats.norm.sf(np.abs(z))


def pct_change(new, base):
    """% change new vs base, as an Est."""
    return (new - base) / base * 100.0


# ---------------------------------------------------------------------------
# Weighted descriptive statistics, vectorised over the replicate weights
# ---------------------------------------------------------------------------
def _with_full(w, Wb):
    """Stack full-sample weights as column 0 in front of the replicates."""
    return np.column_stack([np.asarray(w, float), np.asarray(Wb, float)])


def wmean(Y, w, Wb):
    """
    Weighted means of the columns of Y (n x k, or n,) -> Est of shape (k,) or ().
    """
    Y = np.asarray(Y, float)
    one = Y.ndim == 1
    Y = Y[:, None] if one else Y
    W = _with_full(w, Wb)
    M = (W.T @ Y) / W.sum(axis=0)[:, None]          # (B+1, k)
    if one:
        M = M[:, 0]
    return Est(M[0], M[1:])


def wtotal(Y, w, Wb):
    """Weighted totals of the columns of Y -> Est."""
    Y = np.asarray(Y, float)
    W = _with_full(w, Wb)
    T = W.T @ Y
    return Est(T[0], T[1:])


def wquantiles(y, w, Wb, qs):
    """Weighted quantiles (lowest y whose cumulative weight share >= q) -> Est (len(qs),)."""
    y = np.asarray(y, float)
    order = np.argsort(y, kind="mergesort")
    ys = y[order]
    W = _with_full(w, Wb)[order]
    cw = np.cumsum(W, axis=0) / W.sum(axis=0)
    out = np.empty((W.shape[1], len(qs)))
    for j, q in enumerate(qs):
        idx = np.minimum((cw < q).sum(axis=0), len(ys) - 1)
        out[:, j] = ys[idx]
    return Est(out[0], out[1:])


def _gini_sorted(x_sorted, W_sorted):
    """Gini (or concentration index) of x for rows already sorted by the ranking variable."""
    p = np.cumsum(W_sorted, axis=0) / W_sorted.sum(axis=0)
    xw = W_sorted * x_sorted[:, None]
    L = np.cumsum(xw, axis=0) / xw.sum(axis=0)
    p_prev = np.vstack([np.zeros((1, W_sorted.shape[1])), p[:-1]])
    L_prev = np.vstack([np.zeros((1, W_sorted.shape[1])), L[:-1]])
    return 1.0 - np.sum((p - p_prev) * (L + L_prev), axis=0)


def gini(y, w, Wb):
    """Weighted Gini coefficient of y -> Est."""
    y = np.asarray(y, float)
    order = np.argsort(y, kind="mergesort")
    G = _gini_sorted(y[order], _with_full(w, Wb)[order])
    return Est(G[0], G[1:])


def concentration_index(y, rank_by, w, Wb):
    """
    Weighted concentration index of y with households ranked by `rank_by`
    (e.g. income). +1: all of y held by the richest; 0: y spread evenly
    across the income ranking; negative: y concentrated among the poorest.
    """
    y = np.asarray(y, float)
    order = np.argsort(np.asarray(rank_by, float), kind="mergesort")
    C = _gini_sorted(y[order], _with_full(w, Wb)[order])
    return Est(C[0], C[1:])


def lorenz_points(y, w, rank_by=None):
    """(population share, cumulative share of y) for plotting a Lorenz / concentration curve."""
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    order = np.argsort(y if rank_by is None else np.asarray(rank_by, float), kind="mergesort")
    p = np.concatenate([[0], np.cumsum(w[order]) / w.sum()])
    L = np.concatenate([[0], np.cumsum(w[order] * y[order]) / (w * y).sum()])
    return p, L


def weighted_ks(y1, w1, y2, w2):
    """Largest gap between two weighted cumulative distributions (Kolmogorov-Smirnov D)."""
    grid = np.unique(np.concatenate([y1, y2]))

    def cdf(y, w):
        o = np.argsort(y)
        cw = np.cumsum(w[o]) / w.sum()
        idx = np.searchsorted(y[o], grid, side="right")   # rows with y <= grid value
        return np.where(idx > 0, cw[np.maximum(idx - 1, 0)], 0.0)

    return float(np.max(np.abs(cdf(np.asarray(y1), np.asarray(w1)) - cdf(np.asarray(y2), np.asarray(w2)))))


# ---------------------------------------------------------------------------
# Weighted least squares with bootstrap replicates
# ---------------------------------------------------------------------------
def design_matrix(df, drivers, levels, reference):
    """
    Dummy-coded design matrix with an intercept and the same columns for every
    year. `levels[v]` lists the categories of driver v (first = reference unless
    `reference[v]` says otherwise). Returns (X DataFrame, {driver: [column names]}).
    """
    cols = {"Intercept": np.ones(len(df))}
    groups = {}
    for v in drivers:
        ref = reference.get(v, levels[v][0])
        groups[v] = []
        s = df[v].astype(str).values
        for lev in levels[v]:
            if lev == ref:
                continue
            name = f"{v}[{lev}]"
            cols[name] = (s == lev).astype(float)
            groups[v].append(name)
    return pd.DataFrame(cols, index=df.index), groups


def wls_boot(X, Y, w, Wb):
    """
    WLS coefficients for every column of Y, with the full-sample weights w and
    each bootstrap weight column of Wb. Returns Est of shape (p, k).
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    W = _with_full(w, Wb)
    B1 = W.shape[1]
    out = np.empty((B1, X.shape[1], Y.shape[1]))
    for b in range(B1):
        Xw = X * W[:, b][:, None]
        out[b] = np.linalg.solve(X.T @ Xw, Xw.T @ Y)
    return Est(out[0], out[1:])


def hc1_cov(X, y, w, beta):
    """HC1 (heteroskedasticity-robust) covariance of WLS coefficients, ignoring the survey design."""
    X = np.asarray(X, float)
    w = np.asarray(w, float)
    resid = np.asarray(y, float) - X @ beta
    Xw = X * w[:, None]
    bread = np.linalg.inv(X.T @ Xw)
    meat = (Xw * resid[:, None] ** 2).T @ Xw
    n, p = X.shape
    return n / (n - p) * bread @ meat @ bread


def wald_test(est):
    """
    Joint test that every element of `est` (a 1-D Est) is zero, using the
    bootstrap covariance matrix. Returns (chi2, df, p).
    """
    d = np.atleast_1d(est.value)
    R = est.reps.reshape(est.reps.shape[0], -1) - d
    V = R.T @ R / R.shape[0]
    chi2 = float(d @ np.linalg.pinv(V) @ d)
    df = len(d)
    return chi2, df, float(stats.chi2.sf(chi2, df))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (controls the chance of any false positive)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


# ---------------------------------------------------------------------------
# Decompositions
# ---------------------------------------------------------------------------
def log_mean(a, b):
    """Logarithmic mean L(a, b) used by the LMDI decomposition."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = (a - b) / (np.log(a) - np.log(b))
    return np.where(np.isclose(a, b), a, L)


def lmdi(v0, vT, factors0, factorsT):
    """
    Additive LMDI-I: split vT - v0 (v = product of the factors) into one
    effect per factor. factors0/factorsT: dict name -> value. Effects add up
    exactly to the change.
    """
    L = log_mean(vT, v0)
    return {k: L * np.log(np.asarray(factorsT[k], float) / np.asarray(factors0[k], float))
            for k in factors0}
