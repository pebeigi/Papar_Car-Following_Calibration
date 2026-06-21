"""
Robust group-comparison statistics (Welch ANOVA + Games-Howell post-hoc) with
no dependency on pingouin -- implemented directly on numpy/scipy so the extended
analysis is fully reproducible.

Adds support for stratified comparisons (e.g. vehicle class within each traffic
regime), which is the new analytical contribution: differences masked in the
pooled comparison can emerge within a regime.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
    from scipy.stats import studentized_range as _srange
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def _sig(p: float, alpha: float = 0.05) -> str:
    if p is None or not np.isfinite(p):
        return "Insufficient data"
    return "Significant Difference" if p < alpha else "No Significant Difference"


def welch_anova(groups: Sequence[np.ndarray]) -> Dict[str, float]:
    """One-way Welch ANOVA for k>=2 groups with unequal variances."""
    groups = [np.asarray(g, dtype=float) for g in groups if np.asarray(g).size >= 2]
    k = len(groups)
    if k < 2:
        return {"F": np.nan, "df1": np.nan, "df2": np.nan, "p": np.nan}
    n = np.array([g.size for g in groups], dtype=float)
    means = np.array([g.mean() for g in groups])
    var = np.array([g.var(ddof=1) for g in groups])
    var = np.where(var <= 0, 1e-12, var)

    w = n / var
    W = w.sum()
    grand = (w * means).sum() / W

    numerator = (w * (means - grand) ** 2).sum() / (k - 1)
    tmp = ((1 - w / W) ** 2 / (n - 1)).sum()
    denominator = 1 + (2 * (k - 2) / (k ** 2 - 1)) * tmp
    F = numerator / denominator
    df1 = k - 1
    df2 = (k ** 2 - 1) / (3 * tmp) if tmp > 0 else np.inf
    p = float(_stats.f.sf(F, df1, df2)) if _HAVE_SCIPY else np.nan
    return {"F": float(F), "df1": float(df1), "df2": float(df2), "p": p}


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = a.size, b.size
    sa, sb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / max(na + nb - 2, 1))
    if pooled <= 0:
        return 0.0
    d = (a.mean() - b.mean()) / pooled
    corr = 1 - 3 / (4 * (na + nb) - 9) if (na + nb) > 3 else 1.0
    return float(d * corr)


def games_howell(group_values: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Games-Howell post-hoc pairwise comparisons (unequal n and variance)."""
    names = [g for g, v in group_values.items() if np.asarray(v).size >= 2]
    k = len(names)
    rows = []
    if k < 2:
        return pd.DataFrame(columns=["Group 1", "Group 2", "t-Statistic", "df",
                                     "p-Value", "Hedges g", "Significance"])
    for i in range(k):
        for j in range(i + 1, k):
            a = np.asarray(group_values[names[i]], dtype=float)
            b = np.asarray(group_values[names[j]], dtype=float)
            na, nb = a.size, b.size
            ma, mb = a.mean(), b.mean()
            va, vb = a.var(ddof=1), b.var(ddof=1)
            se = np.sqrt(va / na + vb / nb)
            if se <= 0:
                continue
            t = abs(ma - mb) / se
            df = (va / na + vb / nb) ** 2 / (
                (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
            q = t * np.sqrt(2.0)
            p = float(_srange.sf(q, k, df)) if _HAVE_SCIPY else np.nan
            rows.append({
                "Group 1": names[i], "Group 2": names[j],
                "t-Statistic": float(t), "df": float(df),
                "p-Value": p, "Hedges g": _hedges_g(a, b),
                "Significance": _sig(p),
            })
    return pd.DataFrame(rows)


def compare_params(df: pd.DataFrame, params: List[str],
                   group_col: str = "follower_type",
                   groups: Sequence[str] = ("small", "large", "av"),
                   alpha: float = 0.05) -> Dict[str, pd.DataFrame]:
    """Welch ANOVA + Games-Howell across `groups` for each parameter."""
    sub = df[df[group_col].isin(groups)]
    anova_rows, gh_frames = [], []
    for p in params:
        gv = {g: sub.loc[sub[group_col] == g, p].dropna().to_numpy(float) for g in groups}
        gv = {g: v for g, v in gv.items() if v.size >= 2}
        wa = welch_anova(list(gv.values()))
        anova_rows.append({"Parameter": p, "F-value": wa["F"], "p-value": wa["p"],
                           "Significance": _sig(wa["p"], alpha)})
        gh = games_howell(gv)
        if len(gh):
            gh.insert(0, "Parameter", p)
            gh_frames.append(gh)
    anova = pd.DataFrame(anova_rows)
    gh = pd.concat(gh_frames, ignore_index=True) if gh_frames else pd.DataFrame()
    return {"anova": anova, "games_howell": gh}


def compare_params_by_regime(df: pd.DataFrame, params: List[str],
                             regimes: Sequence[str] = ("free_flow", "synchronized", "stop_and_go"),
                             alpha: float = 0.05) -> Dict[str, pd.DataFrame]:
    """Run class comparisons separately within each traffic regime."""
    anova_all, gh_all = [], []
    for r in regimes:
        sub = df[df["regime"] == r]
        if len(sub) < 6:
            continue
        res = compare_params(sub, params, alpha=alpha)
        a = res["anova"].copy(); a.insert(0, "Regime", r); anova_all.append(a)
        if len(res["games_howell"]):
            g = res["games_howell"].copy(); g.insert(0, "Regime", r); gh_all.append(g)
    return {
        "anova": pd.concat(anova_all, ignore_index=True) if anova_all else pd.DataFrame(),
        "games_howell": pd.concat(gh_all, ignore_index=True) if gh_all else pd.DataFrame(),
    }


def _selftest():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "follower_type": ["small"] * 50 + ["large"] * 50 + ["av"] * 50,
        "T": np.concatenate([rng.normal(1.0, 0.3, 50),
                             rng.normal(1.5, 0.3, 50),
                             rng.normal(1.05, 0.3, 50)]),
    })
    res = compare_params(df, ["T"])
    print(res["anova"].to_string(index=False))
    print(res["games_howell"].to_string(index=False))


if __name__ == "__main__":
    _selftest()
