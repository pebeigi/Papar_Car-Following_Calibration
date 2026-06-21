"""
Out-of-sample validation and practical-identifiability analysis.

Addresses the two biggest methodological weaknesses of the rejected version:
  (1) no out-of-sample validation (in-sample RMSE can reward overfitting), and
  (2) an unexplained PT null result (few significant class differences).

Three tools
-----------
A. param_dispersion_from_csv(): practical identifiability from the multi-start
   GA repeats the pipeline already runs (it writes <param>_std columns). The
   normalized restart dispersion CV = std/|mean| measures how well each
   parameter is pinned down by the data. Poorly identifiable parameters have
   large CV -> their class differences are statistically diluted. This reframes
   the PT null result as a *practical non-identifiability* finding. Runs on the
   existing result CSVs; no recomputation needed.

B. out_of_sample_idm(): temporal hold-out. Calibrate IDM on the first `frac`
   of each episode and predict the remainder; compare in-sample vs out-of-sample
   RMSE to quantify generalization. Reuses the calibration core.

C. profile_sensitivity_idm(): one-at-a-time profile of the fit around the
   optimum; a flat profile => structural non-identifiability.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# A. Practical identifiability from existing multi-start results
# ---------------------------------------------------------------------------
def param_dispersion_from_csv(csv_path: str,
                              params: Sequence[str],
                              group_col: str = "follower_type",
                              groups: Sequence[str] = ("small", "large", "av")
                              ) -> pd.DataFrame:
    """
    Per-class practical identifiability from <param>_std columns.

    Returns a tidy DataFrame: Parameter x Class -> median restart CV
    (= std/|mean| across GA restarts, medianed over episodes). Larger CV means
    the parameter is less identifiable from the trajectory.
    """
    df = pd.read_csv(csv_path)
    rows = []
    for p in params:
        std_col = f"{p}_std"
        if std_col not in df.columns or p not in df.columns:
            continue
        rec = {"Parameter": p}
        for g in groups:
            sub = df[df[group_col] == g]
            if len(sub) == 0:
                rec[g] = np.nan
                continue
            denom = sub[p].abs().replace(0, np.nan)
            cv = (sub[std_col] / denom).replace([np.inf, -np.inf], np.nan)
            rec[g] = float(np.nanmedian(cv))
        rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out):
        out["overall_median_CV"] = out[list(groups)].median(axis=1, skipna=True)
    return out


def identifiability_verdict(disp: pd.DataFrame,
                            well_thr: float = 0.10,
                            poor_thr: float = 0.30,
                            cv_col: str = "overall_median_CV") -> pd.DataFrame:
    """Label each parameter well / moderately / poorly identifiable."""
    if cv_col not in disp.columns:
        return disp
    def verdict(cv):
        if not np.isfinite(cv):
            return "unknown"
        if cv <= well_thr:
            return "well identifiable"
        if cv <= poor_thr:
            return "moderately identifiable"
        return "poorly identifiable"
    out = disp.copy()
    out["Identifiability"] = out[cv_col].apply(verdict)
    return out


# ---------------------------------------------------------------------------
# B. Temporal out-of-sample validation (IDM)
# ---------------------------------------------------------------------------
def _slice_episode(ep, lo: int, hi: int):
    """Return a shallow copy of an Episode with df rows [lo:hi]."""
    import copy
    new = copy.copy(ep)
    new.df = ep.df.iloc[lo:hi].reset_index(drop=True)
    new.start_t = float(new.df["t"].iloc[0])
    new.end_t = float(new.df["t"].iloc[-1])
    return new


def out_of_sample_idm(episodes: List, frac: float = 0.7,
                      seed: int = 42) -> pd.DataFrame:
    """
    Calibrate IDM on first `frac` of each episode, predict the rest.

    Returns per-episode in-sample vs out-of-sample position RMSE/MAE.
    Requires the raw episodes (from IDM_calibration_tgsim.build_episodes).
    """
    import random
    import IDM_calibration_tgsim as idm

    rows = []
    for k, ep in enumerate(episodes):
        n = len(ep.df)
        if n < 20:
            continue
        split = int(n * frac)
        if split < 10 or (n - split) < 5:
            continue

        random.seed(seed + k)
        np.random.seed(seed + k)
        train = _slice_episode(ep, 0, split)
        params, _fit = idm.calibrate_episode_ga(train, show_progress=False)

        # Forward-simulate the FULL episode with train-only params.
        d = ep.df
        t = d["t"].to_numpy(float)
        x_lead = d["x_lead"].to_numpy(float)
        v_lead = d["v_lead"].to_numpy(float)
        x_obs = d["x_foll"].to_numpy(float)
        lead_length = float(d["lead_length"].iloc[0]) if "lead_length" in d.columns else 0.0
        follower_length = float(d["follower_length"].iloc[0]) if "follower_length" in d.columns else 0.0
        x_sim, _v_sim = idm.simulate_follower(
            t=t, x_lead=x_lead, v_lead=v_lead,
            x0=float(x_obs[0]), v0=float(d["v_foll"].iloc[0]), p=params,
            lead_length=lead_length, follower_length=follower_length)

        err = x_obs - x_sim
        is_rmse = float(np.sqrt(np.mean(err[:split] ** 2)))
        oos_rmse = float(np.sqrt(np.mean(err[split:] ** 2)))
        is_mae = float(np.mean(np.abs(err[:split])))
        oos_mae = float(np.mean(np.abs(err[split:])))
        rows.append({
            "follower_type": ep.follower_type,
            "regime": getattr(ep, "regime", "unknown"),
            "in_sample_rmse": is_rmse, "out_sample_rmse": oos_rmse,
            "in_sample_mae": is_mae, "out_sample_mae": oos_mae,
            "degradation_ratio": oos_rmse / is_rmse if is_rmse > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_oos(oos_df: pd.DataFrame) -> pd.DataFrame:
    """Mean in/out-of-sample errors by vehicle class."""
    if len(oos_df) == 0:
        return oos_df
    return (oos_df.groupby("follower_type")
            [["in_sample_rmse", "out_sample_rmse", "degradation_ratio"]]
            .agg(["mean", "std", "count"]))


# ---------------------------------------------------------------------------
# C. Profile sensitivity (IDM): how sharp is the optimum per parameter?
# ---------------------------------------------------------------------------
def profile_sensitivity_idm(ep, params, n_grid: int = 21,
                            span_frac: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Vary each parameter on a grid around its calibrated value (others fixed),
    recording RMSE. Flat profiles indicate non-identifiable directions.
    Returns {param: (grid_values, rmse_values)}.
    """
    import IDM_calibration_tgsim as idm

    base = {k: getattr(params, k) for k in ("T", "a", "b", "v0", "s0", "delta")}
    out = {}
    for name in ("T", "a", "b", "v0", "s0"):
        lo, hi = idm.BOUNDS[name]
        center = base[name]
        half = span_frac * (hi - lo) / 2
        grid = np.clip(np.linspace(center - half, center + half, n_grid), lo, hi)
        rmses = []
        for val in grid:
            kw = dict(base); kw[name] = float(val)
            p = idm.IDMParams(**kw)
            m = idm.calculate_performance_metrics(ep, p)
            rmses.append(m["rmse"])
        out[name] = (grid, np.asarray(rmses))
    return out


def _demo():
    """Demonstrate part A on a synthetic results CSV (no raw data needed)."""
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "follower_type": rng.choice(["small", "large", "av"], n),
        # IDM T: well identifiable (small restart std)
        "T": rng.uniform(0.8, 1.6, n),
        "T_std": rng.uniform(0.01, 0.05, n),
        # PT-like Tmax: poorly identifiable (large restart std)
        "Tmax": rng.uniform(5, 8, n),
        "Tmax_std": rng.uniform(1.5, 3.0, n),
    })
    path = "/tmp/_demo_results.csv"
    df.to_csv(path, index=False)
    disp = param_dispersion_from_csv(path, ["T", "Tmax"])
    print(identifiability_verdict(disp).to_string(index=False))


if __name__ == "__main__":
    _demo()
