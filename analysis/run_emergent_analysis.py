"""
Run the emergent-dynamics analysis (string stability + mixed-autonomy safety/
throughput) and produce publication figures + summary tables.

Runs out-of-the-box on the paper's Table 2 class-mean IDM parameters. To use
data-driven (or regime-stratified) means instead, pass a CSV of per-episode
calibrated parameters via --params-csv (expects columns: follower_type, T, a,
b, v0, s0, delta), optionally filtered by --regime.

Usage:
    python analysis/run_emergent_analysis.py
    python analysis/run_emergent_analysis.py --params-csv "Results IDM/idm_calib_episodes_results.csv"
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cf_models import IDMParams, PAPER_IDM_MEANS, CLASS_LABELS  # noqa: E402
import string_stability as ss  # noqa: E402
import mixed_autonomy as ma  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results Emergent")
CLASS_COLORS = {"small": "#1f77b4", "large": "#d62728", "av": "#2ca02c"}


def load_class_means(params_csv: str | None, regime: str | None) -> dict[str, IDMParams]:
    if not params_csv:
        print(" Using paper Table 2 class-mean IDM parameters.")
        return dict(PAPER_IDM_MEANS)
    df = pd.read_csv(params_csv)
    if regime and "regime" in df.columns:
        df = df[df["regime"] == regime]
        print(f" Filtered to regime='{regime}': {len(df)} episodes.")
    means = {}
    for c in ("small", "large", "av"):
        sub = df[df["follower_type"] == c]
        if len(sub) == 0:
            print(f" [WARN] no '{c}' episodes; falling back to paper means for it.")
            means[c] = PAPER_IDM_MEANS[c]
            continue
        means[c] = IDMParams(
            T=float(sub["T"].mean()), a=float(sub["a"].mean()), b=float(sub["b"].mean()),
            v0=float(sub["v0"].mean()), s0=float(sub["s0"].mean()),
            delta=float(sub["delta"].mean()) if "delta" in sub.columns else 4.0,
        )
    return means


def run_string_stability(params: dict[str, IDMParams]) -> pd.DataFrame:
    print("\n[1/2] String-stability analysis (analytical transfer function)...")
    v_grid = np.linspace(1.0, 33.0, 161)
    rows = []
    curves = {}
    for c, p in params.items():
        curve = ss.stability_curve(p, v_grid)
        curves[c] = curve
        lo, hi = ss.critical_speed_band(p, v_grid)
        # worst-case amplification across the speed band
        finite = curve["g_max"][np.isfinite(curve["g_max"])]
        g_peak = float(np.nanmax(finite)) if finite.size else float("nan")
        unstable_frac = float(np.mean(~curve["stable"]))
        rows.append({
            "Class": CLASS_LABELS[c],
            "Peak |H| (worst)": round(g_peak, 3),
            "Unstable speed band (m/s)": "none" if lo is None else f"{lo:.1f}-{hi:.1f}",
            "Fraction of speeds unstable": round(unstable_frac, 3),
        })
        print(f"  {CLASS_LABELS[c]:>16}: peak|H|={g_peak:.3f}, "
              f"unstable band={'none' if lo is None else f'{lo:.1f}-{hi:.1f} m/s'}")
    if HAVE_MPL:
        _plot_stability(curves, params)
    return pd.DataFrame(rows)


def _plot_stability(curves, params):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for c, curve in curves.items():
        ax[0].plot(curve["v"], curve["g_max"], color=CLASS_COLORS[c],
                   label=CLASS_LABELS[c], lw=2)
        ax[1].plot(curve["v"], curve["lambda2"], color=CLASS_COLORS[c],
                   label=CLASS_LABELS[c], lw=2)
    ax[0].axhline(1.0, color="k", ls="--", lw=1, label="stability limit")
    ax[0].set_xlabel("Equilibrium speed $v_e$ (m/s)")
    ax[0].set_ylabel(r"Peak amplification $\max_\omega |H(\omega)|$")
    ax[0].set_title("String stability: speed-to-speed gain")
    ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_ylim(0.9, 1.25)
    ax[1].axhline(0.0, color="k", ls="--", lw=1)
    ax[1].set_xlabel("Equilibrium speed $v_e$ (m/s)")
    ax[1].set_ylabel(r"Stability coefficient $\lambda_2$")
    ax[1].set_title(r"Low-frequency criterion ($\lambda_2\geq 0$ stable)")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_string_stability.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure -> {path}")


def run_mpr_sweep(params: dict[str, IDMParams]) -> pd.DataFrame:
    print("\n[2/2] Mixed-autonomy MPR sweep (safety + throughput)...")
    mprs = np.linspace(0.0, 1.0, 11)
    res = ma.sweep_mpr(params, mprs=mprs, n_realizations=30)
    df = pd.DataFrame({
        "AV penetration": res.mpr,
        "String amplification": np.round(res.amp_mean, 3),
        "TET (s)": np.round(res.tet_mean, 2),
        "TIT (s)": np.round(res.tit_mean, 3),
        "Min TTC (s)": np.round(res.min_ttc_mean, 2),
        "Max DRAC (m/s2)": np.round(res.drac_mean, 3),
        "Throughput (veh/h)": np.round(res.throughput_mean, 0),
        "Mean speed (m/s)": np.round(res.speed_mean, 2),
    })
    for _, r in df.iterrows():
        print(f"  MPR={r['AV penetration']:.0%}: amp={r['String amplification']:.2f}, "
              f"TET={r['TET (s)']:.1f}s, minTTC={r['Min TTC (s)']:.1f}s, "
              f"q={r['Throughput (veh/h)']:.0f} veh/h")
    if HAVE_MPL:
        _plot_mpr(res)
    return df


def _plot_mpr(res: "ma.MPRSweepResult"):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    x = res.mpr * 100
    ax[0, 0].errorbar(x, res.amp_mean, yerr=res.amp_std, marker="o", capsize=3, color="#6a3d9a")
    ax[0, 0].axhline(1.0, color="k", ls="--", lw=1, label="stability limit")
    ax[0, 0].set_ylabel("String amplification ratio")
    ax[0, 0].set_title("String stability vs AV penetration")
    ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(x, res.drac_mean, marker="s", color="#e31a1c")
    ax[0, 1].set_ylabel(r"Mean max DRAC (m/s$^2$)")
    ax[0, 1].set_title("Required braking intensity vs AV penetration")
    ax[0, 1].grid(alpha=0.3)

    ax[1, 0].plot(x, res.min_ttc_mean, marker="^", color="#ff7f00")
    ax[1, 0].set_xlabel("AV penetration (%)")
    ax[1, 0].set_ylabel("Mean minimum TTC (s)")
    ax[1, 0].set_title("Closest approach vs AV penetration")
    ax[1, 0].grid(alpha=0.3)

    ax[1, 1].plot(x, res.throughput_mean, marker="D", color="#1f78b4")
    ax[1, 1].set_xlabel("AV penetration (%)")
    ax[1, 1].set_ylabel("Throughput (veh/h)")
    ax[1, 1].set_title("Throughput vs AV penetration")
    ax[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_mpr_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-csv", default=None, help="per-episode calibrated params CSV")
    ap.add_argument("--regime", default=None, help="optional regime filter (free_flow/synchronized/stop_and_go)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("EMERGENT-DYNAMICS ANALYSIS (string stability + mixed-autonomy safety)")
    print("=" * 70)

    params = load_class_means(args.params_csv, args.regime)
    for c, p in params.items():
        print(f"  {CLASS_LABELS[c]:>16}: T={p.T:.3f} a={p.a:.3f} b={p.b:.3f} "
              f"v0={p.v0:.2f} s0={p.s0:.3f} delta={p.delta:.3f}")

    stab_df = run_string_stability(params)
    mpr_df = run_mpr_sweep(params)

    stab_path = os.path.join(OUT_DIR, "table_string_stability.csv")
    mpr_path = os.path.join(OUT_DIR, "table_mpr_sweep.csv")
    stab_df.to_csv(stab_path, index=False)
    mpr_df.to_csv(mpr_path, index=False)
    print(f"\nSaved tables -> {stab_path}\n             -> {mpr_path}")
    print("Done.")


if __name__ == "__main__":
    main()
