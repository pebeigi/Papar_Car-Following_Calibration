"""
Model-agnostic car-following calibration engine.

This engine reuses the data ingestion, episode extraction, regime labeling, and
robust-statistics machinery already implemented for IDM (imported from
``IDM_calibration_tgsim``) and adds a generic genetic-algorithm calibrator that
works for ANY closed-loop car-following model described by a ``ModelSpec``.

A model author only needs to provide:
  * the parameter names and bounds, and
  * a pure-Python ``simulate(t, x_lead, v_lead, x0, v0, l_eff, theta)`` function
    that integrates the follower trajectory in CLOSED LOOP (gap recomputed from
    the simulated position at every step).

The engine then provides per-episode calibration (with multi-start dispersion for
identifiability), the near-AV equal-sampling selection, regime labeling,
parameter/performance tables, and Welch-ANOVA / Games-Howell statistics (pooled
and per regime) -- producing the same family of output files as the IDM script.

Run a model via its thin wrapper, e.g. ``python OVRV_calibration_tgsim.py``.
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse all heavy infrastructure from the IDM script (data, episodes, regime).
import IDM_calibration_tgsim as idm

# Generic robust statistics (pure numpy/scipy; no pingouin).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis"))
try:
    import stats_tests as _stats_tests
    STATS_TESTS_AVAILABLE = True
except Exception:  # pragma: no cover
    STATS_TESTS_AVAILABLE = False

# simulate(t, x_lead, v_lead, x0, v0, l_eff, theta, aux) -> (x, v)
SimulateFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, float, float, float, np.ndarray, object],
    Tuple[np.ndarray, np.ndarray],
]


@dataclass
class ModelSpec:
    """Everything the engine needs to calibrate a specific CF model."""
    name: str                      # short id, e.g. "ovrv" (used in folder/file names)
    pretty: str                    # human label, e.g. "OVRV"
    param_names: List[str]         # ordered parameter names
    bounds: np.ndarray             # shape (n_params, 2): (lo, hi) per parameter
    simulate: SimulateFn           # closed-loop trajectory simulator
    stat_params: Optional[List[str]] = None  # subset compared statistically (default: all)
    # Optional: per-episode auxiliary constant (e.g. PT v_desired) computed once
    # from the full episode and passed to ``simulate`` as the ``aux`` argument.
    prep: Optional[Callable[[object], object]] = None
    # Optional: per-episode plotter, signature plot(ep, theta, aux, output_path).
    plot: Optional[Callable[[object, np.ndarray, object, str], None]] = None

    def __post_init__(self):
        self.bounds = np.asarray(self.bounds, dtype=float)
        if self.stat_params is None:
            self.stat_params = list(self.param_names)

    def aux_for(self, ep) -> object:
        return self.prep(ep) if self.prep is not None else None


# ------------------------------------------------------------------ utilities
def _l_eff(d: pd.DataFrame) -> float:
    ll = float(d["lead_length"].iloc[0]) if "lead_length" in d.columns else 0.0
    fl = float(d["follower_length"].iloc[0]) if "follower_length" in d.columns else 0.0
    return 0.5 * ll + 0.5 * fl


def _episode_arrays(ep) -> Dict[str, np.ndarray]:
    d = ep.df
    return {
        "t": d["t"].to_numpy(float),
        "x_lead": d["x_lead"].to_numpy(float),
        "v_lead": d["v_lead"].to_numpy(float),
        "x_obs": d["x_foll"].to_numpy(float),
        "v_obs": d["v_foll"].to_numpy(float),
        "l_eff": _l_eff(d),
    }


def fitness(ep, spec: ModelSpec, theta: np.ndarray,
            w_pos: float, w_speed: float, aux: object = None) -> float:
    a = _episode_arrays(ep)
    x_sim, v_sim = spec.simulate(a["t"], a["x_lead"], a["v_lead"],
                                 float(a["x_obs"][0]), float(a["v_obs"][0]),
                                 a["l_eff"], theta, aux)
    err = w_pos * np.abs(a["x_obs"] - x_sim) + w_speed * np.abs(a["v_obs"] - v_sim)
    return float(np.sum(err))


def metrics(ep, spec: ModelSpec, theta: np.ndarray, aux: object = None) -> Dict[str, float]:
    a = _episode_arrays(ep)
    x_sim, _ = spec.simulate(a["t"], a["x_lead"], a["v_lead"],
                             float(a["x_obs"][0]), float(a["v_obs"][0]),
                             a["l_eff"], theta, aux)
    err = a["x_obs"] - x_sim
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((a["x_obs"] - np.mean(a["x_obs"])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "r_squared": float(r2)}


# ------------------------------------------------------------------ genetic algorithm
def _random_vec(spec: ModelSpec) -> np.ndarray:
    lo, hi = spec.bounds[:, 0], spec.bounds[:, 1]
    return lo + np.random.rand(len(spec.param_names)) * (hi - lo)


def _clamp_vec(v: np.ndarray, spec: ModelSpec) -> np.ndarray:
    return np.minimum(np.maximum(v, spec.bounds[:, 0]), spec.bounds[:, 1])


def _tournament(pop, fit, k):
    idxs = random.sample(range(len(pop)), k)
    return pop[min(idxs, key=lambda i: fit[i])].copy()


def _crossover(a, b):
    mask = np.random.rand(a.size) < 0.5
    return np.where(mask, a, b), np.where(mask, b, a)


def _mutate(v, spec: ModelSpec):
    out = v.copy()
    for i in range(out.size):
        if random.random() < idm.GA_MUTATION_PROB:
            lo, hi = spec.bounds[i]
            out[i] += random.gauss(0.0, idm.GA_MUTATION_SCALE * (hi - lo))
    return _clamp_vec(out, spec)


def calibrate_episode_ga(ep, spec: ModelSpec,
                         w_pos: float, w_speed: float,
                         aux: object = None) -> Tuple[np.ndarray, float]:
    pop = [_random_vec(spec) for _ in range(idm.GA_POP)]
    fit = [fitness(ep, spec, ind, w_pos, w_speed, aux) for ind in pop]
    elite_n = max(1, int(idm.GA_ELITE_FRAC * idm.GA_POP))

    best_hist, no_improve = [], 0
    for gen in range(idm.GA_GENS):
        order = np.argsort(fit)
        pop = [pop[i] for i in order]
        fit = [fit[i] for i in order]
        best_hist.append(fit[0])
        if gen > 0:
            if best_hist[-2] - fit[0] < idm.GA_EARLY_STOP_TOL:
                no_improve += 1
            else:
                no_improve = 0
            if no_improve >= idm.GA_EARLY_STOP_GENS:
                break
        new_pop = pop[:elite_n]
        while len(new_pop) < idm.GA_POP:
            p1 = _tournament(pop, fit, idm.GA_TOURN_K)
            p2 = _tournament(pop, fit, idm.GA_TOURN_K)
            if random.random() < idm.GA_CROSSOVER_PROB:
                c1, c2 = _crossover(p1, p2)
            else:
                c1, c2 = p1, p2
            new_pop.append(_mutate(c1, spec))
            if len(new_pop) < idm.GA_POP:
                new_pop.append(_mutate(c2, spec))
        pop = new_pop
        fit = [fitness(ep, spec, ind, w_pos, w_speed, aux) for ind in pop]

    best_i = int(np.argmin(fit))
    return pop[best_i], float(fit[best_i])


def calibrate_episode_robust(ep, spec: ModelSpec, w_pos, w_speed,
                             n_runs: int, use_best: bool,
                             base_seed: Optional[int],
                             aux: object = None) -> Tuple[np.ndarray, float, Dict]:
    if n_runs == 1:
        v, f = calibrate_episode_ga(ep, spec, w_pos, w_speed, aux)
        return v, f, {"n_runs": 1, "std": np.zeros(len(spec.param_names))}
    all_v, all_f = [], []
    for r in range(n_runs):
        if base_seed is not None:
            random.seed(base_seed + r)
            np.random.seed(base_seed + r)
        v, f = calibrate_episode_ga(ep, spec, w_pos, w_speed, aux)
        all_v.append(v)
        all_f.append(f)
    all_v = np.array(all_v)
    all_f = np.array(all_f)
    mean_v = all_v.mean(axis=0)
    std_v = all_v.std(axis=0)
    if use_best:
        bi = int(np.argmin(all_f))
        best_v, best_f = all_v[bi], float(all_f[bi])
    else:
        best_v, best_f = mean_v, float(all_f.mean())
    stats = {
        "n_runs": n_runs,
        "mean_fitness": float(all_f.mean()),
        "std_fitness": float(all_f.std()),
        "min_fitness": float(all_f.min()),
        "max_fitness": float(all_f.max()),
        "std": std_v,
    }
    return best_v, best_f, stats


# ------------------------------------------------------------------ worker
def _worker(args) -> Tuple[int, Dict]:
    (episode_idx, ep, dataset_name, spec, w_pos, w_speed,
     base_seed, episode_base_seed) = args

    if episode_base_seed is not None:
        random.seed(episode_base_seed)
        np.random.seed(episode_base_seed)
    elif base_seed is not None:
        random.seed(base_seed + episode_idx - 1)
        np.random.seed(base_seed + episode_idx - 1)

    aux = spec.aux_for(ep)
    if idm.N_CALIBRATION_RUNS > 1:
        theta, fit, cstats = calibrate_episode_robust(
            ep, spec, w_pos, w_speed,
            n_runs=idm.N_CALIBRATION_RUNS, use_best=idm.USE_BEST_RUN,
            base_seed=episode_base_seed, aux=aux)
    else:
        theta, fit = calibrate_episode_ga(ep, spec, w_pos, w_speed, aux)
        cstats = None

    m = metrics(ep, spec, theta, aux)
    gap = ep.df["gap"].to_numpy(float) if "gap" in ep.df.columns else np.array([])
    rd = {
        "dataset": dataset_name,
        "episode_idx": episode_idx,
        "run_index": ep.run_index,
        "follower_id": ep.follower_id,
        "leader_id": ep.leader_id,
        "follower_type": ep.follower_type,
        "regime": idm.episode_regime(ep),
        "start_t": ep.start_t,
        "end_t": ep.end_t,
        "duration_s": ep.end_t - ep.start_t,
        "min_gap": float(np.min(gap)) if gap.size else np.nan,
        "max_gap": float(np.max(gap)) if gap.size else np.nan,
        "fitness": float(fit),
        "rmse": m["rmse"], "mae": m["mae"], "r_squared": m["r_squared"],
    }
    for j, name in enumerate(spec.param_names):
        rd[name] = float(theta[j])
    if cstats is not None:
        for j, name in enumerate(spec.param_names):
            rd[f"{name}_std"] = float(cstats["std"][j])
        rd["fitness_std"] = cstats.get("std_fitness", np.nan)
    return episode_idx, rd


# ------------------------------------------------------------------ selection
def _select_episodes(episodes, dataset_name) -> List:
    """Replicate the IDM near-AV equal-sampling selection."""
    by_type: Dict[str, List] = {}
    for ep in episodes:
        if ep.follower_type != "unknown":
            by_type.setdefault(ep.follower_type, []).append(ep)
    for vt, lst in by_type.items():
        print(f"   - {vt}: {len(lst)} episodes")

    if not idm.CALIBRATE_ONLY_NEAR_AVS:
        return list(episodes)

    av_eps = by_type.get("av", [])
    if not av_eps:
        print(f"   [WARN] NO AV EPISODES in {dataset_name}; skipping.")
        return []

    av_info = []
    for ep in av_eps:
        lane = ep.df["lane_id"].iloc[0] if "lane_id" in ep.df.columns else None
        pmin = ep.df["x_foll"].min() if "x_foll" in ep.df.columns else None
        pmax = ep.df["x_foll"].max() if "x_foll" in ep.df.columns else None
        av_info.append((ep.run_index, ep.start_t, ep.end_t, lane, pmin, pmax))

    def around(ep, t_tol=5.0, s_tol=100.0) -> bool:
        lane = ep.df["lane_id"].iloc[0] if "lane_id" in ep.df.columns else None
        pmin = ep.df["x_foll"].min() if "x_foll" in ep.df.columns else None
        pmax = ep.df["x_foll"].max() if "x_foll" in ep.df.columns else None
        for av_run, av_s, av_e, av_lane, av_pmin, av_pmax in av_info:
            if ep.run_index is not None and av_run is not None and ep.run_index != av_run:
                continue
            if not (ep.end_t < av_s - t_tol or ep.start_t > av_e + t_tol):
                same_lane = lane is not None and av_lane is not None and lane == av_lane
                near = False
                if None not in (pmin, pmax, av_pmin, av_pmax):
                    near = abs((pmin + pmax) / 2 - (av_pmin + av_pmax) / 2) < s_tol
                if same_lane or near:
                    return True
        return False

    n_av = len(av_eps)
    small_near = [e for e in by_type.get("small", []) if around(e)]
    large_near = [e for e in by_type.get("large", []) if around(e)]
    chosen = list(av_eps)
    if small_near:
        k = min(n_av, len(small_near))
        chosen += random.sample(small_near, k) if len(small_near) > k else small_near
    if large_near:
        k = min(n_av, len(large_near))
        chosen += random.sample(large_near, k) if len(large_near) > k else large_near
    print(f"   Final: {len(chosen)} episodes "
          f"(AV={n_av}, small={min(n_av, len(small_near))}, large={min(n_av, len(large_near))})")
    return chosen


def process_single_dataset(csv_path, dataset_name, spec: ModelSpec,
                           w_pos, w_speed, results_dir=None) -> List[Dict]:
    print(f"\n{'=' * 70}\nProcessing dataset: {dataset_name} [{spec.pretty}]\n{'=' * 70}")
    if not os.path.exists(csv_path):
        print(f" WARNING: file not found: {csv_path} (skipping)")
        return []
    df = pd.read_csv(csv_path)
    print(f" Loaded {len(df):,} rows")
    sc = idm.infer_schema(df, dataset_name=dataset_name)
    episodes = idm.build_episodes(df, sc)
    print(f" Extracted {len(episodes):,} episodes")
    if not episodes:
        return []

    print(" Selecting episodes:")
    to_cal = _select_episodes(episodes, dataset_name)
    if not to_cal:
        return []

    base_seed = idm.RANDOM_SEED
    args = []
    for i, ep in enumerate(to_cal, 1):
        ep_seed = None
        if base_seed is not None and idm.N_CALIBRATION_RUNS > 1:
            ep_seed = base_seed + (i - 1) * idm.N_CALIBRATION_RUNS
        args.append((i, ep, dataset_name, spec, w_pos, w_speed, base_seed, ep_seed))

    n_workers = idm.N_PARALLEL_WORKERS or cpu_count()
    n_workers = max(1, min(n_workers, len(to_cal)))
    results: Dict[int, Dict] = {}
    print(f" Calibrating {len(to_cal)} episodes with {n_workers} worker(s)...")

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            done = 0
            for idx, rd in pool.imap_unordered(_worker, args):
                results[idx] = rd
                done += 1
                if done == 1 or done % 10 == 0 or done == len(to_cal):
                    print(f"   [{done}/{len(to_cal)}] RMSE={rd['rmse']:.3f} R2={rd['r_squared']:.3f}")
    else:
        for a in args:
            idx, rd = _worker(a)
            results[idx] = rd
            if idx == 1 or idx % 10 == 0 or idx == len(to_cal):
                print(f"   [{idx}/{len(to_cal)}] RMSE={rd['rmse']:.3f} R2={rd['r_squared']:.3f}")

    ordered = [results[i] for i in sorted(results)]
    if results_dir is not None:
        _make_episode_plots(to_cal, ordered, spec, results_dir, dataset_name)
    return ordered


# ------------------------------------------------------------------ tables + stats
def _class_param_table(res_df, spec) -> pd.DataFrame:
    sub = res_df[res_df["follower_type"].isin(["small", "large", "av"])]
    g = sub.groupby("follower_type")
    rows = []
    for p in spec.param_names:
        row = {"Parameter": p}
        for c, lab in (("small", "Small"), ("large", "Large"), ("av", "AV")):
            if c in g.groups:
                vals = g.get_group(c)[p]
                row[lab] = f"{vals.mean():.4g} ± {vals.std():.3g} ({vals.median():.4g})"
            else:
                row[lab] = "N/A"
        rows.append(row)
    return pd.DataFrame(rows)


def _class_perf_table(res_df) -> pd.DataFrame:
    sub = res_df[res_df["follower_type"].isin(["small", "large", "av"])]
    g = sub.groupby("follower_type")[["rmse", "mae", "r_squared"]].mean()
    rows = []
    for c, lab in (("small", "Small"), ("large", "Large"), ("av", "AV")):
        if c in g.index:
            rows.append({"Vehicle Type": lab, "RMSE": f"{g.loc[c,'rmse']:.3f}",
                         "MAE": f"{g.loc[c,'mae']:.3f}", "R2": f"{g.loc[c,'r_squared']:.4f}"})
    return pd.DataFrame(rows)


def _kruskal_table(res_df, params) -> pd.DataFrame:
    """Generic Kruskal-Wallis across vehicle classes (non-parametric)."""
    from scipy import stats as _ss
    sub = res_df[res_df["follower_type"].isin(["small", "large", "av"])]
    rows = []
    for p in params:
        groups = [sub[sub["follower_type"] == c][p].dropna().to_numpy()
                  for c in ("small", "large", "av")]
        groups = [g for g in groups if len(g) >= 1]
        if len(groups) >= 2 and all(len(g) >= 1 for g in groups):
            try:
                H, pv = _ss.kruskal(*groups)
            except Exception:
                H, pv = np.nan, np.nan
        else:
            H, pv = np.nan, np.nan
        rows.append({"Parameter": p, "H-value": H, "p-value": pv,
                     "Significant": "Yes" if (pv == pv and pv < 0.05) else "No"})
    return pd.DataFrame(rows)


def generic_episode_plot(spec: ModelSpec, ep, theta: np.ndarray, aux, output_path: str) -> None:
    """3-panel observed vs simulated plot for any model (position, speed, gap)."""
    if not getattr(idm, "MATPLOTLIB_AVAILABLE", False):
        return
    plt = idm.plt
    a = _episode_arrays(ep)
    d = ep.df
    x_sim, v_sim = spec.simulate(
        a["t"], a["x_lead"], a["v_lead"],
        float(a["x_obs"][0]), float(a["v_obs"][0]),
        a["l_eff"], theta, aux,
    )
    ll = float(d["lead_length"].iloc[0]) if "lead_length" in d.columns else 0.0
    fl = float(d["follower_length"].iloc[0]) if "follower_length" in d.columns else 0.0
    gap_obs = d["gap"].to_numpy(float) if "gap" in d.columns else a["x_lead"] - a["x_obs"] - 0.5 * ll - 0.5 * fl
    gap_sim = a["x_lead"] - x_sim - 0.5 * ll - 0.5 * fl
    param_str = ", ".join(f"{n}={theta[i]:.3g}" for i, n in enumerate(spec.param_names))
    type_label = ep.follower_type.upper() if ep.follower_type != "av" else "AV"

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(
        f"Follower {ep.follower_id} -> Leader {ep.leader_id}  |  {type_label}  |  "
        f"{ep.end_t - ep.start_t:.1f}s\n{spec.pretty}: {param_str}",
        fontsize=10,
    )
    t = a["t"]
    axes[0].plot(t, a["x_obs"], label="Observed", linewidth=2)
    axes[0].plot(t, x_sim, "--", label=f"Simulated ({spec.pretty})", linewidth=2)
    axes[0].plot(t, a["x_lead"], ":", label="Leader", linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel("Position (m)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, a["v_obs"], label="Observed", linewidth=2)
    axes[1].plot(t, v_sim, "--", label=f"Simulated ({spec.pretty})", linewidth=2)
    axes[1].plot(t, a["v_lead"], ":", label="Leader (Observed)", linewidth=1.5, alpha=0.8)
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, gap_obs, label="Observed gap", linewidth=2)
    axes[2].plot(t, gap_sim, "--", label=f"Simulated gap ({spec.pretty})", linewidth=2)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Gap (m)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_generic_plotter(spec: ModelSpec):
    """Return a plot callback that closes over ``spec``."""
    def _plot(ep, theta, aux, output_path):
        generic_episode_plot(spec, ep, theta, aux, output_path)
    return _plot


def _episode_plot_path(plots_dir: str, rd: Dict) -> str:
    """Filename keyed by follower/leader vehicle IDs."""
    return os.path.join(
        plots_dir,
        f"follower_{rd['follower_id']}_leader_{rd['leader_id']}_{rd['follower_type']}.png",
    )


def _make_episode_plots(to_cal, results, spec, results_dir, dataset_name) -> None:
    """Plot calibrated episodes (all by default; see MAX_PLOT_EPISODES)."""
    plotter = spec.plot if spec.plot is not None else make_generic_plotter(spec)
    if not getattr(idm, "MATPLOTLIB_AVAILABLE", False):
        return
    if not getattr(idm, "PLOT_COMPARISONS", False):
        return

    max_n = getattr(idm, "MAX_PLOT_EPISODES", None)
    batch = results if max_n is None else results[: int(max_n)]

    plots_dir = os.path.join(results_dir, "comparison_plots", str(dataset_name))
    os.makedirs(plots_dir, exist_ok=True)
    n_ok = 0
    for rd in batch:
        ep = to_cal[rd["episode_idx"] - 1]
        theta = np.array([rd[p] for p in spec.param_names], dtype=float)
        out = _episode_plot_path(plots_dir, rd)
        try:
            plotter(ep, theta, spec.aux_for(ep), out)
            n_ok += 1
        except Exception as e:
            print(f"   [WARN] plot failed follower {rd['follower_id']}: {e}")
    print(f"   Saved {n_ok}/{len(batch)} comparison plots -> {plots_dir}")


# ------------------------------------------------------------------ summary plots (post-calibration aggregates)
_CLASS_ORDER = ["small", "large", "av"]
_CLASS_LABELS = {"small": "Small", "large": "Large", "av": "AV"}
_CLASS_COLORS = {"small": "#4C72B0", "large": "#DD8452", "av": "#55A868"}


def _class_subset(res_df: pd.DataFrame) -> pd.DataFrame:
    return res_df[res_df["follower_type"].isin(_CLASS_ORDER)].copy()


def _boxplot_by_class(ax, data: pd.DataFrame, col: str, ylabel: str = "") -> None:
    """Simple boxplot of ``col`` for small / large / AV."""
    groups = [_CLASS_LABELS[c] for c in _CLASS_ORDER
              if c in data["follower_type"].values and data.loc[data["follower_type"] == c, col].notna().any()]
    if not groups:
        ax.set_visible(False)
        return
    vals = [data.loc[data["follower_type"] == c, col].dropna().to_numpy()
            for c in _CLASS_ORDER if _CLASS_LABELS[c] in groups]
    bp = ax.boxplot(vals, labels=groups, patch_artist=True, showfliers=True)
    for patch, c in zip(bp["boxes"], [k for k in _CLASS_ORDER if _CLASS_LABELS[k] in groups]):
        patch.set_facecolor(_CLASS_COLORS[c])
        patch.set_alpha(0.7)
    ax.set_ylabel(ylabel or col)
    ax.grid(True, axis="y", alpha=0.3)


def _save_fig(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    idm.plt.close(fig)


def _plot_param_distributions(sub: pd.DataFrame, spec: ModelSpec, out_dir: str) -> None:
    plt = idm.plt
    n = len(spec.param_names)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)
    fig.suptitle(f"{spec.pretty}: calibrated parameter distributions by vehicle class", fontsize=12)
    for i, pname in enumerate(spec.param_names):
        ax = axes[i // ncols][i % ncols]
        _boxplot_by_class(ax, sub, pname, pname)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "fig_param_distributions.png"))


def _plot_error_distributions(sub: pd.DataFrame, spec: ModelSpec, out_dir: str) -> None:
    plt = idm.plt
    metrics = [("rmse", "RMSE (m)"), ("mae", "MAE (m)"), ("r_squared", "R²"), ("fitness", "Fitness")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"{spec.pretty}: calibration error metrics by vehicle class", fontsize=12)
    for ax, (col, ylab) in zip(axes.ravel(), metrics):
        if col in sub.columns:
            _boxplot_by_class(ax, sub, col, ylab)
        else:
            ax.set_visible(False)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "fig_error_distributions.png"))


def _plot_rmse_vs_mae(sub: pd.DataFrame, spec: ModelSpec, out_dir: str) -> None:
    plt = idm.plt
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in _CLASS_ORDER:
        mask = sub["follower_type"] == c
        if not mask.any():
            continue
        ax.scatter(sub.loc[mask, "rmse"], sub.loc[mask, "mae"],
                   c=_CLASS_COLORS[c], label=_CLASS_LABELS[c], alpha=0.65, s=36, edgecolors="none")
    ax.set_xlabel("RMSE (m)")
    ax.set_ylabel("MAE (m)")
    ax.set_title(f"{spec.pretty}: per-episode fit quality")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "fig_rmse_vs_mae.png"))


def _plot_regime_counts(sub: pd.DataFrame, spec: ModelSpec, out_dir: str) -> None:
    if "regime" not in sub.columns or sub["regime"].nunique() <= 1:
        return
    plt = idm.plt
    ct = (sub.groupby(["follower_type", "regime"]).size()
          .unstack(fill_value=0).reindex(_CLASS_ORDER).fillna(0))
    if ct.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(ct.columns)), 5))
    x = np.arange(len(_CLASS_ORDER))
    width = 0.8 / max(len(ct.columns), 1)
    for i, regime in enumerate(ct.columns):
        offset = (i - (len(ct.columns) - 1) / 2) * width
        ax.bar(x + offset, ct[regime].to_numpy(), width=width, label=str(regime))
    ax.set_xticks(x)
    ax.set_xticklabels([_CLASS_LABELS[c] for c in _CLASS_ORDER])
    ax.set_ylabel("Episode count")
    ax.set_title(f"{spec.pretty}: traffic regime mix by vehicle class")
    ax.legend(title="Regime", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "fig_regime_counts.png"))


def _plot_identifiability_cv(sub: pd.DataFrame, spec: ModelSpec, out_dir: str) -> None:
    """Restart dispersion CV = std/|mean| per parameter (multi-start GA)."""
    std_cols = [f"{p}_std" for p in spec.param_names if f"{p}_std" in sub.columns]
    if not std_cols:
        return
    plt = idm.plt
    n = len(spec.param_names)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)
    fig.suptitle(f"{spec.pretty}: practical identifiability (restart CV = std/|mean|)", fontsize=11)
    for i, pname in enumerate(spec.param_names):
        ax = axes[i // ncols][i % ncols]
        scol = f"{pname}_std"
        if scol not in sub.columns:
            ax.set_visible(False)
            continue
        tmp = sub[["follower_type", pname, scol]].copy()
        tmp["cv"] = tmp[scol] / np.maximum(np.abs(tmp[pname]), 1e-9)
        _boxplot_by_class(ax, tmp, "cv", f"CV({pname})")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "fig_identifiability_cv.png"))


def _plot_errors_by_dataset(sub: pd.DataFrame, spec: ModelSpec, out_dir: str) -> None:
    if "dataset" not in sub.columns or sub["dataset"].nunique() <= 1:
        return
    plt = idm.plt
    datasets = sorted(sub["dataset"].unique())
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(datasets)), 5))
    positions, labels, data, colors = [], [], [], []
    pos = 0
    for ds in datasets:
        for c in _CLASS_ORDER:
            vals = sub.loc[(sub["dataset"] == ds) & (sub["follower_type"] == c), "rmse"].dropna()
            if len(vals) == 0:
                continue
            positions.append(pos)
            labels.append(f"{ds}\n{_CLASS_LABELS[c]}")
            data.append(vals.to_numpy())
            colors.append(_CLASS_COLORS[c])
            pos += 1
        pos += 0.5  # gap between datasets
    if not data:
        plt.close(fig)
        return
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_ylabel("RMSE (m)")
    ax.set_title(f"{spec.pretty}: RMSE by dataset and vehicle class")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "fig_rmse_by_dataset.png"))


def _plot_param_correlation(sub: pd.DataFrame, spec: ModelSpec, out_dir: str) -> None:
    """Pairwise correlation heatmap of calibrated parameters (pooled)."""
    cols = [p for p in spec.param_names if p in sub.columns]
    if len(cols) < 2:
        return
    plt = idm.plt
    corr = sub[cols].corr()
    fig, ax = plt.subplots(figsize=(max(6, len(cols)), max(5, len(cols) - 1)))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{spec.pretty}: parameter correlation (all episodes)")
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "fig_param_correlation.png"))


def _make_summary_plots(res_df: pd.DataFrame, spec: ModelSpec, results_dir: str) -> None:
    """Aggregate diagnostic figures written to ``summary_plots/``."""
    if not getattr(idm, "MATPLOTLIB_AVAILABLE", False):
        return
    if not getattr(idm, "PLOT_SUMMARY", True):
        return
    sub = _class_subset(res_df)
    if sub.empty:
        return

    out_dir = os.path.join(results_dir, "summary_plots")
    os.makedirs(out_dir, exist_ok=True)
    plt = idm.plt
    try:
        _plot_param_distributions(sub, spec, out_dir)
        _plot_error_distributions(sub, spec, out_dir)
        _plot_rmse_vs_mae(sub, spec, out_dir)
        _plot_regime_counts(sub, spec, out_dir)
        _plot_identifiability_cv(sub, spec, out_dir)
        _plot_errors_by_dataset(sub, spec, out_dir)
        _plot_param_correlation(sub, spec, out_dir)
        print(f" Summary plots -> {out_dir}/")
    except Exception as e:
        print(f" WARNING: summary plots failed: {e}")
        plt.close("all")


def run_calibration(spec: ModelSpec):
    """Top-level entry point a model wrapper calls in __main__."""
    if idm.RANDOM_SEED is not None:
        random.seed(idm.RANDOM_SEED)
        np.random.seed(idm.RANDOM_SEED)

    w_pos = idm.W_POS
    w_speed = idm.W_SPEED

    results_dir = os.path.join(
        idm.SCRIPT_DIR,
        f"Results {spec.pretty}" if idm.CALIBRATE_ONLY_NEAR_AVS else f"Results Total {spec.pretty}")
    # Allow sweep override.
    if "CALIB_OUTPUT_SUBFOLDER" in os.environ:
        results_dir = os.path.join(idm.SCRIPT_DIR, os.environ["CALIB_OUTPUT_SUBFOLDER"])
    os.makedirs(results_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tee = idm.Tee(os.path.join(results_dir, f"calibration_log_{ts}.txt"))
    try:
        print("=" * 70)
        print(f"{spec.pretty} Calibration on TGSIM (closed-loop)")
        print("=" * 70)
        print(f" Params: {spec.param_names}")
        print(f" W_POS={w_pos} W_SPEED={w_speed} | runs/episode={idm.N_CALIBRATION_RUNS} | seed={idm.RANDOM_SEED}")
        print(f" Results dir: {results_dir}")

        all_results: List[Dict] = []
        for path, name in zip(idm.CSV_PATHS, idm.DATASET_NAMES):
            all_results.extend(
                process_single_dataset(path, name, spec, w_pos, w_speed, results_dir))

        if not all_results:
            print("\n ERROR: no results produced (check Dataset/ files).")
            return
        res_df = pd.DataFrame(all_results)

        # ---- episode CSV ----
        ep_csv = os.path.join(results_dir, f"{spec.name}_calib_episodes_results.csv")
        res_df.to_csv(ep_csv, index=False)
        print(f"\n Wrote episodes: {ep_csv} ({len(res_df):,} rows)")

        # ---- excel summary (optional, matches IDM/PT output) ----
        if hasattr(idm, "create_episodes_excel"):
            try:
                idm.create_episodes_excel(
                    res_df, os.path.join(results_dir, f"{spec.name}_calib_episodes_summary.xlsx"))
            except Exception as e:
                print(f" WARNING: excel export failed: {e}")

        # ---- parameter + performance tables ----
        pt = _class_param_table(res_df, spec)
        idm.print_formatted_table(pt, f"{spec.pretty}: Calibrated Parameters by Vehicle Class")
        pt.to_csv(os.path.join(results_dir, f"{spec.name}_parameters_table.csv"), index=False)
        perf = _class_perf_table(res_df)
        idm.print_formatted_table(perf, f"{spec.pretty}: Performance by Vehicle Class")
        perf.to_csv(os.path.join(results_dir, f"{spec.name}_performance_table.csv"), index=False)

        # ---- aggregate summary plots (distributions, errors, identifiability, …) ----
        _make_summary_plots(res_df, spec, results_dir)

        # ---- statistics (pooled + by regime) ----
        if STATS_TESTS_AVAILABLE:
            res = _stats_tests.compare_params(res_df, spec.stat_params)
            if len(res["anova"]):
                idm.print_formatted_table_numeric(
                    res["anova"], f"{spec.pretty}: Welch ANOVA across vehicle classes",
                    float_cols=["F-value", "p-value"])
                res["anova"].to_csv(os.path.join(results_dir, f"{spec.name}_welch_anova.csv"), index=False)
            if len(res["games_howell"]):
                res["games_howell"].to_csv(
                    os.path.join(results_dir, f"{spec.name}_games_howell.csv"), index=False)

            kw = _kruskal_table(res_df, spec.stat_params)
            if len(kw):
                idm.print_formatted_table_numeric(
                    kw, f"{spec.pretty}: Kruskal-Wallis across vehicle classes",
                    float_cols=["H-value", "p-value"])
                kw.to_csv(os.path.join(results_dir, f"{spec.name}_kruskal_wallis.csv"), index=False)

            if "regime" in res_df.columns and res_df["regime"].nunique() > 0:
                try:
                    dist = (res_df[res_df["follower_type"].isin(["small", "large", "av"])]
                            .groupby(["follower_type", "regime"]).size().unstack(fill_value=0))
                    print("\n Regime distribution by class:\n" + dist.to_string())
                    dist.to_csv(os.path.join(results_dir, f"{spec.name}_regime_distribution.csv"))
                except Exception as e:
                    print(f" WARNING: regime distribution failed: {e}")
                rr = _stats_tests.compare_params_by_regime(res_df, spec.stat_params)
                if len(rr["anova"]):
                    idm.print_formatted_table_numeric(
                        rr["anova"], f"{spec.pretty}: Welch ANOVA by class WITHIN each regime",
                        float_cols=["F-value", "p-value"])
                    rr["anova"].to_csv(
                        os.path.join(results_dir, f"{spec.name}_welch_anova_by_regime.csv"), index=False)
                if len(rr["games_howell"]):
                    rr["games_howell"].to_csv(
                        os.path.join(results_dir, f"{spec.name}_games_howell_by_regime.csv"), index=False)

        # ---- summary ----
        summary = (res_df[res_df["follower_type"].isin(["small", "large", "av"])]
                   .groupby("follower_type")[spec.param_names + ["fitness", "rmse", "mae", "r_squared"]]
                   .agg(["mean", "std", "count"]))
        summary.to_csv(os.path.join(results_dir, f"{spec.name}_calib_vehicle_type_summary.csv"))
        print("\n" + "=" * 70 + "\nCALIBRATION COMPLETE\n" + "=" * 70)
    finally:
        tee.close()
