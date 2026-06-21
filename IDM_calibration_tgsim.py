"""
IDM Calibration on TGSIM using the paper's methodology (Beigi et al.):

- Extract valid car-following episodes:
  * duration > 10 s
  * headway < 200 m
  * no lane change during the episode
  * preceding vehicle ID remains unchanged (leader constant in the episode)
- Calibrate IDM parameters for EACH leader–follower episode via Genetic Algorithm (GA)
- Fitness = sum_j [ w_pos*|x_obs - x_sim| + w_speed*|v_obs - v_sim| ]

Paper basis: "A Data-Driven Comparison of Car-Following Behaviors..." (Beigi et al.)
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

# Try to import numba for JIT compilation (much faster)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Create a dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# -----------------------------
# Logging setup - Tee class to write to both console and file
# -----------------------------
class Tee:
    """Write to both console and file simultaneously"""
    def __init__(self, file_path: str):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, text: str):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()  # Ensure immediate write

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()

# -----------------------------
# CONFIG (edit these)
# -----------------------------
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "Dataset")

# Extended-analysis modules (regime classification + robust statistics).
# Optional: calibration still works if these are absent.
sys.path.insert(0, os.path.join(SCRIPT_DIR, "analysis"))
try:
    from regime import dominant_regime, RegimeConfig
    _REGIME_CFG = RegimeConfig()
    REGIME_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    REGIME_AVAILABLE = False
    print(f" [WARN] regime module unavailable ({_e}); regime labels will be 'unknown'.")
try:
    import stats_tests as _stats_tests
    STATS_TESTS_AVAILABLE = True
except Exception:  # pragma: no cover
    STATS_TESTS_AVAILABLE = False


def episode_regime(ep) -> str:
    """Dominant traffic regime of an episode from its follower speed profile."""
    if not REGIME_AVAILABLE:
        return "unknown"
    try:
        d = ep.df
        return dominant_regime(d["t"].to_numpy(dtype=float),
                               d["v_foll"].to_numpy(dtype=float), _REGIME_CFG)
    except Exception:
        return "unknown"

# Define all dataset paths
CSV_PATHS = [
    os.path.join(DATASET_DIR, r"Third_Generation_Simulation_Data__TGSIM__I-395_Trajectories.csv"),
    # Disabled for now (I-90/I-94 geometry is diagonal; longitudinal axis needs projection)
    # os.path.join(DATASET_DIR, r"Third_Generation_Simulation_Data__TGSIM__I-90_I-94_Stationary_Trajectories.csv"),
    os.path.join(DATASET_DIR, r"Third_Generation_Simulation_Data__TGSIM__I-294_L1_Trajectories.csv"),
    os.path.join(DATASET_DIR, r"Third_Generation_Simulation_Data__TGSIM__I-294_L2_Trajectories.csv"),
]

# Dataset names for tracking
DATASET_NAMES = [
    "I-395",
    # Disabled for now: "I-90_I-94",
    "I-294_L1",
    "I-294_L2",
]

# Calibration settings
CALIBRATE_ONLY_NEAR_AVS = True  # If True: only calibrate vehicles near AVs (equal sampling)
                                 # If False: calibrate ALL episodes (no filtering)

# Results directory: "Results IDM" for equal sampling, "Results Total IDM" for all episodes
RESULTS_DIR = os.path.join(SCRIPT_DIR, "Results IDM" if CALIBRATE_ONLY_NEAR_AVS else "Results Total IDM")

OUTPUT_EPISODES_CSV = os.path.join(RESULTS_DIR, "idm_calib_episodes_results.csv")
OUTPUT_SUMMARY_CSV = os.path.join(RESULTS_DIR, "idm_calib_vehicle_type_summary.csv")
OUTPUT_EPISODES_EXCEL = os.path.join(RESULTS_DIR, "idm_calib_episodes_summary.xlsx")
PLOT_COMPARISONS = True  # Per-episode observed vs simulated plots
PLOT_SUMMARY = True       # Aggregate plots after all episodes are calibrated (distributions, errors, …)
# None = plot every calibrated episode; set to an int to cap (e.g. 5 for quick runs)
MAX_PLOT_EPISODES = None

# Vehicle type mapping
# Type 1: small cars
# Type 2: trucks -> large
# Type 3: buses -> large
# Type 4: autonomous vehicles (AV)
VEHICLE_TYPE_MAP = {
    1: "small",
    2: "large",  # trucks
    3: "large",  # buses
    4: "av",     # autonomous vehicles
}

# Fitness weights
# These control the relative importance of position vs speed errors in calibration
# Equal weights (1.0, 1.0) means both are equally important
# Note: Position errors are typically in meters (often 0-10m), speed errors in m/s (often 0-5 m/s)
# If position errors dominate numerically, consider increasing W_SPEED or normalizing errors
# Common choices:
#   - Equal importance: W_POS = 1.0, W_SPEED = 1.0 (current)
#   - Emphasize speed: W_POS = 1.0, W_SPEED = 2.0 or 3.0
#   - Emphasize position: W_POS = 2.0, W_SPEED = 1.0
W_POS = 1.0
W_SPEED = 1

# Episode constraints from paper
MIN_EPISODE_DURATION_S = 10
MAX_HEADWAY_M = 200.0

# Integration settings
DT_MIN = 0.02
DT_MAX = 0.30

# GA settings (optimized for speed; increase if you want tighter calibration)
GA_POP = 50
GA_GENS = 80
GA_ELITE_FRAC = 0.15
GA_TOURN_K = 3
GA_CROSSOVER_PROB = 0.9
GA_MUTATION_PROB = 0.25
GA_MUTATION_SCALE = 0.15  # fraction of parameter range for gaussian mutation
GA_EARLY_STOP_GENS = 10
GA_EARLY_STOP_TOL = 1e-6

# Random seed for reproducibility (set to None for non-deterministic results)
RANDOM_SEED = 42  # Change this value to get different but reproducible results

# Multiple runs configuration for robust calibration
# If > 1: run calibration N times with different seeds and aggregate results
# If 1: single run (faster, but less robust)
N_CALIBRATION_RUNS = 20  # Recommended: 10-30 for robust results, 1 for speed
USE_BEST_RUN = True  # If True: use best run (lowest fitness). If False: use mean of all runs

# Parallel processing configuration
# Number of parallel workers for episode calibration (None = use all CPU cores, 1 = no parallelization)
N_PARALLEL_WORKERS = 12  # Set to None for auto (uses all cores), or specify number (e.g., 4)
# Note: Parallelization gives ~4-8x speedup on multi-core CPUs, but uses more memory
# since each episode calibration is independent and can run in parallel on CPU cores.

# Override from environment (used by run_calibration_sweep.py for W_POS/W_SPEED combos)
if "CALIB_W_POS" in os.environ:
    W_POS = float(os.environ["CALIB_W_POS"])
if "CALIB_W_SPEED" in os.environ:
    W_SPEED = float(os.environ["CALIB_W_SPEED"])
if "CALIB_OUTPUT_SUBFOLDER" in os.environ:
    RESULTS_DIR = os.path.join(SCRIPT_DIR, os.environ["CALIB_OUTPUT_SUBFOLDER"])
    OUTPUT_EPISODES_CSV = os.path.join(RESULTS_DIR, "idm_calib_episodes_results.csv")
    OUTPUT_SUMMARY_CSV = os.path.join(RESULTS_DIR, "idm_calib_vehicle_type_summary.csv")
    OUTPUT_EPISODES_EXCEL = os.path.join(RESULTS_DIR, "idm_calib_episodes_summary.xlsx")
# Quick verification mode (skip heavy calibration)
# - CLI:  python idm_calibration_tgsim_V2.py --stop-after-selection
# - Env:  set CALIB_STOP_AFTER_SELECTION=1
STOP_AFTER_SELECTION = "--stop-after-selection" in sys.argv
if not STOP_AFTER_SELECTION and "CALIB_STOP_AFTER_SELECTION" in os.environ:
    STOP_AFTER_SELECTION = str(os.environ["CALIB_STOP_AFTER_SELECTION"]).strip().lower() not in ("0", "false", "no", "")

# -----------------------------
# Column guessing helpers
# -----------------------------
def guess_column(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in cols}

    # exact matches
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    # boundary-ish partial matches
    for cand in candidates:
        cand_lower = cand.lower()
        for c in cols:
            c_lower = c.lower()
            if c_lower == cand_lower or c_lower.startswith(cand_lower + '_') or c_lower.startswith(cand_lower + '-'):
                return c

    # fallback substring
    for cand in candidates:
        cand_lower = cand.lower()
        for c in cols:
            if cand_lower in c.lower():
                return c
    return None


@dataclass
class Schema:
    time: str
    veh_id: str
    lead_id: Optional[str]  # optional; will be computed if missing
    lane: str
    speed: str
    pos: str
    veh_type: str
    run_index: Optional[str] = None  # optional; separates distinct data-collection runs
    av_column: Optional[str] = None  # optional; separate AV column
    length: Optional[str] = None     # optional
    # True = position increases in direction of travel (leader has larger pos); False = reversed axis
    pos_increases_downstream: bool = True


def infer_schema(df: pd.DataFrame, dataset_name: str = None) -> Schema:
    cols = list(df.columns)
    time = guess_column(cols, ["time", "t", "timestamp", "sec", "seconds"])
    veh_id = guess_column(cols, ["veh_id", "vehicle_id", "id", "track_id"])
    lead_id = guess_column(cols, ["preceding", "leader", "lead_id", "preceding_vehicle_id", "front_id"])
    lane = guess_column(cols, ["lane", "lane_id", "laneindex", "lane_kf"])
    speed = guess_column(cols, ["speed", "v", "vel", "velocity", "speed_mps", "speed_kf"])
    run_index = guess_column(cols, ["run_index", "runid", "run_id", "run", "collection_index", "collection_id"])

    xloc_col = guess_column(cols, ["xloc_kf", "x", "x_position", "xloc"])
    yloc_col = guess_column(cols, ["yloc_kf", "y", "y_position", "yloc"])

    if xloc_col and yloc_col:
        if dataset_name and ("I-90" in dataset_name or "I-94" in dataset_name or "I-294" in dataset_name):
            pos = xloc_col
            if "I-294" in dataset_name:
                print(f" Using X ({xloc_col}) as longitudinal direction (I-294 dataset)")
            else:
                print(f" Using X ({xloc_col}) as longitudinal direction (I-90/I-94 dataset)")
        elif dataset_name and "I-395" in dataset_name:
            pos = yloc_col
            print(f" Using Y ({yloc_col}) as longitudinal direction (I-395 dataset)")
        else:
            try:
                sample_df = df.sample(min(5000, len(df))) if len(df) > 5000 else df
                sample_df = sample_df.dropna(subset=[speed, xloc_col, yloc_col, time, veh_id])
                if len(sample_df) > 100:
                    # If run_index exists, ensure we don't mix trajectories from different runs.
                    group_cols = [veh_id]
                    sort_cols = [veh_id, time]
                    if run_index and run_index in sample_df.columns:
                        group_cols = [run_index, veh_id]
                        sort_cols = [run_index, veh_id, time]

                    sample_df = sample_df.sort_values(sort_cols)
                    sample_df['dt'] = sample_df.groupby(group_cols)[time].diff()
                    sample_df['dx'] = sample_df.groupby(group_cols)[xloc_col].diff().abs()
                    sample_df['dy'] = sample_df.groupby(group_cols)[yloc_col].diff().abs()
                    valid = (sample_df['dt'] > 0) & (sample_df['dt'] < 1.0)
                    if valid.sum() > 50:
                        dx_dt = (sample_df.loc[valid, 'dx'] / sample_df.loc[valid, 'dt']).replace([np.inf, -np.inf], np.nan)
                        dy_dt = (sample_df.loc[valid, 'dy'] / sample_df.loc[valid, 'dt']).replace([np.inf, -np.inf], np.nan)
                        speed_vals = sample_df.loc[valid, speed]
                        mask = ~(np.isnan(dx_dt) | np.isnan(dy_dt) | np.isnan(speed_vals))
                        if mask.sum() > 50:
                            dx_dt_clean = dx_dt[mask]
                            dy_dt_clean = dy_dt[mask]
                            speed_clean = speed_vals[mask]
                            x_avg = dx_dt_clean.mean()
                            y_avg = dy_dt_clean.mean()
                            speed_avg = speed_clean.mean()
                            x_diff = abs(x_avg - speed_avg) if x_avg > 0 else float('inf')
                            y_diff = abs(y_avg - speed_avg) if y_avg > 0 else float('inf')
                            if x_diff < y_diff:
                                pos = xloc_col
                                print(f" Detected: X ({xloc_col}) is the longitudinal direction")
                            else:
                                pos = yloc_col
                                print(f" Detected: Y ({yloc_col}) is the longitudinal direction")
                        else:
                            pos = yloc_col
                            print(f" Using Y ({yloc_col}) as longitudinal (default, insufficient data)")
                    else:
                        pos = yloc_col
                        print(f" Using Y ({yloc_col}) as longitudinal (default, insufficient data)")
                else:
                    pos = yloc_col
                    print(f" Using Y ({yloc_col}) as longitudinal (default, insufficient data)")
            except Exception as e:
                pos = yloc_col
                print(f" Using Y ({yloc_col}) as longitudinal (default, detection failed: {e})")
    elif yloc_col:
        pos = yloc_col
    elif xloc_col:
        pos = xloc_col
    else:
        pos = guess_column(cols, ["y", "y_position", "long", "longitudinal", "pos", "position", "x", "x_position"])

    veh_type = guess_column(cols, ["type", "vehicle_type", "veh_type", "class", "vehclass", "type_most_common"])

    # AV column: exact match first, with validation
    # Note: I-395 uses vehicle type=4 for AVs, not a separate AV column
    # Skip AV column detection for I-395 dataset
    av_column = None
    if dataset_name and "I-395" in dataset_name:
        # I-395 doesn't have a separate AV column - AVs are identified by type=4
        av_column = None
    else:
        cols_lower = {c.lower(): c for c in cols}
        av_strings = ['yes', 'no', 'true', 'false', '1', '0', 'y', 'n']

        def _try_av_column(candidate_col: str, min_ratio: float = 0.5) -> bool:
            """Return True if column looks like an AV flag (Yes/No etc.)."""
            try:
                sample_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df
                if candidate_col not in sample_df.columns:
                    return False
                sample_values = sample_df[candidate_col].dropna().astype(str).str.lower().str.strip()
                if len(sample_values) == 0:
                    return False
                av_count = sample_values.isin(av_strings).sum()
                return (av_count / len(sample_values)) >= min_ratio
            except Exception:
                return False

        # Try "av" and similar first
        av_candidates = ["av", "autonomous", "is_av", "av_flag"]
        for cand in av_candidates:
            if cand in cols_lower:
                candidate_col = cols_lower[cand]
                if _try_av_column(candidate_col, min_ratio=0.5):
                    av_column = candidate_col
                    break

        # Try "acc" (avoid matching acceleration_kf by exact name only)
        if av_column is None and "acc" in cols_lower:
            acc_col = cols_lower["acc"]
            if _try_av_column(acc_col, min_ratio=0.5):
                av_column = acc_col

        # Fallback: for I-90 / I-294, if column "av" or "acc" exists, use it even without validation
        # (TGSIM data often uses these column names for AV flag)
        if av_column is None and dataset_name:
            if "I-90" in dataset_name or "I-94" in dataset_name or "I-294" in dataset_name:
                if "av" in cols_lower:
                    av_column = cols_lower["av"]
                elif "acc" in cols_lower:
                    av_column = cols_lower["acc"]

        # Last resort: guess by name only
        if av_column is None:
            av_column = guess_column(cols, ["av", "autonomous", "is_av", "av_flag"])

    length = guess_column(cols, ["length", "veh_length", "vehicle_length", "length_smoothed"])

    # Infer longitudinal direction: does position increase in direction of travel?
    # Correlation of d(pos)/dt with speed: positive => pos increases downstream (leader has larger pos)
    pos_increases_downstream = True
    try:
        sample_df = df.sample(min(5000, len(df))) if len(df) > 5000 else df
        sample_df = sample_df.dropna(subset=[time, pos, speed, veh_id])
        if run_index and run_index in sample_df.columns:
            sample_df = sample_df.dropna(subset=[run_index])
        group_cols = [veh_id] if not (run_index and run_index in sample_df.columns) else [run_index, veh_id]
        sort_cols = [veh_id, time] if not (run_index and run_index in sample_df.columns) else [run_index, veh_id, time]
        sample_df = sample_df.sort_values(sort_cols)
        dpos = sample_df.groupby(group_cols)[pos].diff()
        dt = sample_df.groupby(group_cols)[time].diff()
        speed_vals = sample_df[speed].astype(float)
        valid = (dt.notna()) & (dt > 0) & (dt < 2.0) & (dpos.notna()) & (speed_vals.notna())
        if valid.sum() > 100:
            dpos_dt = (dpos / dt).replace([np.inf, -np.inf], np.nan)
            mask = valid & dpos_dt.notna()
            if mask.sum() > 100:
                corr = np.corrcoef(dpos_dt[mask].values.astype(float), speed_vals[mask].values.astype(float))[0, 1]
                if not np.isnan(corr):
                    pos_increases_downstream = corr >= 0
                    print(f" Longitudinal direction: pos {'increases' if pos_increases_downstream else 'decreases'} downstream (corr d(pos)/dt vs speed = {corr:.3f})")
    except Exception as e:
        print(f" Could not infer longitudinal direction, assuming pos increases downstream ({e})")

    missing = [k for k, v in {
        "time": time, "veh_id": veh_id, "lane": lane, "speed": speed, "pos": pos, "veh_type": veh_type
    }.items() if v is None]
    if missing:
        raise ValueError(
            f"Could not infer required columns: {missing}. "
            f"Please set them manually in infer_schema(). Found columns: {cols}"
        )

    return Schema(
        time=time,
        veh_id=veh_id,
        lead_id=lead_id,
        lane=lane,
        speed=speed,
        pos=pos,
        veh_type=veh_type,
        run_index=run_index,
        av_column=av_column,
        length=length,
        pos_increases_downstream=pos_increases_downstream
    )


def compute_leader_ids(df: pd.DataFrame, sc: Schema) -> pd.DataFrame:
    """
    Compute leader IDs by picking the next vehicle ahead in same lane for each timestamp.
    Assumes df contains multiple vehicles at same time and lane.
    """
    df = df.copy()
    print(" Preparing data for leader ID computation...")

    df[sc.time] = pd.to_numeric(df[sc.time], errors="coerce")
    df[sc.veh_id] = pd.to_numeric(df[sc.veh_id], errors="coerce")
    df[sc.lane] = pd.to_numeric(df[sc.lane], errors="coerce")
    df[sc.pos] = pd.to_numeric(df[sc.pos], errors="coerce")
    # Only drop rows missing essential trajectory columns (never run_index — optional for grouping)
    drop_cols = [sc.time, sc.veh_id, sc.lane, sc.pos]
    df = df.dropna(subset=drop_cols)
    # Coerce run_index for grouping; fill NaN so grouping still works
    if sc.run_index is not None and sc.run_index in df.columns:
        df[sc.run_index] = pd.to_numeric(df[sc.run_index], errors="coerce")
        df[sc.run_index] = df[sc.run_index].fillna(-1).astype("int64")

    print(" Computing leader IDs using optimized numpy operations...")
    # Sort so that "next row" is the vehicle ahead in the direction of travel.
    # pos_increases_downstream: ahead = larger pos; else ahead = smaller pos.
    sort_cols = [sc.time, sc.lane, sc.pos]
    if sc.run_index is not None and sc.run_index in df.columns:
        sort_cols = [sc.run_index] + sort_cols
    ascending = [True] * (len(sort_cols) - 1) + [sc.pos_increases_downstream]  # pos ascending iff pos_increases_downstream
    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    time_arr = df[sc.time].values
    lane_arr = df[sc.lane].values
    pos_arr = df[sc.pos].values
    veh_id_arr = df[sc.veh_id].values
    if sc.run_index is not None and sc.run_index in df.columns:
        run_arr = df[sc.run_index].values
    else:
        run_arr = None

    lead_id_arr = np.full(len(df), np.nan, dtype=float)

    time_changed = np.concatenate([[True], time_arr[1:] != time_arr[:-1]])
    lane_changed = np.concatenate([[True], lane_arr[1:] != lane_arr[:-1]])
    if run_arr is not None:
        run_changed = np.concatenate([[True], run_arr[1:] != run_arr[:-1]])
        group_boundary = time_changed | lane_changed | run_changed
    else:
        group_boundary = time_changed | lane_changed
    group_ids = np.cumsum(group_boundary)

    next_group = np.concatenate([group_ids[1:], [group_ids[-1] + 1]])
    next_pos = np.concatenate([pos_arr[1:], [np.nan]])
    next_veh_id = np.concatenate([veh_id_arr[1:], [np.nan]])

    same_group = (group_ids == next_group)
    # Vehicle ahead: larger pos if pos_increases_downstream, else smaller pos (we sorted so "next" row is ahead)
    ahead = (next_pos > pos_arr) if sc.pos_increases_downstream else (next_pos < pos_arr)
    valid = ~np.isnan(next_veh_id)

    mask = same_group & ahead & valid
    lead_id_arr[mask] = next_veh_id[mask]

    df["lead_id"] = lead_id_arr
    has_leader = (~np.isnan(lead_id_arr)).sum()
    print(f" Completed: {has_leader:,} vehicle-time records have identified leaders ({100*has_leader/len(df):.1f}%)")
    return df

# -----------------------------
# IDM model
# -----------------------------
@dataclass
class IDMParams:
    T: float
    a: float
    b: float
    v0: float
    s0: float
    delta: float

BOUNDS = {
    "T": (0.5, 2.5),
    "a": (0.3, 5.0),
    "b": (0.5, 3.0),
    "v0": (5.0, 35.0),
    "s0": (1.0, 5.0),
    # delta is fixed at the standard IDM value: it is not practically
    # identifiable from short trajectories (a near-uniform "calibrated"
    # distribution otherwise). Keeping it constant removes a spurious DOF.
    "delta": (4.0, 4.0),
}

# Acceleration hard bounds (match PT_CF_Calibration.ipynb / Talebpour defaults)
ACC_MAX = 5.0
ACC_MIN = -8.0

def idm_acc(v: float, s: float, dv: float, p: IDMParams) -> float:
    v = max(0.0, v)
    s = max(0.1, s)
    sqrt_ab = math.sqrt(max(1e-6, p.a * p.b))
    s_star = p.s0 + max(0.0, v * p.T + (v * dv) / (2.0 * sqrt_ab))
    term_free = (v / max(1e-6, p.v0)) ** p.delta
    term_int = (s_star / s) ** 2
    a_raw = p.a * (1.0 - term_free - term_int)
    return max(ACC_MIN, min(ACC_MAX, a_raw))

def idm_acc_vectorized(v: np.ndarray, s: np.ndarray, dv: np.ndarray, p: IDMParams) -> np.ndarray:
    v = np.maximum(0.0, v)
    s = np.maximum(0.1, s)
    sqrt_ab = np.sqrt(np.maximum(1e-6, p.a * p.b))
    s_star = p.s0 + np.maximum(0.0, v * p.T + (v * dv) / (2.0 * sqrt_ab))
    term_free = (v / np.maximum(1e-6, p.v0)) ** p.delta
    term_int = (s_star / s) ** 2
    a_raw = p.a * (1.0 - term_free - term_int)
    return np.clip(a_raw, ACC_MIN, ACC_MAX)

# NOTE:
# - cache=True can cause failures/corruption when disk is full or when many
#   multiprocessing workers compile/load the same cache concurrently (Windows).
# - We keep JIT on, but disable cache to avoid disk/cache race issues.
@jit(nopython=True, cache=False)
def _simulate_follower_numba(
    dt_arr: np.ndarray,
    x_lead: np.ndarray,
    v_lead: np.ndarray,
    l_eff: float,
    x0: float,
    v0: float,
    T: float,
    a: float,
    b: float,
    v0_param: float,
    s0: float,
    delta: float
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(dt_arr) + 1
    x = np.zeros(n)
    v = np.zeros(n)
    x[0] = x0
    v[0] = max(0.0, v0)

    sqrt_ab = np.sqrt(max(1e-6, a * b))
    v0_inv = 1.0 / max(1e-6, v0_param)

    for i in range(n - 1):
        dt = dt_arr[i]
        dv = v[i] - v_lead[i]
        # Closed-loop gap: recompute from the SIMULATED follower position so the
        # trajectory (and thus the calibration) is sensitive to the parameters.
        s = x_lead[i] - x[i] - l_eff
        v_clamped = max(0.0, v[i])
        s_clamped = max(0.1, s)

        s_star = s0 + max(0.0, v_clamped * T + (v_clamped * dv) / (2.0 * sqrt_ab))
        term_free = (v_clamped * v0_inv) ** delta
        term_int = (s_star / s_clamped) ** 2
        a_i = a * (1.0 - term_free - term_int)
        # Hard acceleration bounds (match PT_CF_Calibration.ipynb / Talebpour defaults)
        if a_i > ACC_MAX:
            a_i = ACC_MAX
        elif a_i < ACC_MIN:
            a_i = ACC_MIN

        v_next = max(0.0, v[i] + a_i * dt)
        x_next = x[i] + v_next * dt

        v[i + 1] = v_next
        x[i + 1] = x_next

    return x, v

def simulate_follower(
    t: np.ndarray,
    x_lead: np.ndarray,
    v_lead: np.ndarray,
    x0: float,
    v0: float,
    p: IDMParams,
    *,
    lead_length: float = 0.0,
    follower_length: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    global NUMBA_AVAILABLE
    dt_arr = np.diff(t)
    dt_arr = np.clip(dt_arr, DT_MIN, DT_MAX)

    # Closed-loop simulation: gap is recomputed each step from the SIMULATED
    # follower position using the same bumper-to-bumper convention as build_episodes
    # and PT (gap = x_lead - x_foll - lead_len/2 - foll_len/2).
    l_eff = 0.5 * float(lead_length) + 0.5 * float(follower_length)

    if NUMBA_AVAILABLE:
        # Be robust: if Numba compilation/import fails in a worker (common with
        # broken SciPy/BLAS installs or corrupted caches), fall back to pure Python.
        try:
            x, v = _simulate_follower_numba(
                dt_arr, x_lead, v_lead, l_eff, x0, v0, p.T, p.a, p.b, p.v0, p.s0, p.delta
            )
            return x, v
        except Exception as e:
            NUMBA_AVAILABLE = False
            # Keep output ASCII-only (Windows console encodings can vary)
            print(f" [WARN] Numba JIT failed ({type(e).__name__}: {e}). Falling back to pure Python simulation.")

    n = len(t)
    x = np.zeros(n, dtype=float)
    v = np.zeros(n, dtype=float)
    x[0] = x0
    v[0] = max(0.0, v0)

    sqrt_ab = np.sqrt(max(1e-6, p.a * p.b))
    v0_inv = 1.0 / max(1e-6, p.v0)

    for i in range(n - 1):
        dt = dt_arr[i]
        dv = v[i] - v_lead[i]
        # Closed-loop gap from the SIMULATED follower position (see numba note).
        s = x_lead[i] - x[i] - l_eff

        v_clamped = max(0.0, v[i])
        s_clamped = max(0.1, s)
        s_star = p.s0 + max(0.0, v_clamped * p.T + (v_clamped * dv) / (2.0 * sqrt_ab))
        term_free = (v_clamped * v0_inv) ** p.delta
        term_int = (s_star / s_clamped) ** 2
        a_i = p.a * (1.0 - term_free - term_int)
        # Hard acceleration bounds (match PT_CF_Calibration.ipynb / Talebpour defaults)
        a_i = max(ACC_MIN, min(ACC_MAX, a_i))

        v_next = max(0.0, v[i] + a_i * dt)
        x_next = x[i] + v_next * dt

        v[i + 1] = v_next
        x[i + 1] = x_next

    return x, v

# -----------------------------
# Episode extraction
# -----------------------------
@dataclass
class Episode:
    run_index: Optional[int]
    follower_id: int
    leader_id: int
    follower_type: str
    start_t: float
    end_t: float
    df: pd.DataFrame

def build_episodes(df: pd.DataFrame, sc: Schema) -> List[Episode]:
    df = df.copy()

    # leader IDs
    if sc.lead_id is None:
        print(" Leader ID column not found - computing from trajectory data...")
        df = compute_leader_ids(df, sc)
        sc.lead_id = "lead_id"
    else:
        print(" Using existing leader ID column from dataset")

    print(" Converting data types and filtering...")
    df[sc.time] = pd.to_numeric(df[sc.time], errors="coerce")
    df[sc.veh_id] = pd.to_numeric(df[sc.veh_id], errors="coerce").astype("Int64")
    df[sc.lead_id] = pd.to_numeric(df[sc.lead_id], errors="coerce").astype("Int64")
    df[sc.lane] = pd.to_numeric(df[sc.lane], errors="coerce").astype("Int64")
    df[sc.speed] = pd.to_numeric(df[sc.speed], errors="coerce")
    df[sc.pos] = pd.to_numeric(df[sc.pos], errors="coerce")
    # run_index: coerce for grouping, fill NaN with sentinel (do NOT require in dropna)
    if sc.run_index is not None and sc.run_index in df.columns:
        df[sc.run_index] = pd.to_numeric(df[sc.run_index], errors="coerce")
        df[sc.run_index] = df[sc.run_index].fillna(-1).astype("int64")
    # veh_type and av_column: never coerce to numeric here; keep as-is for episode labeling

    # Only require essential trajectory columns (never run_index, veh_type, av_column)
    required = [sc.time, sc.veh_id, sc.lead_id, sc.lane, sc.speed, sc.pos]
    if sc.length and sc.length in df.columns:
        required.append(sc.length)
    initial_rows = len(df)
    df = df.dropna(subset=required)
    print(f" After removing missing values: {len(df):,} rows (removed {initial_rows - len(df):,})")
    sort_cols = [sc.veh_id, sc.time]
    if sc.run_index is not None and sc.run_index in df.columns:
        sort_cols = [sc.run_index] + sort_cols
    df = df.sort_values(sort_cols)

    # Build leader dataframe with position, speed, and length if available
    leader_cols = [sc.time, sc.veh_id, sc.pos, sc.speed]
    if sc.run_index is not None and sc.run_index in df.columns:
        leader_cols.append(sc.run_index)
    if sc.length:
        leader_cols.append(sc.length)
    
    leaders = df[leader_cols].rename(columns={
        sc.time: "t",
        sc.veh_id: "leader_id",
        sc.pos: "x_lead",
        sc.speed: "v_lead",
    })
    if sc.run_index is not None and sc.run_index in leaders.columns:
        leaders = leaders.rename(columns={sc.run_index: "run_index"})
    # Rename length column if it exists
    if sc.length:
        leaders = leaders.rename(columns={sc.length: "lead_length"})
    else:
        leaders["lead_length"] = 0.0  # Default to 0 if length not available

    follower_cols = {
        sc.veh_id: "follower_id",
        sc.lead_id: "leader_id",
        sc.pos: "x_foll",
        sc.speed: "v_foll",
        sc.lane: "lane_id",
        sc.veh_type: "veh_type_code",
        sc.time: "t",
    }
    if sc.run_index is not None and sc.run_index in df.columns:
        follower_cols[sc.run_index] = "run_index"

    if sc.av_column and sc.av_column in df.columns:
        follower_cols[sc.av_column] = sc.av_column
    
    # Add length column if available (for follower)
    if sc.length:
        follower_cols[sc.length] = sc.length

    follower = df.rename(columns=follower_cols)
    
    # Rename follower length column if it exists
    if sc.length and sc.length in follower.columns:
        follower = follower.rename(columns={sc.length: "follower_length"})
    else:
        follower["follower_length"] = 0.0  # Default to 0 if length not available

    print(" Merging follower and leader trajectories...")
    merge_keys = ["t", "leader_id"]
    if "run_index" in follower.columns and "run_index" in leaders.columns:
        merge_keys = ["run_index"] + merge_keys
    merged = follower.merge(leaders, on=merge_keys, how="inner")
    print(f" Merged data: {len(merged):,} follower-leader pairs")

    # CRITICAL FIX: Compute bumper-to-bumper gap (consistent with fitness/metrics)
    # Ensure lead_length and follower_length are numeric
    if "lead_length" not in merged.columns:
        merged["lead_length"] = 0.0
    merged["lead_length"] = pd.to_numeric(merged["lead_length"], errors="coerce").fillna(0.0)
    
    if "follower_length" not in merged.columns:
        merged["follower_length"] = 0.0
    merged["follower_length"] = pd.to_numeric(merged["follower_length"], errors="coerce").fillna(0.0)
    
    # Bumper-to-bumper gap (center-based positions):
    # gap = x_lead - x_foll - (L_lead/2) - (L_foll/2)
    # Leader rear bumper = x_lead - L_lead/2
    # Follower front bumper = x_foll + L_foll/2
    # Gap = (x_lead - L_lead/2) - (x_foll + L_foll/2) = x_lead - x_foll - L_lead/2 - L_foll/2
    merged["gap"] = merged["x_lead"] - merged["x_foll"] - merged["lead_length"]/2 - merged["follower_length"]/2

    before_gap_filter = len(merged)
    merged = merged[(merged["gap"] > 0.0) & (merged["gap"] < MAX_HEADWAY_M)]
    print(f" After gap filter (0 < gap < {MAX_HEADWAY_M}m): {len(merged):,} pairs (removed {before_gap_filter - len(merged):,})")

    print(" Extracting episodes with constraints:")
    print(f" - Duration > {MIN_EPISODE_DURATION_S}s")
    print(f" - No lane changes")
    print(f" - Constant leader ID")

    episodes: List[Episode] = []

    if "run_index" in merged.columns:
        unique_followers = merged[["run_index", "follower_id"]].drop_duplicates().to_records(index=False)
    else:
        unique_followers = merged["follower_id"].unique()
    total_followers = len(unique_followers)

    # Avoid pandas FutureWarning: groupby(list_of_len_1) will yield tuple keys in future versions.
    group_keys: str | List[str]
    if "run_index" in merged.columns:
        group_keys = ["run_index", "follower_id"]
    else:
        group_keys = "follower_id"

    for idx_f, (key, g) in enumerate(merged.groupby(group_keys, sort=False), 1):
        g = g.sort_values("t").reset_index(drop=True)
        if isinstance(key, tuple):
            run_i, fid = key
            run_i = int(run_i) if pd.notna(run_i) else None
        else:
            run_i, fid = None, key

        leader_change = g["leader_id"].ne(g["leader_id"].shift(1))
        lane_change = g["lane_id"].ne(g["lane_id"].shift(1))
        dt = g["t"].diff()
        time_break = (dt.isna()) | (dt > DT_MAX * 2.5)

        cut = leader_change | lane_change | time_break
        seg_id = cut.cumsum()

        for _, seg in g.groupby(seg_id):
            if len(seg) < 5:
                continue

            dur = float(seg["t"].iloc[-1] - seg["t"].iloc[0])
            if dur <= MIN_EPISODE_DURATION_S:
                continue

            if seg["leader_id"].nunique() != 1:
                continue
            if seg["lane_id"].nunique() != 1:
                continue

            vt_raw = seg["veh_type_code"].iloc[0]

            # detect AV using separate column if exists (acc/av)
            is_av = False
            if sc.av_column and sc.av_column in seg.columns:
                try:
                    av_value = str(seg[sc.av_column].iloc[0]).lower().strip()
                    # Yes/No-style and column-name-style (acc, av) both mean AV when value indicates yes
                    is_av = av_value in ['yes', 'true', '1', 'y', 'acc', 'av', 'autonomous']
                except Exception:
                    is_av = False

            if is_av:
                follower_type = "av"
            elif isinstance(vt_raw, str) or pd.isna(vt_raw):
                if pd.isna(vt_raw):
                    follower_type = "unknown"
                else:
                    vt_lower = str(vt_raw).lower()
                    if 'small' in vt_lower:
                        follower_type = "small"
                    elif 'large' in vt_lower:
                        follower_type = "large"
                    else:
                        follower_type = "unknown"
            else:
                try:
                    vt = int(vt_raw)
                    follower_type = VEHICLE_TYPE_MAP.get(vt, "unknown")
                except Exception:
                    follower_type = "unknown"

            episodes.append(Episode(
                run_index=run_i,
                follower_id=int(fid),
                leader_id=int(seg["leader_id"].iloc[0]),
                follower_type=follower_type,
                start_t=float(seg["t"].iloc[0]),
                end_t=float(seg["t"].iloc[-1]),
                df=seg.copy()
            ))

        if idx_f % max(1, total_followers // 10) == 0 or idx_f == total_followers:
            print(f" Processed {idx_f}/{total_followers} followers, found {len(episodes)} episodes so far...")

    print(f" Episode extraction complete: {len(episodes):,} valid episodes found")
    return episodes

# -----------------------------
# Genetic Algorithm
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def random_params() -> IDMParams:
    return IDMParams(
        T=random.uniform(*BOUNDS["T"]),
        a=random.uniform(*BOUNDS["a"]),
        b=random.uniform(*BOUNDS["b"]),
        v0=random.uniform(*BOUNDS["v0"]),
        s0=random.uniform(*BOUNDS["s0"]),
        delta=random.uniform(*BOUNDS["delta"]),
    )

def params_to_vec(p: IDMParams) -> np.ndarray:
    return np.array([p.T, p.a, p.b, p.v0, p.s0, p.delta], dtype=float)

def vec_to_params(v: np.ndarray) -> IDMParams:
    return IDMParams(T=float(v[0]), a=float(v[1]), b=float(v[2]), v0=float(v[3]), s0=float(v[4]), delta=float(v[5]))

VEC_BOUNDS = np.array([BOUNDS["T"], BOUNDS["a"], BOUNDS["b"], BOUNDS["v0"], BOUNDS["s0"], BOUNDS["delta"]], dtype=float)

def fitness_episode(ep: Episode, p: IDMParams) -> float:
    d = ep.df
    t = d["t"].to_numpy(dtype=float)
    x_lead = d["x_lead"].to_numpy(dtype=float)
    v_lead = d["v_lead"].to_numpy(dtype=float)
    x_obs = d["x_foll"].to_numpy(dtype=float)
    v_obs = d["v_foll"].to_numpy(dtype=float)
    lead_length = float(d["lead_length"].iloc[0]) if "lead_length" in d.columns else 0.0
    follower_length = float(d["follower_length"].iloc[0]) if "follower_length" in d.columns else 0.0

    x_sim, v_sim = simulate_follower(
        t=t,
        x_lead=x_lead,
        v_lead=v_lead,
        x0=float(x_obs[0]),
        v0=float(v_obs[0]),
        p=p,
        lead_length=lead_length,
        follower_length=follower_length,
    )

    err = W_POS * np.abs(x_obs - x_sim) + W_SPEED * np.abs(v_obs - v_sim)
    return float(np.sum(err))

# -----------------------------
# Shared-engine adapters
# -----------------------------
# The genetic algorithm, multiprocessing pipeline, statistics and output
# generation now live in the model-agnostic cf_engine so that IDM, PT, OVRV,
# Gipps and ACC-IDM all share ONE calibration implementation. IDM keeps delta
# fixed at 4 (non-identifiable from short trajectories), so only five parameters
# are calibrated; simulate_idm reuses the numba kernel via simulate_follower.
IDM_ENGINE_PARAMS = ["T", "a", "b", "v0", "s0"]
IDM_DELTA_FIXED = 4.0


def _theta_to_idmparams(theta) -> IDMParams:
    return IDMParams(T=float(theta[0]), a=float(theta[1]), b=float(theta[2]),
                     v0=float(theta[3]), s0=float(theta[4]), delta=IDM_DELTA_FIXED)


def simulate_idm(t, x_lead, v_lead, x0, v0, l_eff, theta, aux=None):
    """Engine-compatible closed-loop IDM simulator (reuses the numba kernel)."""
    p = _theta_to_idmparams(theta)
    # simulate_follower computes l_eff = 0.5*lead + 0.5*foll; pass lead=2*l_eff.
    return simulate_follower(t, x_lead, v_lead, x0, v0, p,
                             lead_length=2.0 * float(l_eff), follower_length=0.0)


def _plot_adapter(ep, theta, aux, output_path):
    plot_episode_comparison(ep, _theta_to_idmparams(theta), output_path)


_IDM_SPEC = None


def idm_spec():
    """Lazily build the IDM ModelSpec (import cf_engine here to avoid a cycle)."""
    global _IDM_SPEC
    if _IDM_SPEC is None:
        import cf_engine as eng
        _IDM_SPEC = eng.ModelSpec(
            name="idm", pretty="IDM",
            param_names=IDM_ENGINE_PARAMS,
            bounds=np.array([BOUNDS["T"], BOUNDS["a"], BOUNDS["b"],
                             BOUNDS["v0"], BOUNDS["s0"]], dtype=float),
            simulate=simulate_idm,
            plot=_plot_adapter,
        )
    return _IDM_SPEC

def calibrate_episode_ga(ep: Episode, show_progress: bool = True) -> Tuple[IDMParams, float]:
    """Calibrate one episode via the shared engine GA (returns IDMParams)."""
    import cf_engine as eng
    theta, fit = eng.calibrate_episode_ga(ep, idm_spec(), W_POS, W_SPEED)
    return _theta_to_idmparams(theta), float(fit)

def calibrate_episode_robust(ep: Episode, n_runs: int = 1, use_best: bool = True, base_seed: int = None) -> Tuple[IDMParams, float, Dict]:
    """Multi-start calibration via the shared engine (returns IDMParams + stats)."""
    import cf_engine as eng
    theta, fit, st = eng.calibrate_episode_robust(
        ep, idm_spec(), W_POS, W_SPEED,
        n_runs=n_runs, use_best=use_best, base_seed=base_seed)
    std = st.get("std", np.zeros(len(IDM_ENGINE_PARAMS)))
    stats = {
        "n_runs": st.get("n_runs", n_runs),
        "mean_fitness": st.get("mean_fitness", fit),
        "std_fitness": st.get("std_fitness", 0.0),
        "std_params": {name: float(std[i]) for i, name in enumerate(IDM_ENGINE_PARAMS)},
    }
    return _theta_to_idmparams(theta), float(fit), stats

# -----------------------------
# Visualization / metrics
# -----------------------------
def calculate_performance_metrics(ep: Episode, params: IDMParams) -> Dict[str, float]:
    d = ep.df
    t = d["t"].to_numpy(dtype=float)
    x_lead = d["x_lead"].to_numpy(dtype=float)
    v_lead = d["v_lead"].to_numpy(dtype=float)
    x_obs = d["x_foll"].to_numpy(dtype=float)
    v_obs = d["v_foll"].to_numpy(dtype=float)
    lead_length = float(d["lead_length"].iloc[0]) if "lead_length" in d.columns else 0.0
    follower_length = float(d["follower_length"].iloc[0]) if "follower_length" in d.columns else 0.0

    x_sim, v_sim = simulate_follower(
        t=t,
        x_lead=x_lead,
        v_lead=v_lead,
        x0=float(x_obs[0]),
        v0=float(v_obs[0]),
        p=params,
        lead_length=lead_length,
        follower_length=follower_length,
    )

    pos_errors = x_obs - x_sim
    rmse = np.sqrt(np.mean(pos_errors ** 2))
    mae = np.mean(np.abs(pos_errors))

    ss_res = np.sum((x_obs - x_sim) ** 2)
    ss_tot = np.sum((x_obs - np.mean(x_obs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"rmse": float(rmse), "mae": float(mae), "r_squared": float(r_squared)}

def plot_episode_comparison(ep: Episode, params: IDMParams, output_path: str):
    if not MATPLOTLIB_AVAILABLE:
        print(f" Warning: matplotlib not available, skipping plot for episode {ep.follower_id}")
        return

    d = ep.df
    t = d["t"].to_numpy(dtype=float)
    x_obs = d["x_foll"].to_numpy(dtype=float)
    v_obs = d["v_foll"].to_numpy(dtype=float)
    x_lead = d["x_lead"].to_numpy(dtype=float)
    v_lead = d["v_lead"].to_numpy(dtype=float)
    gap = d["gap"].to_numpy(dtype=float)
    
    # Get vehicle lengths for gap calculation
    lead_length = float(d["lead_length"].iloc[0]) if "lead_length" in d.columns else 0.0
    follower_length = float(d["follower_length"].iloc[0]) if "follower_length" in d.columns else 0.0

    x_sim, v_sim = simulate_follower(
        t=t,
        x_lead=x_lead,
        v_lead=v_lead,
        x0=float(x_obs[0]),
        v0=float(v_obs[0]),
        p=params,
        lead_length=lead_length,
        follower_length=follower_length,
    )
    
    # Calculate simulated gap (center-based positions)
    gap_sim = x_lead - x_sim - lead_length/2 - follower_length/2

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    type_label = ep.follower_type.upper() if ep.follower_type != "av" else "AV"
    fig.suptitle(
        f"Episode: Follower {ep.follower_id} following Leader {ep.leader_id}\n"
        f"Vehicle Type: {type_label}, Duration: {ep.end_t - ep.start_t:.1f}s\n"
        f"IDM Params: T={params.T:.2f}, a={params.a:.2f}, b={params.b:.2f}, "
        f"v0={params.v0:.2f}, s0={params.s0:.2f}, δ={params.delta:.2f}",
        fontsize=11
    )

    axes[0].plot(t, x_obs, label="Observed", linewidth=2)
    axes[0].plot(t, x_sim, "--", label="Simulated (IDM)", linewidth=2)
    axes[0].plot(t, x_lead, ":", label="Leader (Observed)", linewidth=2, alpha=0.9)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Longitudinal Position")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, v_obs, label="Observed", linewidth=2)
    axes[1].plot(t, v_sim, "--", label="Simulated (IDM)", linewidth=2)
    axes[1].plot(t, v_lead, ":", label="Leader (Observed)", linewidth=2, alpha=0.9)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].set_title("Speed")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, gap, label="Observed Gap", linewidth=2)
    axes[2].plot(t, gap_sim, "--", label="Simulated Gap (IDM)", linewidth=2)
    axes[2].axhline(y=MAX_HEADWAY_M, linestyle=":", label=f"Max headway ({MAX_HEADWAY_M}m)", alpha=0.5)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Gap (m)")
    axes[2].set_title("Gap to Leader")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def print_formatted_table(df: pd.DataFrame, title: str):
    col_widths = {}
    for col in df.columns:
        col_widths[col] = max(len(str(col)), df[col].astype(str).str.len().max())
        col_widths[col] = max(col_widths[col], 12)

    total_width = sum(col_widths.values()) + (len(df.columns) - 1) * 3 + 4
    print(f"\n{'=' * total_width}")
    print(f"{title:^{total_width}}")
    print(f"{'=' * total_width}")

    header_parts = [f"{col:^{col_widths[col]}}" for col in df.columns]
    header = " | ".join(header_parts)
    print(f"| {header} |")
    print(f"{'-' * total_width}")

    for _, row in df.iterrows():
        row_parts = [f"{str(row[col]):^{col_widths[col]}}" for col in df.columns]
        row_str = " | ".join(row_parts)
        print(f"| {row_str} |")

    print(f"{'=' * total_width}\n")

def create_episodes_excel(res_df: pd.DataFrame, output_path: str):
    """
    Create an Excel file with episode summary including leader, follower, time, and gap information.
    """
    try:
        # Select and reorder columns for Excel output
        excel_columns = [
            "dataset",
            "follower_id",
            "leader_id",
            "follower_type",
            "start_t",
            "end_t",
            "duration_s",
            "min_gap",
            "max_gap",
        ]
        
        # Check which columns exist in the dataframe
        available_columns = [col for col in excel_columns if col in res_df.columns]
        excel_df = res_df[available_columns].copy()
        
        # Rename columns for better readability
        excel_df = excel_df.rename(columns={
            "dataset": "Dataset",
            "follower_id": "Follower ID",
            "leader_id": "Leader ID",
            "follower_type": "Vehicle Type",
            "start_t": "Start Time (s)",
            "end_t": "End Time (s)",
            "duration_s": "Duration (s)",
            "min_gap": "Min Gap (m)",
            "max_gap": "Max Gap (m)",
        })
        
        # Sort by dataset, then by follower_id, then by start_t
        if "Dataset" in excel_df.columns:
            excel_df = excel_df.sort_values(["Dataset", "Follower ID", "Start Time (s)"])
        
        # Write to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            excel_df.to_excel(writer, sheet_name='Episodes Summary', index=False)
            
            # Auto-adjust column widths
            from openpyxl.utils import get_column_letter
            worksheet = writer.sheets['Episodes Summary']
            for idx, col in enumerate(excel_df.columns, 1):
                max_length = max(
                    excel_df[col].astype(str).str.len().max(),
                    len(str(col))
                )
                # Set column width (add some padding, max 50)
                col_letter = get_column_letter(idx)
                worksheet.column_dimensions[col_letter].width = min(max_length + 2, 50)
        
        return True
    except ImportError:
        print(f" WARNING: openpyxl not available. Install with: pip install openpyxl")
        return False
    except Exception as e:
        print(f" WARNING: Could not create Excel file: {e}")
        return False

def print_formatted_table_numeric(df: pd.DataFrame, title: str, float_cols: List[str] = None):
    """
    Like your print_formatted_table, but formats selected float columns nicely.
    """
    df2 = df.copy()
    float_cols = float_cols or []
    for c in float_cols:
        if c in df2.columns:
            df2[c] = df2[c].apply(lambda x: f"{x:.6g}" if pd.notna(x) else "NaN")
    print_formatted_table(df2, title)


# -----------------------------
# Entry point: calibration runs through the shared cf_engine, so IDM uses
# the same GA / pipeline / statistics / outputs as PT, OVRV, Gipps and
# ACC-IDM. The model-specific pieces are simulate_idm + idm_spec() above.
# -----------------------------
if __name__ == "__main__":
    import cf_engine as eng
    eng.run_calibration(idm_spec())
