"""
Traffic-regime classification for car-following episodes.

Splits/labels follower trajectories into three interpretable congestion regimes:

    free_flow      : high, smooth speed (uncongested cruising)
    synchronized   : moderate, sustained speed with limited oscillation
                     (congested but flowing -- "synchronized flow")
    stop_and_go     : recurring near-stops and strong speed oscillation

Motivation: pooling all regimes (as in the rejected version) averages away
behavioral differences. AV controllers and human drivers differ most in the
transient stop-and-go regime, so regime-stratified calibration exposes
differences masked in the pooled analysis.

The classifier is rule-based and interpretable; thresholds are configurable and
their sensitivity can be reported. It depends only on numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

REGIMES = ("free_flow", "synchronized", "stop_and_go")


@dataclass
class RegimeConfig:
    v_freeflow: float = 17.0     # m/s; mean speed at/above this can be free-flow (~61 km/h)
    sigma_smooth: float = 1.5    # m/s; speed std below this is "smooth"
    v_stop: float = 3.0          # m/s; speed below this counts as a near-stop
    stop_frac_thr: float = 0.05  # fraction of time near-stop -> stop&go
    sigma_oscillation: float = 2.0  # m/s; speed std above this is "oscillatory"
    range_frac_thr: float = 0.5  # (v_max-v_min) > range_frac_thr * v_mean -> oscillatory
    window_s: float = 12.0       # s; sliding window for per-instant labeling
    min_regime_s: float = 10.0   # s; minimum duration for a regime-homogeneous segment


def compute_features(t: np.ndarray, v: np.ndarray) -> Dict[str, float]:
    """Summary speed-profile features used for regime classification."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"v_mean": 0.0, "v_std": 0.0, "v_min": 0.0, "v_max": 0.0,
                "v_cv": 0.0, "stop_frac": 0.0, "v_range": 0.0}
    v_mean = float(np.mean(v))
    v_std = float(np.std(v))
    return {
        "v_mean": v_mean,
        "v_std": v_std,
        "v_min": float(np.min(v)),
        "v_max": float(np.max(v)),
        "v_cv": float(v_std / v_mean) if v_mean > 1e-6 else 0.0,
        "stop_frac": float(np.mean(v < 3.0)),
        "v_range": float(np.max(v) - np.min(v)),
    }


def classify(feat: Dict[str, float], cfg: RegimeConfig = RegimeConfig()) -> str:
    """Assign a single regime label from summary features."""
    v_mean = feat["v_mean"]
    v_std = feat["v_std"]
    v_min = feat["v_min"]
    v_range = feat["v_range"]
    stop_frac = float(np.mean(np.asarray([feat["stop_frac"]])))  # already a fraction

    # 1) Stop-and-go: recurring near-stops OR deep + oscillatory speed swings.
    oscillatory = (v_std >= cfg.sigma_oscillation) and (v_range >= cfg.range_frac_thr * max(v_mean, 1e-6))
    near_stops = (stop_frac > cfg.stop_frac_thr) or (v_min < cfg.v_stop)
    if near_stops and oscillatory:
        return "stop_and_go"
    if stop_frac > cfg.stop_frac_thr:
        return "stop_and_go"

    # 2) Free-flow: high, smooth speed.
    if v_mean >= cfg.v_freeflow and v_std <= cfg.sigma_smooth:
        return "free_flow"

    # 3) Otherwise synchronized (congested but flowing).
    return "synchronized"


def classify_series(t: np.ndarray, v: np.ndarray,
                    cfg: RegimeConfig = RegimeConfig()) -> np.ndarray:
    """Per-instant regime labels using a centered sliding window."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(t)
    labels = np.empty(n, dtype=object)
    if n == 0:
        return labels
    dt = np.median(np.diff(t)) if n > 1 else 0.1
    half = max(1, int((cfg.window_s / max(dt, 1e-6)) / 2))
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        labels[i] = classify(compute_features(t[lo:hi], v[lo:hi]), cfg)
    return labels


def segment_by_regime(t: np.ndarray, v: np.ndarray,
                      cfg: RegimeConfig = RegimeConfig()) -> List[Tuple[str, int, int]]:
    """
    Partition a trajectory into contiguous regime-homogeneous segments.

    Returns list of (regime, i_start, i_end_exclusive). Short segments (< min
    duration) are merged into the neighboring segment with the longer duration.
    """
    t = np.asarray(t, dtype=float)
    labels = classify_series(t, v, cfg)
    n = len(labels)
    if n == 0:
        return []

    # Initial contiguous runs.
    runs: List[List] = []
    start = 0
    for i in range(1, n):
        if labels[i] != labels[start]:
            runs.append([labels[start], start, i])
            start = i
    runs.append([labels[start], start, n])

    # Merge too-short runs into the longer neighbor.
    def dur(run):
        return t[min(run[2], n - 1)] - t[run[1]]

    changed = True
    while changed and len(runs) > 1:
        changed = False
        for k, run in enumerate(runs):
            if dur(run) < cfg.min_regime_s:
                if k == 0:
                    nb = k + 1
                elif k == len(runs) - 1:
                    nb = k - 1
                else:
                    nb = k - 1 if dur(runs[k - 1]) >= dur(runs[k + 1]) else k + 1
                runs[nb][1] = min(runs[nb][1], run[1])
                runs[nb][2] = max(runs[nb][2], run[2])
                runs.pop(k)
                changed = True
                break

    return [(r[0], r[1], r[2]) for r in runs]


def dominant_regime(t: np.ndarray, v: np.ndarray,
                    cfg: RegimeConfig = RegimeConfig()) -> str:
    """Single regime label for a whole episode (longest-duration segment)."""
    segs = segment_by_regime(t, v, cfg)
    if not segs:
        return classify(compute_features(t, v), cfg)
    best = max(segs, key=lambda s: t[min(s[2], len(t) - 1)] - t[s[1]])
    return best[0]


# ---------------------------------------------------------------------------
# Self-test with synthetic profiles (runs without TGSIM data).
# ---------------------------------------------------------------------------
def _selftest():
    t = np.arange(0, 60, 0.1)
    ff = np.full_like(t, 28.0) + np.random.normal(0, 0.3, t.size)
    sync = np.full_like(t, 10.0) + np.random.normal(0, 0.8, t.size)
    sng = 8.0 + 7.0 * np.sin(2 * np.pi * t / 15.0)
    sng = np.clip(sng, 0.5, None)
    for name, v in [("free_flow", ff), ("synchronized", sync), ("stop_and_go", sng)]:
        dom = dominant_regime(t, v)
        feat = compute_features(t, v)
        print(f"{name:>14} -> classified '{dom}' "
              f"(v_mean={feat['v_mean']:.1f}, v_std={feat['v_std']:.1f}, "
              f"v_min={feat['v_min']:.1f}, stop_frac={feat['stop_frac']:.2f})")


if __name__ == "__main__":
    np.random.seed(0)
    _selftest()
