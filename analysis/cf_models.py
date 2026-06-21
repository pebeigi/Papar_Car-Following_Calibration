"""
Shared car-following model primitives for the extended (post-rejection) analysis.

This module is intentionally self-contained (depends only on numpy) so the
emergent-dynamics analysis can run *without* the raw TGSIM trajectories, using
the class-level calibrated parameters reported in the paper (Table 2). When the
raw data and per-episode calibration outputs are available, the same functions
are reused for regime-stratified and validation analyses.

The IDM acceleration here mirrors IDM_calibration_tgsim.py exactly (same
clamping and desired-gap formulation) so emergent results are consistent with
the calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np

# Acceleration hard bounds (match IDM_calibration_tgsim.py)
ACC_MAX = 5.0
ACC_MIN = -8.0


@dataclass(frozen=True)
class IDMParams:
    T: float       # safe time headway (s)
    a: float       # max acceleration (m/s^2)
    b: float       # comfortable deceleration (m/s^2)
    v0: float      # desired speed (m/s)
    s0: float      # minimum bumper-to-bumper gap (m)
    delta: float = 4.0  # acceleration exponent

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Class-level calibrated IDM parameters from the paper (Table 2, mean values).
# These let the emergent-dynamics analysis run immediately. They are overridden
# by data-driven (and regime-stratified) means once calibration CSVs exist.
# ---------------------------------------------------------------------------
PAPER_IDM_MEANS: Dict[str, IDMParams] = {
    "small": IDMParams(T=1.073, a=1.589, b=1.631, v0=28.085, s0=2.879, delta=4.064),
    "large": IDMParams(T=1.462, a=1.572, b=1.644, v0=25.015, s0=3.414, delta=4.027),
    "av":    IDMParams(T=1.059, a=1.261, b=1.906, v0=29.546, s0=3.425, delta=4.052),
}

# Paper Table 2 medians (more robust central tendency; reported for sensitivity).
PAPER_IDM_MEDIANS: Dict[str, IDMParams] = {
    "small": IDMParams(T=0.786, a=0.935, b=1.166, v0=34.860, s0=2.309, delta=4.199),
    "large": IDMParams(T=1.298, a=0.737, b=1.140, v0=26.632, s0=3.894, delta=4.110),
    "av":    IDMParams(T=0.785, a=0.692, b=2.245, v0=34.959, s0=4.243, delta=4.124),
}

CLASS_LABELS = {"small": "Small (HDV)", "large": "Large (HDV)", "av": "Autonomous"}


def idm_acc(v: float, s: float, dv: float, p: IDMParams) -> float:
    """IDM acceleration. dv = v_follower - v_leader (closing speed positive)."""
    v = max(0.0, v)
    s = max(0.1, s)
    sqrt_ab = np.sqrt(max(1e-6, p.a * p.b))
    s_star = p.s0 + max(0.0, v * p.T + (v * dv) / (2.0 * sqrt_ab))
    term_free = (v / max(1e-6, p.v0)) ** p.delta
    term_int = (s_star / s) ** 2
    a_raw = p.a * (1.0 - term_free - term_int)
    return float(min(ACC_MAX, max(ACC_MIN, a_raw)))


def idm_acc_vec(v: np.ndarray, s: np.ndarray, dv: np.ndarray, p: IDMParams) -> np.ndarray:
    v = np.maximum(0.0, v)
    s = np.maximum(0.1, s)
    sqrt_ab = np.sqrt(max(1e-6, p.a * p.b))
    s_star = p.s0 + np.maximum(0.0, v * p.T + (v * dv) / (2.0 * sqrt_ab))
    term_free = (v / max(1e-6, p.v0)) ** p.delta
    term_int = (s_star / s) ** 2
    a_raw = p.a * (1.0 - term_free - term_int)
    return np.clip(a_raw, ACC_MIN, ACC_MAX)


def equilibrium_gap(v_e: float, p: IDMParams) -> float:
    """
    Steady-state bumper-to-bumper gap for IDM at follower speed v_e with dv=0.

    From v_dot = 0, dv = 0:  s_e = (s0 + v_e*T) / sqrt(1 - (v_e/v0)^delta)
    Valid only for v_e < v0 (below desired speed). Returns np.inf at/above v0.
    """
    if v_e <= 0:
        return p.s0
    ratio = (v_e / p.v0) ** p.delta
    denom = 1.0 - ratio
    if denom <= 1e-9:
        return float("inf")
    s_star = p.s0 + v_e * p.T
    return s_star / np.sqrt(denom)


def equilibrium_speed_from_gap(s_e: float, p: IDMParams,
                               v_lo: float = 0.0, v_hi: float | None = None) -> float:
    """Invert equilibrium_gap: find v_e such that equilibrium_gap(v_e) = s_e."""
    if v_hi is None:
        v_hi = p.v0 - 1e-6
    lo, hi = v_lo, v_hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        g = equilibrium_gap(mid, p)
        if g > s_e:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
