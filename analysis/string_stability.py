"""
Linear string-stability analysis for calibrated IDM parameters.

Method
------
For a car-following model  a_n = f(s_n, dv_n, v_n)  with
    s_n  = x_{n-1} - x_n      (gap),
    dv_n = v_n - v_{n-1}      (closing speed, follower minus leader),
we linearize around a uniform-flow equilibrium (v_e, s_e, dv=0) and obtain the
speed-to-speed transfer function between consecutive vehicles:

    H(w) = (f_s + i*w*f_dv) / (f_s - w^2 + i*w*(f_dv - f_v))

where
    f_s  = df/ds   (>0),
    f_v  = df/dv   (<0, partial wrt own speed at fixed gap),
    f_dv = df/d(dv).

The platoon is *string stable* iff |H(w)| <= 1 for all w > 0. We compute the
peak magnitude  G_max = max_w |H(w)|  (string stable iff G_max <= 1) and the
low-frequency stability coefficient

    lambda2 = f_v^2/2 - f_dv*f_v - f_s            (string stable iff >= 0)

which is the standard Treiber-Kesting criterion recovered from the w->0
expansion of |H(w)|^2. Partials are computed by central finite differences on
the exact (clamped) IDM acceleration so they match the calibration model.

References: Treiber & Kesting, *Traffic Flow Dynamics* (2013), Ch. 15;
Ward (2009). Wilson & Ward (2011).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from cf_models import IDMParams, idm_acc, equilibrium_gap


@dataclass
class StabilityResult:
    v_e: float
    s_e: float
    f_s: float
    f_v: float
    f_dv: float
    lambda2: float          # >=0 => string stable (low-freq criterion)
    g_max: float            # <=1 => string stable (full-spectrum criterion)
    string_stable: bool
    locally_stable: bool


def _partials(v_e: float, s_e: float, p: IDMParams,
              hs: float = 1e-3, hv: float = 1e-3) -> tuple[float, float, float]:
    """Central finite-difference partials of f(s, dv, v) at (s_e, dv=0, v_e)."""
    f_s = (idm_acc(v_e, s_e + hs, 0.0, p) - idm_acc(v_e, s_e - hs, 0.0, p)) / (2 * hs)
    f_v = (idm_acc(v_e + hv, s_e, 0.0, p) - idm_acc(v_e - hv, s_e, 0.0, p)) / (2 * hv)
    f_dv = (idm_acc(v_e, s_e, hv, p) - idm_acc(v_e, s_e, -hv, p)) / (2 * hv)
    return f_s, f_v, f_dv


def analyze_speed(p: IDMParams, v_e: float,
                  n_freq: int = 4000, w_max: float = 5.0) -> StabilityResult:
    """String-stability analysis at a single equilibrium speed v_e."""
    s_e = equilibrium_gap(v_e, p)
    if not np.isfinite(s_e):
        # Free-flow: vehicle at/above desired speed -> no car-following equilibrium
        # (decoupled, trivially stable). Use NaN so plots simply end here.
        return StabilityResult(v_e, s_e, 0.0, 0.0, 0.0, np.nan, np.nan, True, True)

    f_s, f_v, f_dv = _partials(v_e, s_e, p)

    # Local (platoon-independent) stability: f_dv + f_v < 0  and  f_s > 0.
    locally_stable = (f_s > 0) and ((f_dv + f_v) < 0)

    # Low-frequency string-stability coefficient (Treiber & Kesting, 2013).
    # With convention dv = v_follower - v_leader: stable iff lambda2 >= 0.
    lambda2 = 0.5 * f_v ** 2 + f_dv * f_v - f_s

    # Full-spectrum peak of |H(w)|, with
    #   H(w) = (f_s - i*w*f_dv) / ((f_s - w^2) - i*w*(f_dv + f_v)).
    w = np.linspace(1e-4, w_max, n_freq)
    num = np.abs(f_s - 1j * w * f_dv)
    den = np.abs((f_s - w ** 2) - 1j * w * (f_dv + f_v))
    mag = num / np.maximum(den, 1e-12)
    g_max = float(np.max(mag))

    string_stable = (g_max <= 1.0 + 1e-6) and locally_stable
    return StabilityResult(v_e, float(s_e), f_s, f_v, f_dv,
                           float(lambda2), g_max, string_stable, locally_stable)


def stability_curve(p: IDMParams, v_grid: np.ndarray) -> Dict[str, np.ndarray]:
    """Sweep equilibrium speed; return arrays for plotting G_max(v_e), lambda2(v_e)."""
    g = np.full(v_grid.shape, np.nan)
    lam = np.full(v_grid.shape, np.nan)
    stable = np.zeros(v_grid.shape, dtype=bool)
    for i, v_e in enumerate(v_grid):
        r = analyze_speed(p, float(v_e))
        g[i] = r.g_max
        lam[i] = r.lambda2
        stable[i] = r.string_stable
    return {"v": v_grid, "g_max": g, "lambda2": lam, "stable": stable}


def critical_speed_band(p: IDMParams, v_grid: np.ndarray) -> tuple[float | None, float | None]:
    """Return (v_lower, v_upper) bounding the string-UNSTABLE speed band, if any."""
    curve = stability_curve(p, v_grid)
    unstable = ~curve["stable"]
    if not unstable.any():
        return (None, None)
    vs = v_grid[unstable]
    return (float(vs.min()), float(vs.max()))
