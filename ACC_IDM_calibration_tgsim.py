"""
ACC-IDM (Improved IDM with Constant-Acceleration Heuristic) calibration on
TGSIM, closed-loop, using the shared cf_engine.

This is the ACC-oriented IDM of Kesting, Treiber & Helbing (2010), widely used as
a surrogate for adaptive-cruise-control / AV longitudinal control. It augments
the standard IDM with a constant-acceleration heuristic (CAH) that avoids the
over-reaction of IDM to large approaching rates, blended via a "coolness"
coefficient c.

  a_IDM : standard IDM acceleration (Eq. 1-2 of the paper)
  a_CAH : constant-acceleration heuristic using the leader acceleration a_lead
          (estimated here from the observed leader speed series):
      atil = min(a_lead, a)
      if v_lead*(v - v_lead) <= -2*s*atil:
          a_CAH = v^2 * atil / (v_lead^2 - 2*s*atil)
      else:
          a_CAH = atil - (v - v_lead)^2 * H(v - v_lead) / (2*s)
  a_ACC :
      if a_IDM >= a_CAH:  a = a_IDM
      else:               a = (1-c)*a_IDM + c*( a_CAH + b*tanh((a_IDM - a_CAH)/b) )

Parameters: T, a, b, v0, s0 (as in IDM; delta fixed at 4) plus c (coolness,
0<c<1). Run:  python ACC_IDM_calibration_tgsim.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import IDM_calibration_tgsim as idm  # noqa: E402
import cf_engine as eng  # noqa: E402

ACC_MIN, ACC_MAX = idm.ACC_MIN, idm.ACC_MAX
DELTA = 4.0  # fixed (non-identifiable from short trajectories; see paper)


def _leader_accel(t, v_lead):
    """Estimate leader acceleration from the observed leader speed series."""
    n = len(t)
    a = np.zeros(n)
    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        if dt > 1e-6:
            a[i] = (v_lead[i + 1] - v_lead[i]) / dt
    if n >= 2:
        a[-1] = a[-2]
    return np.clip(a, ACC_MIN, ACC_MAX)


def simulate_acc_idm(t, x_lead, v_lead, x0, v0, l_eff, theta, aux=None):
    """Closed-loop ACC-IDM integration. theta = [T, a, b, v0, s0, c]."""
    T, a_max, b, v0p, s0, c = (float(theta[0]), float(theta[1]), float(theta[2]),
                               float(theta[3]), float(theta[4]), float(theta[5]))
    a_max = max(a_max, 1e-3)
    b = max(b, 1e-3)
    v0p = max(v0p, 1e-3)
    c = min(max(c, 0.0), 1.0)
    sqrt_ab = math.sqrt(max(1e-6, a_max * b))
    a_lead = _leader_accel(t, v_lead)
    dt_arr = np.clip(np.diff(t), idm.DT_MIN, idm.DT_MAX)
    n = len(t)
    x = np.zeros(n)
    v = np.zeros(n)
    x[0] = float(x0)
    v[0] = max(0.0, float(v0))
    for i in range(n - 1):
        vi = max(0.0, v[i])
        vl = max(0.0, v_lead[i])
        s = x_lead[i] - x[i] - l_eff
        s = s if s > 0.1 else 0.1
        dv = vi - vl  # closing rate (follower minus leader)

        # --- standard IDM ---
        s_star = s0 + max(0.0, vi * T + (vi * dv) / (2.0 * sqrt_ab))
        a_idm = a_max * (1.0 - (vi / v0p) ** DELTA - (s_star / s) ** 2)

        # --- constant-acceleration heuristic (CAH) ---
        atil = min(a_lead[i], a_max)
        if vl * (vi - vl) <= -2.0 * s * atil:
            denom = vl * vl - 2.0 * s * atil
            a_cah = (vi * vi * atil) / denom if abs(denom) > 1e-9 else a_idm
        else:
            heav = 1.0 if (vi - vl) > 0.0 else 0.0
            a_cah = atil - (dv * dv * heav) / (2.0 * s)

        # --- ACC blend ---
        if a_idm >= a_cah:
            a = a_idm
        else:
            a = (1.0 - c) * a_idm + c * (a_cah + b * math.tanh((a_idm - a_cah) / b))

        if a > ACC_MAX:
            a = ACC_MAX
        elif a < ACC_MIN:
            a = ACC_MIN
        v_next = vi + a * dt_arr[i]
        if v_next < 0.0:
            v_next = 0.0
        x[i + 1] = x[i] + v_next * dt_arr[i]
        v[i + 1] = v_next
    return x, v


SPEC = eng.ModelSpec(
    name="acc_idm",
    pretty="ACC-IDM",
    param_names=["T", "a", "b", "v0", "s0", "c"],
    bounds=np.array([
        (0.5, 2.5),     # T   safe time headway (s)
        (0.3, 5.0),     # a   max acceleration (m/s^2)
        (0.5, 3.0),     # b   comfortable deceleration (m/s^2)
        (5.0, 35.0),    # v0  desired speed (m/s)
        (1.0, 5.0),     # s0  minimum gap (m)
        (0.0, 1.0),     # c   coolness / CAH blend
    ], dtype=float),
    # c is an AV-control smoothing factor; compare the IDM-shared params + c.
    simulate=simulate_acc_idm,
)


if __name__ == "__main__":
    eng.run_calibration(SPEC)
