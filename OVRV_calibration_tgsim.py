"""
OVRV (Optimal-Velocity Relative-Velocity / Full-Velocity-Difference) calibration
on TGSIM, closed-loop, using the shared cf_engine.

Model (Bando 1995 optimal-velocity term + Jiang 2001 relative-velocity term):

    a_n = kappa * (V(s_n) - v_n) + lam * (v_lead - v_n)

with the optimal-velocity function (normalized so V(0)=0, V(inf)=vmax):

    V(s) = vmax * (tanh((s - sc)/sw) + tanh(sc/sw)) / (1 + tanh(sc/sw))

Parameters: kappa (speed sensitivity, 1/s), lam (relative-velocity gain, 1/s),
vmax (free-flow speed, m/s), sc (inflection gap, m), sw (transition width, m).

This is a different modeling paradigm than IDM (gap-physics) and PT (behavioral
risk): a flow/optimal-velocity formulation. Run:  python OVRV_calibration_tgsim.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import IDM_calibration_tgsim as idm  # noqa: E402  (config + DT bounds + accel clamps)
import cf_engine as eng  # noqa: E402

ACC_MIN, ACC_MAX = idm.ACC_MIN, idm.ACC_MAX


def _opt_velocity(s: float, vmax: float, sc: float, sw: float) -> float:
    sw = max(sw, 1e-6)
    denom = 1.0 + math.tanh(sc / sw)
    return vmax * (math.tanh((s - sc) / sw) + math.tanh(sc / sw)) / max(denom, 1e-9)


def simulate_ovrv(t, x_lead, v_lead, x0, v0, l_eff, theta, aux=None):
    """Closed-loop OVRV integration. theta = [kappa, lam, vmax, sc, sw]."""
    kappa, lam, vmax, sc, sw = (float(theta[0]), float(theta[1]),
                                float(theta[2]), float(theta[3]), float(theta[4]))
    dt_arr = np.clip(np.diff(t), idm.DT_MIN, idm.DT_MAX)
    n = len(t)
    x = np.zeros(n)
    v = np.zeros(n)
    x[0] = float(x0)
    v[0] = max(0.0, float(v0))
    sw = max(sw, 1e-6)
    base = math.tanh(sc / sw)
    denom = 1.0 + base
    for i in range(n - 1):
        s = x_lead[i] - x[i] - l_eff
        s = s if s > 0.1 else 0.1
        Vs = vmax * (math.tanh((s - sc) / sw) + base) / denom
        a = kappa * (Vs - v[i]) + lam * (v_lead[i] - v[i])
        if a > ACC_MAX:
            a = ACC_MAX
        elif a < ACC_MIN:
            a = ACC_MIN
        v_next = v[i] + a * dt_arr[i]
        if v_next < 0.0:
            v_next = 0.0
        x[i + 1] = x[i] + v_next * dt_arr[i]
        v[i + 1] = v_next
    return x, v


SPEC = eng.ModelSpec(
    name="ovrv",
    pretty="OVRV",
    param_names=["kappa", "lam", "vmax", "sc", "sw"],
    bounds=np.array([
        (0.05, 2.0),    # kappa  (1/s)
        (0.0, 3.0),     # lam    (1/s)
        (5.0, 40.0),    # vmax   (m/s)
        (1.0, 40.0),    # sc     (m)
        (1.0, 40.0),    # sw     (m)
    ], dtype=float),
    simulate=simulate_ovrv,
)


if __name__ == "__main__":
    eng.run_calibration(SPEC)
