"""
Gipps (1981) safety-based car-following calibration on TGSIM, closed-loop,
using the shared cf_engine.

Gipps computes the next follower speed as the minimum of a free-acceleration
target and a safe (collision-avoiding) target:

  Free accel:
    v_acc = v + 2.5 * a * tau * (1 - v/V0) * sqrt(0.025 + v/V0)      (for v < V0)

  Safe speed (leader braking estimate b_hat = b, our derivation with gap g and
  jam/safety margin s0):
    v_safe = -b*tau + sqrt( max(0, b^2*tau^2 + b*(2*(g - s0) - v*tau) + v_lead^2) )

  v_target = max(0, min(v_acc, v_safe))

Because TGSIM is sampled finely (dt << tau), we convert the tau-ahead Gipps
target into an instantaneous acceleration a_eff = (v_target - v)/tau and
integrate forward at the data dt (closed loop: gap recomputed from simulated x).

Parameters: a (max accel, m/s^2), b (max decel magnitude, m/s^2), V0 (desired
speed, m/s), s0 (effective jam/safety gap, m), tau (reaction time, s).

Run:  python Gipps_calibration_tgsim.py
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


def simulate_gipps(t, x_lead, v_lead, x0, v0, l_eff, theta, aux=None):
    """Closed-loop Gipps integration. theta = [a, b, V0, s0, tau]."""
    a_max, b, V0, s0, tau = (float(theta[0]), float(theta[1]), float(theta[2]),
                             float(theta[3]), float(theta[4]))
    b = max(b, 1e-3)
    V0 = max(V0, 1e-3)
    tau = max(tau, 1e-2)
    dt_arr = np.clip(np.diff(t), idm.DT_MIN, idm.DT_MAX)
    n = len(t)
    x = np.zeros(n)
    v = np.zeros(n)
    x[0] = float(x0)
    v[0] = max(0.0, float(v0))
    for i in range(n - 1):
        vi = v[i]
        # Free-acceleration target (only meaningful below desired speed).
        ratio = vi / V0
        if ratio < 1.0:
            v_acc = vi + 2.5 * a_max * tau * (1.0 - ratio) * math.sqrt(max(0.025 + ratio, 0.0))
        else:
            v_acc = V0
        # Safe target from the braking constraint.
        g = x_lead[i] - x[i] - l_eff
        radicand = b * b * tau * tau + b * (2.0 * (g - s0) - vi * tau) + v_lead[i] * v_lead[i]
        v_safe = -b * tau + math.sqrt(radicand) if radicand > 0.0 else 0.0
        v_target = min(v_acc, v_safe)
        if v_target < 0.0:
            v_target = 0.0
        a_eff = (v_target - vi) / tau
        if a_eff > ACC_MAX:
            a_eff = ACC_MAX
        elif a_eff < ACC_MIN:
            a_eff = ACC_MIN
        v_next = vi + a_eff * dt_arr[i]
        if v_next < 0.0:
            v_next = 0.0
        x[i + 1] = x[i] + v_next * dt_arr[i]
        v[i + 1] = v_next
    return x, v


SPEC = eng.ModelSpec(
    name="gipps",
    pretty="Gipps",
    param_names=["a", "b", "V0", "s0", "tau"],
    bounds=np.array([
        (0.3, 5.0),     # a   max acceleration (m/s^2)
        (1.0, 5.0),     # b   max deceleration magnitude (m/s^2)
        (5.0, 40.0),    # V0  desired speed (m/s)
        (0.5, 7.0),     # s0  effective jam/safety gap (m)
        (0.3, 2.0),     # tau reaction time (s)
    ], dtype=float),
    simulate=simulate_gipps,
)


if __name__ == "__main__":
    eng.run_calibration(SPEC)
