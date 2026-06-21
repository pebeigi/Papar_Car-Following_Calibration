"""
Diagnostic: can the IDM calibrator recover KNOWN parameters?

We synthesize a follower trajectory from known IDM parameters using a proper
closed-loop simulation (gap recomputed from simulated position each step), then:

  1) run the EXISTING (open-loop) calibrator many times and look at the spread
     of recovered parameters, and
  2) compare the fitness landscape under open-loop vs closed-loop simulation.

If the existing calibrator returns a near-uniform spread across the bounds while
the true parameters are fixed, the calibration is not identifying parameters.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import IDM_calibration_tgsim as idm  # noqa: E402

L_EFF = 5.0  # effective vehicle length subtracted to get bumper-to-bumper gap
DT = 0.1


def make_synthetic_episode(p_true: idm.IDMParams, T_total: float = 90.0,
                           pos_noise: float = 0.0, speed_noise: float = 0.0,
                           seed: int = 0):
    """Closed-loop IDM follower behind an oscillating leader -> Episode.

    Optional Gaussian measurement noise on follower position/speed makes the
    test realistic (real trajectories are noisy and not perfectly IDM).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, T_total, DT)
    n = len(t)
    # Leader: oscillating speed (stop-and-go-like), integrated to position.
    v_lead = 14.0 + 6.0 * np.sin(2 * np.pi * t / 25.0)
    v_lead = np.clip(v_lead, 1.0, None)
    x_lead = np.zeros(n)
    x_lead[0] = 500.0
    for i in range(n - 1):
        x_lead[i + 1] = x_lead[i] + v_lead[i] * DT

    # Follower: closed-loop IDM (gap from simulated position).
    x_f = np.zeros(n)
    v_f = np.zeros(n)
    v_f[0] = v_lead[0]
    gap0 = (p_true.s0 + v_f[0] * p_true.T)
    x_f[0] = x_lead[0] - gap0 - L_EFF
    for i in range(n - 1):
        gap = x_lead[i] - x_f[i] - L_EFF
        dv = v_f[i] - v_lead[i]
        a = _idm_acc_true(v_f[i], gap, dv, p_true)
        v_f[i + 1] = max(0.0, v_f[i] + a * DT)
        x_f[i + 1] = x_f[i] + v_f[i + 1] * DT

    # Add measurement noise (observed = truth + noise).
    if pos_noise > 0:
        x_f = x_f + rng.normal(0, pos_noise, n)
    if speed_noise > 0:
        v_f = np.clip(v_f + rng.normal(0, speed_noise, n), 0.0, None)

    gap = x_lead - x_f - L_EFF
    df = pd.DataFrame({
        "t": t, "x_lead": x_lead, "v_lead": v_lead,
        "x_foll": x_f, "v_foll": v_f, "gap": gap,
        "lane_id": 1, "leader_id": 1, "follower_id": 2,
        "lead_length": L_EFF, "follower_length": L_EFF,
    })
    ep = idm.Episode(run_index=None, follower_id=2, leader_id=1,
                     follower_type="small", start_t=float(t[0]),
                     end_t=float(t[-1]), df=df)
    return ep


def _idm_acc_true(v, s, dv, p):
    v = max(0.0, v); s = max(0.1, s)
    sqrt_ab = np.sqrt(max(1e-6, p.a * p.b))
    s_star = p.s0 + max(0.0, v * p.T + (v * dv) / (2 * sqrt_ab))
    a_raw = p.a * (1 - (v / max(1e-6, p.v0)) ** p.delta - (s_star / s) ** 2)
    return float(min(idm.ACC_MAX, max(idm.ACC_MIN, a_raw)))


def closed_loop_sim(ep, p):
    """Closed-loop simulate (the CORRECT way) for fitness comparison."""
    d = ep.df
    t = d["t"].to_numpy(float)
    x_lead = d["x_lead"].to_numpy(float)
    v_lead = d["v_lead"].to_numpy(float)
    n = len(t)
    x = np.zeros(n); v = np.zeros(n)
    x[0] = float(d["x_foll"].iloc[0]); v[0] = float(d["v_foll"].iloc[0])
    for i in range(n - 1):
        dt = min(max(t[i + 1] - t[i], idm.DT_MIN), idm.DT_MAX)
        gap = x_lead[i] - x[i] - L_EFF
        dv = v[i] - v_lead[i]
        a = _idm_acc_true(v[i], gap, dv, p)
        v[i + 1] = max(0.0, v[i] + a * dt)
        x[i + 1] = x[i] + v[i + 1] * dt
    return x, v


def fitness_closed_loop(ep, p):
    d = ep.df
    x_obs = d["x_foll"].to_numpy(float); v_obs = d["v_foll"].to_numpy(float)
    x_sim, v_sim = closed_loop_sim(ep, p)
    return float(np.sum(idm.W_POS * np.abs(x_obs - x_sim) + idm.W_SPEED * np.abs(v_obs - v_sim)))


def landscape(ep, p_true, name, fit_fn):
    """Sweep each parameter around truth; report fitness range (sensitivity)."""
    print(f"\n  Fitness sensitivity ({name}):")
    base = {k: getattr(p_true, k) for k in ("T", "a", "b", "v0", "s0", "delta")}
    for k in ("T", "a", "b", "v0", "s0"):
        lo, hi = idm.BOUNDS[k]
        vals = np.linspace(lo, hi, 21)
        fits = []
        for val in vals:
            kw = dict(base); kw[k] = float(val)
            fits.append(fit_fn(ep, idm.IDMParams(**kw)))
        fits = np.array(fits)
        f_true = fit_fn(ep, p_true)
        rel = (fits.max() - fits.min()) / max(f_true, 1e-9)
        argmin_val = vals[int(np.argmin(fits))]
        print(f"    {k:>5}: fit range/true = {rel:6.2f}  (min at {k}={argmin_val:.3f}, "
              f"true={base[k]:.3f})")


def _recover(ep, fit_fn, n_restarts, base_seed):
    """Run the real GA n_restarts times using fit_fn as the objective."""
    import random
    orig = idm.fitness_episode
    idm.fitness_episode = fit_fn  # monkeypatch the objective used by the GA
    recs = []
    try:
        for s in range(n_restarts):
            random.seed(base_seed + s); np.random.seed(base_seed + s)
            p, _f = idm.calibrate_episode_ga(ep, show_progress=False)
            recs.append([p.T, p.a, p.b, p.v0, p.s0, p.delta])
    finally:
        idm.fitness_episode = orig
    return np.array(recs)


def _report(tag, recs, truth):
    cols = ["T", "a", "b", "v0", "s0", "delta"]
    print(f"\n  === {tag} ===")
    print(f"  {'param':>6} {'true':>8} {'rec.mean':>10} {'std':>8} {'std/range':>10}")
    for j, c in enumerate(cols):
        lo, hi = idm.BOUNDS[c]
        col = recs[:, j]
        print(f"  {c:>6} {truth[j]:>8.3f} {col.mean():>10.3f} {col.std():>8.3f} "
              f"{col.std() / (hi - lo):>10.2f}")


def main():
    p_true = idm.IDMParams(T=1.2, a=1.5, b=2.0, v0=22.0, s0=2.5, delta=4.0)
    truth = [p_true.T, p_true.a, p_true.b, p_true.v0, p_true.s0, p_true.delta]
    n_restarts = 6

    print("=" * 70)
    print("CALIBRATION DIAGNOSTIC — recover known IDM parameters")
    print("=" * 70)
    print(f" True params: T={p_true.T} a={p_true.a} b={p_true.b} "
          f"v0={p_true.v0} s0={p_true.s0} delta={p_true.delta}")
    print(" std/range ~0.29 => uniform across bound (NOT identified); ~0 => recovered.")

    for pos_noise, spd_noise, label in [(0.0, 0.0, "NOISE-FREE"),
                                        (0.30, 0.30, "REALISTIC NOISE (pos 0.3m, speed 0.3 m/s)")]:
        ep = make_synthetic_episode(p_true, T_total=60.0,
                                    pos_noise=pos_noise, speed_noise=spd_noise)
        print(f"\n{'#' * 66}\n# {label}: {len(ep.df)} steps", flush=True)
        recs_cur = _recover(ep, _ORIG_FIT, n_restarts, 1000)
        _report("CURRENT calibrator (idm.fitness_episode)", recs_cur, truth)


_ORIG_FIT = idm.fitness_episode


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
