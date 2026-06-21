"""
Mixed-autonomy platoon simulation + safety/throughput surrogates.

Given class-level calibrated IDM parameters (small HDV, large HDV, AV), simulate
a single-lane platoon at a chosen AV market-penetration rate (MPR), subject the
lead vehicle to a stop-and-go speed perturbation, and measure emergent outcomes:

  * String stability (empirical): amplification of the speed-oscillation
    amplitude from the lead vehicle to the last vehicle.
  * Safety surrogates per following vehicle:
        TTC  = s / dv          (only when closing, dv > 0)
        DRAC = dv^2 / (2 s)    (deceleration rate to avoid crash)
        TET  = time exposed to TTC below a threshold (s)
        TIT  = integral of (1/TTC - 1/TTC*) over exposed time (s)
  * Throughput at the downstream boundary (veh/h) and mean speed.

The vehicle composition is randomized over many seeds for each MPR so results
are reported as mean +/- std across realizations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from cf_models import IDMParams, idm_acc, equilibrium_gap, PAPER_IDM_MEANS

TTC_STAR = 3.0          # TTC safety threshold (s) for TET/TIT
DT = 0.1                # integration step (s)


@dataclass
class SimConfig:
    n_vehicles: int = 21          # 1 leader + 20 followers
    sim_time: float = 300.0       # s
    dt: float = DT
    v_set: float = 20.0           # base cruising speed (m/s)
    v_floor: float = 3.0          # speed at the bottom of each stop-and-go dip (m/s)
    perturb_period: float = 30.0  # s, period of one brake/recover cycle
    perturb_start: float = 30.0   # s
    n_perturb_cycles: int = 4
    veh_length: float = 5.0       # m (used for spacing/throughput)
    large_share_in_hdv: float = 0.30  # fraction of HDVs that are 'large'
    seed: int = 0


def _lead_speed_profile(cfg: SimConfig) -> np.ndarray:
    """Lead vehicle speed: cruise, then several deep stop-and-go brake/recover dips.

    Each cycle uses a raised-cosine dip from v_set down to v_floor and back,
    representative of congested stop-and-go waves that stress car-following safety.
    """
    n = int(cfg.sim_time / cfg.dt) + 1
    t = np.arange(n) * cfg.dt
    v = np.full(n, cfg.v_set, dtype=float)
    depth = cfg.v_set - cfg.v_floor
    end = cfg.perturb_start + cfg.n_perturb_cycles * cfg.perturb_period
    mask = (t >= cfg.perturb_start) & (t <= end)
    phase = 2 * np.pi * (t[mask] - cfg.perturb_start) / cfg.perturb_period
    v[mask] = cfg.v_set - depth * 0.5 * (1 - np.cos(phase))
    return np.clip(v, 0.0, None)


def assign_composition(cfg: SimConfig, mpr: float,
                       params: Dict[str, IDMParams]) -> List[IDMParams]:
    """Assign a class (and IDM params) to each follower given AV penetration mpr."""
    rng = np.random.default_rng(cfg.seed)
    n_foll = cfg.n_vehicles - 1
    types: List[IDMParams] = []
    for _ in range(n_foll):
        if rng.random() < mpr:
            types.append(params["av"])
        else:
            if rng.random() < cfg.large_share_in_hdv:
                types.append(params["large"])
            else:
                types.append(params["small"])
    return types


def simulate_platoon(cfg: SimConfig, mpr: float,
                     params: Dict[str, IDMParams]) -> Dict[str, np.ndarray]:
    """Forward-simulate the platoon. Returns time series for all vehicles."""
    v_lead = _lead_speed_profile(cfg)
    n_steps = len(v_lead)
    n_veh = cfg.n_vehicles
    foll_params = assign_composition(cfg, mpr, params)

    x = np.zeros((n_steps, n_veh))
    v = np.zeros((n_steps, n_veh))

    # Initialize at equilibrium spacing for the base speed.
    v[0, :] = cfg.v_set
    x[0, 0] = 0.0
    for j in range(1, n_veh):
        p = foll_params[j - 1]
        s_e = equilibrium_gap(cfg.v_set, p)
        if not np.isfinite(s_e):
            s_e = p.s0 + cfg.v_set * p.T
        x[0, j] = x[0, j - 1] - (s_e + cfg.veh_length)

    for i in range(n_steps - 1):
        # Leader (vehicle 0) follows the prescribed speed profile.
        v[i, 0] = v_lead[i]
        x[i + 1, 0] = x[i, 0] + v_lead[i] * cfg.dt
        v[i + 1, 0] = v_lead[i + 1] if i + 1 < n_steps else v_lead[i]

        for j in range(1, n_veh):
            p = foll_params[j - 1]
            gap = x[i, j - 1] - x[i, j] - cfg.veh_length
            dv = v[i, j] - v[i, j - 1]
            acc = idm_acc(v[i, j], gap, dv, p)
            v_next = max(0.0, v[i, j] + acc * cfg.dt)
            x[i + 1, j] = x[i, j] + v_next * cfg.dt
            v[i + 1, j] = v_next

    t = np.arange(n_steps) * cfg.dt
    return {"t": t, "x": x, "v": v, "params": foll_params}


def _safety_surrogates(sim: Dict[str, np.ndarray], cfg: SimConfig) -> Dict[str, float]:
    x, v = sim["x"], sim["v"]
    n_steps, n_veh = x.shape
    tet = 0.0
    tit = 0.0
    min_ttc = np.inf
    max_drac = 0.0
    for j in range(1, n_veh):
        gap = x[:, j - 1] - x[:, j] - cfg.veh_length
        gap = np.maximum(gap, 1e-3)
        dv = v[:, j] - v[:, j - 1]            # closing speed (>0 => approaching)
        closing = dv > 1e-3
        dv_safe = np.where(closing, dv, 1.0)  # avoid 0-division in inactive branch
        ttc = np.where(closing, gap / dv_safe, np.inf)
        drac = np.where(closing, dv_safe ** 2 / (2 * gap), 0.0)
        exposed = closing & (ttc < TTC_STAR)
        tet += np.sum(exposed) * cfg.dt
        tit += np.sum(np.maximum(0.0, 1.0 / np.maximum(ttc[exposed], 1e-6) - 1.0 / TTC_STAR)) * cfg.dt
        if np.any(closing):
            min_ttc = min(min_ttc, float(np.min(ttc[closing])))
        max_drac = max(max_drac, float(np.max(drac)))
    return {"TET": tet, "TIT": tit, "min_TTC": min_ttc, "max_DRAC": max_drac}


def _string_amplification(sim: Dict[str, np.ndarray]) -> float:
    """Ratio of last-vehicle speed-oscillation std to lead std (>1 => unstable)."""
    v = sim["v"]
    lead_std = float(np.std(v[:, 0]))
    last_std = float(np.std(v[:, -1]))
    if lead_std < 1e-6:
        return 1.0
    return last_std / lead_std


def _throughput(sim: Dict[str, np.ndarray], cfg: SimConfig) -> Tuple[float, float]:
    """Downstream throughput (veh/h) past the last vehicle's mean position & mean speed."""
    v = sim["v"]
    mean_speed = float(np.mean(v[:, 1:]))  # exclude prescribed leader
    # Headway-based capacity proxy: q = mean_speed / mean_spacing
    x = sim["x"]
    spacings = []
    for j in range(1, cfg.n_vehicles):
        sp = np.mean(x[:, j - 1] - x[:, j])
        spacings.append(sp)
    mean_spacing = float(np.mean(spacings))
    q = (mean_speed / mean_spacing) * 3600.0 if mean_spacing > 0 else 0.0
    return q, mean_speed


@dataclass
class MPRSweepResult:
    mpr: np.ndarray
    amp_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    amp_std: np.ndarray = field(default_factory=lambda: np.array([]))
    tet_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    tet_std: np.ndarray = field(default_factory=lambda: np.array([]))
    tit_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    min_ttc_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    drac_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    throughput_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    speed_mean: np.ndarray = field(default_factory=lambda: np.array([]))


def sweep_mpr(params: Dict[str, IDMParams],
              mprs: np.ndarray | None = None,
              n_realizations: int = 30,
              base_cfg: SimConfig | None = None) -> MPRSweepResult:
    """Sweep AV penetration, averaging emergent metrics over random compositions."""
    if mprs is None:
        mprs = np.linspace(0.0, 1.0, 11)
    base_cfg = base_cfg or SimConfig()

    amp_m, amp_s, tet_m, tet_s, tit_m = [], [], [], [], []
    ttc_m, drac_m, q_m, sp_m = [], [], [], []

    for mpr in mprs:
        amps, tets, tits, ttcs, dracs, qs, sps = [], [], [], [], [], [], []
        for r in range(n_realizations):
            cfg = SimConfig(**{**base_cfg.__dict__, "seed": r})
            sim = simulate_platoon(cfg, float(mpr), params)
            amps.append(_string_amplification(sim))
            saf = _safety_surrogates(sim, cfg)
            tets.append(saf["TET"]); tits.append(saf["TIT"])
            ttcs.append(saf["min_TTC"]); dracs.append(saf["max_DRAC"])
            q, sp = _throughput(sim, cfg)
            qs.append(q); sps.append(sp)
        amp_m.append(np.mean(amps)); amp_s.append(np.std(amps))
        tet_m.append(np.mean(tets)); tet_s.append(np.std(tets))
        tit_m.append(np.mean(tits))
        ttc_m.append(np.mean([t for t in ttcs if np.isfinite(t)] or [np.inf]))
        drac_m.append(np.mean(dracs)); q_m.append(np.mean(qs)); sp_m.append(np.mean(sps))

    return MPRSweepResult(
        mpr=np.asarray(mprs),
        amp_mean=np.asarray(amp_m), amp_std=np.asarray(amp_s),
        tet_mean=np.asarray(tet_m), tet_std=np.asarray(tet_s),
        tit_mean=np.asarray(tit_m),
        min_ttc_mean=np.asarray(ttc_m), drac_mean=np.asarray(drac_m),
        throughput_mean=np.asarray(q_m), speed_mean=np.asarray(sp_m),
    )
