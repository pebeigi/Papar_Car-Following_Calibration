<h1 align="center"> A Data-Driven Comparison of Car-Following Behavior Between Autonomous and Human-Driven Vehicles </h1>

<p align="center">
  <a href="#"><img alt="TRR Paper" src="https://img.shields.io/static/v1?label=TRR%20Paper&message=Under%20Review&color=purple&style=flat-square"></a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/pedrambeigi/Papar_Car-Following_Calibration/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/static/v1?label=License&message=MIT&color=blue&style=flat-square"></a>
</p>

---

## Paper & Repository

**Paper.** *A Data-Driven Comparison of Car-Following Behavior Between Autonomous and Human-Driven Vehicles* examines longitudinal car-following using **TGSIM trajectory data**. Two models are calibrated per follower–leader episode: **IDM** and a **Prospect Theory (PT)** formulation. Calibration quality is judged with trajectory errors (RMSE, MAE); parameter differences across **small vehicles, large vehicles, and AVs** are assessed with **Welch’s ANOVA** and **Games–Howell** post hoc tests.

**Repository.** This codebase implements that pipeline end-to-end: episode extraction from trajectory CSVs, **genetic-algorithm** calibration, optional multiprocessing / Numba, exports (CSV / Excel / plots), and the statistical comparisons above. Defaults target **balanced sampling near AV episodes** so classes are comparable.

## What This Repository Does

At a high level, the repository:

- extracts valid leader–follower car-following episodes from TGSIM trajectories,
- calibrates **Intelligent Driver Model (IDM)**<sup><a href="#note-repo-idm">1</a></sup> and **Prospect Theory (PT)**<sup><a href="#note-repo-pt">2</a></sup> parameters for each episode using a Genetic Algorithm (GA),
- evaluates fit quality with trajectory-level error metrics (RMSE, MAE, R-squared),
- aggregates parameter / performance summaries by vehicle type,
- runs statistical tests<sup><a href="#note-repo-stats">3</a></sup> across vehicle classes.

**Notes:**

<a id="note-repo-idm"></a>**[1] IDM** — parameters: `T, a, b, v0, s0, delta`.

<a id="note-repo-pt"></a>**[2] PT** — parameters: `Wm, Alpha, Beta, Wc, Tmax, Gamma`.

<a id="note-repo-stats"></a>**[3] Statistical tests** — include `Welch ANOVA, Games–Howell post hoc tests, Kruskal–Wallis`.

## Data

The analysis uses **Third Generation Simulation (TGSIM)** <a href="https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim?from_hint=eyJxIjoidGdzaW0ifQ%3D%3D"><img alt="Data — TGSIM" src="https://img.shields.io/static/v1?label=Data&message=TGSIM&color=blue&style=flat-square" style="vertical-align: middle;"></a> <a href="https://journals.sagepub.com/doi/10.1177/03611981241257257"><img alt="Paper — TGSIM" src="https://img.shields.io/static/v1?label=Paper&message=TGSIM%20TRR&color=purple&style=flat-square" style="vertical-align: middle;"></a>

TGSIM is public trajectory data from FHWA’s Third Generation Simulation project (e.g., I-395 DC, I-294 IL; see the [Data.gov catalog entry](https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim?from_hint=eyJxIjoidGdzaW0ifQ%3D%3D)). Place the trajectory CSVs expected by the scripts under `Dataset/`. The [TGSIM TRR paper](https://journals.sagepub.com/doi/10.1177/03611981241257257) describes the data collection and context.

## Quick Start

1. Place TGSIM trajectory CSV files in the `Dataset` directory expected by the scripts.
2. Run one model:

```bash
python Idm_calibration_tgsim.py
python Pt_calibration_tgsim.py
```

3. Run sweep experiments:

```bash
python Run_calibration_sweep.py --script both --combos "1,0" "1,1"
```

## Car-Following Models (four paradigms)

To avoid a two-model comparison, the repository calibrates four representative
car-following formulations spanning the main modeling traditions. All use the
same **closed-loop** trajectory simulation (gap recomputed from the simulated
follower position) and the same genetic-algorithm calibration engine
(`cf_engine.py`):

| Model | Paradigm | Script | Parameters |
|---|---|---|---|
| **IDM** | physics / gap-based | `IDM_calibration_tgsim.py` | `T, a, b, v0, s0` (δ fixed at 4) |
| **PT** | behavioral / prospect theory | `PT_calibration_tgsim.py` | `Wm, Alpha, Beta, Wc, Tmax, Gamma` |
| **OVRV** | optimal-velocity / flow | `OVRV_calibration_tgsim.py` | `kappa, lam, vmax, sc, sw` |
| **Gipps** | safety / collision-avoidance | `Gipps_calibration_tgsim.py` | `a, b, V0, s0, tau` |
| **ACC-IDM** | AV control surrogate (improved IDM + CAH) | `ACC_IDM_calibration_tgsim.py` | `T, a, b, v0, s0, c` |

Each script produces the same outputs as IDM (per-episode parameters with
multi-start dispersion, regime labels, parameter/performance tables, and
Welch-ANOVA / Games–Howell statistics pooled and per regime), written to
`Results <MODEL>/`.

```bash
python OVRV_calibration_tgsim.py
python Gipps_calibration_tgsim.py
python ACC_IDM_calibration_tgsim.py
# or sweep several models / weightings at once:
python run_calibration_sweep.py --script all --combos "1,0.5"
```

## Extended Analysis (v2 — repositioned contribution)

Beyond per-episode calibration, the `analysis/` package adds three contributions
that turn descriptive parameter comparison into a regime-aware, validated, and
consequence-linked study:

1. **Regime-stratified behavior** (`analysis/regime.py`). Each car-following
   episode is labeled `free_flow` / `synchronized` / `stop_and_go` from its
   follower speed profile. Calibration now writes a `regime` column, and class
   comparisons are repeated **within each regime** (`*_welch_anova_by_regime.csv`,
   `*_games_howell_by_regime.csv`) — exposing AV–HDV differences that are masked
   when regimes are pooled.

2. **Validation & identifiability** (`analysis/validation_identifiability.py`).
   - *Practical identifiability* from the multi-start GA repeats already run by
     the pipeline (normalized restart dispersion `CV = std/|mean|`). This reframes
     the weak PT result as **practical non-identifiability** rather than "no
     difference."
   - *Temporal out-of-sample validation*: calibrate on the first 70% of each
     episode and predict the rest (in-sample vs out-of-sample RMSE).
   - *Profile sensitivity*: one-at-a-time fit profiles around the optimum.

3. **Emergent dynamics** (`analysis/string_stability.py`,
   `analysis/mixed_autonomy.py`, `analysis/run_emergent_analysis.py`). Calibrated
   class/regime parameters are mapped to **linear string stability** (analytical
   speed-to-speed transfer function) and **mixed-autonomy platoon simulations**
   that report string amplification, safety surrogates (TTC / TET / TIT / DRAC),
   and throughput across AV market-penetration rates.

Robust statistics (Welch ANOVA + Games–Howell) are reimplemented in
`analysis/stats_tests.py` using only numpy/scipy (no `pingouin` dependency).

### Running the extended analysis

```bash
# 1) Calibration now auto-adds the regime column and class x regime statistics:
python IDM_calibration_tgsim.py
python PT_calibration_tgsim.py

# 2) Emergent dynamics on the calibrated class means (or paper Table 2 by default):
python analysis/run_emergent_analysis.py \
    --params-csv "Results IDM/idm_calib_episodes_results.csv"
# ...optionally restricted to one regime:
python analysis/run_emergent_analysis.py \
    --params-csv "Results IDM/idm_calib_episodes_results.csv" --regime stop_and_go
```

Each `analysis/*.py` module has a built-in self-test (`python analysis/<module>.py`)
that runs on synthetic data without the TGSIM trajectories.

## Citation

If you use this repository in your research, please consider citing our paper:

```bibtex
@article{beigi2026data,
  title={{A Data-Driven Comparison of Car-Following Behavior Between Autonomous and Human-Driven Vehicles}},
  author={Beigi, Pedram and Rashidi, Mohammad Emad and Li, Nachuan and Bafandkar, Shayan and Monzer, Dana and Hourdos, John and Mahmassani, Hani and Talebpour, Alireza and Hamdar, Samer H.},
  journal={Transportation Research Board - Under Review},
  pages={},
  year={},
  publisher={}
}
```
