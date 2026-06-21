"""
Run calibration script with different W_POS and W_SPEED combinations.
Each run saves results in a separate folder: model-W_POS=<value>-W_SPEED=<value>
With --script both: runs IDM and PT for each combo (folders: idm-W_POS=...-W_SPEED=..., pt-W_POS=...-W_SPEED=...).

Usage:
  1) Set SCRIPT_NAME below (idm, pt, ovrv, gipps, acc_idm, both, all, or path to .py).
  2) Set COMBOS to your (W_POS, W_SPEED) pairs.
  3) Run: python run_calibration_sweep.py

  Or from command line:
    python run_calibration_sweep.py --script idm --combos "1,0" "1,1" "0.5,1"
    python run_calibration_sweep.py --script ovrv --combos "1,0.5"
    python run_calibration_sweep.py --script both --combos "1,0" "1,1"   # idm+pt
    python run_calibration_sweep.py --script all  --combos "1,0.5"        # all 5 models
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Tuple

# -----------------------------
# 1) SCRIPT TO RUN (edit here or use --script)
# -----------------------------
# Options: "idm", "pt", "both" (run IDM and PT), or full path to .py
SCRIPT_NAME = "both"

# -----------------------------
# 2) W_POS / W_SPEED COMBINATIONS (edit here or use --combos)
# -----------------------------
# Each tuple is (W_POS, W_SPEED). Results go to folder: model-W_POS=<w>-W_SPEED=<s>
COMBOS: List[Tuple[float, float]] = [
    (1.0, 0.0),
    (1.0, 0.25),
    (1.0, 0.5),
    (1.0, 0.75),
    (1.0, 1.0),
]


# Registry of available model calibration scripts (label -> filename).
MODEL_SCRIPTS = {
    "idm": "IDM_calibration_tgsim.py",
    "pt": "PT_calibration_tgsim.py",
    "ovrv": "OVRV_calibration_tgsim.py",
    "gipps": "Gipps_calibration_tgsim.py",
    "acc_idm": "ACC_IDM_calibration_tgsim.py",
}
# Convenience group: every model.
MODEL_GROUPS = {
    "both": ["idm", "pt"],
    "all": ["idm", "pt", "ovrv", "gipps", "acc_idm"],
}


def script_path(name: str, base_dir: str) -> str:
    """Resolve a model label or .py path to a full path."""
    name = name.strip().lower()
    if name in MODEL_SCRIPTS:
        return os.path.join(base_dir, MODEL_SCRIPTS[name])
    if name in MODEL_GROUPS:
        return ""  # caller should use get_scripts_for_run()
    if name.endswith(".py") and os.path.isabs(name):
        return name
    if name.endswith(".py"):
        return os.path.join(base_dir, name)
    return os.path.join(base_dir, name)


def get_scripts_for_run(script_arg: str, base_dir: str) -> List[Tuple[str, str]]:
    """
    Return list of (label, script_path) to run.
    - a single model label ("idm", "pt", "ovrv", "gipps", "acc_idm") or a .py path
      -> [(label, path)]
    - a group ("both" = idm+pt, "all" = every model) -> one entry per model
    """
    script_arg = script_arg.strip().lower()
    if script_arg in MODEL_GROUPS:
        return [(lbl, os.path.join(base_dir, MODEL_SCRIPTS[lbl]))
                for lbl in MODEL_GROUPS[script_arg]]
    path = script_path(script_arg, base_dir)
    if path:
        label = script_arg if script_arg in MODEL_SCRIPTS else os.path.splitext(os.path.basename(path))[0]
        return [(label, path)]
    return []


def folder_name(w_pos: float, w_speed: float, model_label: str = "model") -> str:
    """Output folder name: <model_label>-W_POS=<value>-W_SPEED=<value>"""
    return f"{model_label}-W_POS={w_pos}-W_SPEED={w_speed}"


def parse_combo(s: str) -> Tuple[float, float]:
    """Parse '1,0' or '1.0,0.5' into (float, float)."""
    parts = s.replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid combo '{s}': expected 'W_POS,W_SPEED' (e.g. 1,0)")
    return float(parts[0]), float(parts[1])


def run_one(script_path_str: str, w_pos: float, w_speed: float, subfolder: str = None) -> int:
    """Run calibration with given W_POS, W_SPEED. Returns process return code."""
    script_dir = os.path.dirname(os.path.abspath(script_path_str))
    if subfolder is None:
        subfolder = folder_name(w_pos, w_speed)

    env = os.environ.copy()
    env["CALIB_W_POS"] = str(w_pos)
    env["CALIB_W_SPEED"] = str(w_speed)
    env["CALIB_OUTPUT_SUBFOLDER"] = subfolder

    cmd = [sys.executable, os.path.basename(script_path_str)]
    result = subprocess.run(
        cmd,
        cwd=script_dir,
        env=env,
    )
    return result.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run calibration with multiple W_POS/W_SPEED combinations; each run saves to model-W_POS=<w>-W_SPEED=<s>"
    )
    parser.add_argument(
        "--script",
        type=str,
        default=SCRIPT_NAME,
        help="Model to run: 'idm', 'pt', 'ovrv', 'gipps', 'acc_idm', 'both' (idm+pt), 'all', or path to .py (default: %(default)s)",
    )
    parser.add_argument(
        "--combos",
        type=str,
        nargs="+",
        default=None,
        metavar="W_POS,W_SPEED",
        help="Space-separated combos, e.g. '1,0' '1,1' '0.5,1' (default: use COMBOS in script)",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = get_scripts_for_run(args.script, base_dir)
    if not scripts:
        print(f"ERROR: Unknown script: {args.script}")
        sys.exit(1)

    for _label, path in scripts:
        if not os.path.isfile(path):
            print(f"ERROR: Script not found: {path}")
            sys.exit(1)

    if args.combos:
        combos = [parse_combo(c) for c in args.combos]
    else:
        combos = COMBOS

    if not combos:
        print("ERROR: No combinations specified. Set COMBOS in script or use --combos.")
        sys.exit(1)

    run_both = len(scripts) > 1
    total_runs = len(scripts) * len(combos)

    print("=" * 60)
    print("Calibration sweep: W_POS / W_SPEED")
    print("=" * 60)
    print(f"Script(s): {args.script}" + (" (IDM + PT)" if run_both else ""))
    for label, path in scripts:
        print(f"  - {label}: {path}")
    print(f"Combinations: {combos}")
    if run_both:
        print("Output folders (per script):")
        for label, _ in scripts:
            print(f"  {label}: {[folder_name(w, s, label) for w, s in combos]}")
    else:
        print(f"Output folders: {[folder_name(w, s) for w, s in combos]}")
    print("=" * 60)

    failed = []
    run_index = 0
    for (w_pos, w_speed) in combos:
        for label, script_path_str in scripts:
            run_index += 1
            subfolder = folder_name(w_pos, w_speed, label if run_both else "model")
            print(f"\n[{run_index}/{total_runs}] {label.upper()} W_POS={w_pos}, W_SPEED={w_speed} -> {subfolder}")
            code = run_one(script_path_str, w_pos, w_speed, subfolder=subfolder)
            if code != 0:
                print(f"  FAILED (exit code {code})")
                failed.append((label, w_pos, w_speed))
            else:
                print(f"  Done. Results in: {subfolder}")

    print("\n" + "=" * 60)
    if failed:
        print(f"Completed with errors: {len(failed)} run(s) failed: {failed}")
        sys.exit(1)
    print("All runs completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
