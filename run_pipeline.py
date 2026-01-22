from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

STEPS = {
    "truth_join": REPO_ROOT / "truth_join" / "join_truth_to_hemipy.py",
    "dataset_index": REPO_ROOT / "dataset_index" / "build_image_dataset_index.py",
    "image_baseline": REPO_ROOT / "image_baseline" / "image_baseline_features.py",
    "cnn_baseline": REPO_ROOT / "cnn_baseline" / "train_cnn_pai.py",
    "estimations_eval": REPO_ROOT / "estimations_eval" / "evaluate_estimations.py",
}


def _run(cmd: list[str], cwd: Path, dry_run: bool) -> int:
    printable = " ".join([f'"{c}"' if " " in c else c for c in cmd])
    print(f"\n>> {printable}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd))
    return int(proc.returncode)


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Run the HemiPy helper scripts in order using the reorganized folder layout.\n"
            "Tip: skip 'cnn_baseline' if you don't have PyTorch installed."
        )
    )
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use (default: current interpreter).",
    )
    p.add_argument(
        "--steps",
        nargs="+",
        default=[
            "truth_join",
            "dataset_index",
            "image_baseline",
            "estimations_eval",
        ],
        help=(
            "Steps to run in order. Choices: "
            + ", ".join(STEPS.keys())
            + ". Default excludes cnn_baseline."
        ),
    )
    p.add_argument(
        "--include-cnn",
        action="store_true",
        help="Convenience flag to append cnn_baseline before estimations_eval.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    args = p.parse_args()

    steps = list(args.steps)
    if args.include_cnn and "cnn_baseline" not in steps:
        if "estimations_eval" in steps:
            i = steps.index("estimations_eval")
            steps.insert(i, "cnn_baseline")
        else:
            steps.append("cnn_baseline")

    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise SystemExit(f"Unknown steps: {unknown}. Valid: {list(STEPS.keys())}")

    py = str(args.python)
    for step in steps:
        script = STEPS[step]
        if not script.exists():
            raise SystemExit(f"Missing script for step '{step}': {script}")

        rc = _run([py, str(script)], cwd=REPO_ROOT, dry_run=bool(args.dry_run))
        if rc != 0:
            print(f"\nStep '{step}' failed with exit code {rc}.")
            return rc

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
