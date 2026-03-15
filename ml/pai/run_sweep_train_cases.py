from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Sequence

# Allow `from pai_cnn...` when invoked as `python ml/pai/run_sweep_train_cases.py`
import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))

from pai_cnn.common import get_repo_root_from_any_ml_file, read_index_csv  # noqa: E402


@dataclass(frozen=True)
class SweepResult:
    n_train_cases: int
    n_val_cases: int
    best_epoch: int
    train_loss: float
    val_loss: float
    val_mae_image: float
    val_rmse_image: float
    val_mae_case: float
    lr: float
    run_dir: str


def _read_best(metrics_csv: Path, *, best_metric: str) -> SweepResult:
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"No rows in metrics.csv: {metrics_csv}")
    if best_metric not in rows[0]:
        raise RuntimeError(
            f"best_metric '{best_metric}' not found in metrics.csv columns: {sorted(rows[0].keys())}"
        )

    best = min(rows, key=lambda r: float(r[best_metric]))

    return SweepResult(
        n_train_cases=int(best.get("n_train_cases", -1)),
        n_val_cases=int(best.get("n_val_cases", -1)),
        best_epoch=int(best["epoch"]),
        train_loss=float(best["train_loss"]),
        val_loss=float(best["val_loss"]),
        val_mae_image=float(best["val_mae_image"]),
        val_rmse_image=float(best["val_rmse_image"]),
        val_mae_case=float(best["val_mae_case"]),
        lr=float(best["lr"]),
        run_dir=str(metrics_csv.parent.as_posix()),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sweep training set size (cases) for PAI CNN: train on 1..N cases, hold out the rest for validation."
        )
    )
    p.add_argument("--max-train-cases", type=int, default=74, help="Upper limit for training cases to sweep (default: 74)")
    p.add_argument(
        "--train-cases-list",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of training case counts to run (e.g., '15,30,45,60'). "
            "If set, overrides the 1..max-train-cases sweep."
        ),
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for shuffling case order")

    # Dataset filters
    p.add_argument("--index-csv", type=str, default=None)
    p.add_argument("--orientation", type=str, default=None)
    p.add_argument("--simulation-set", type=str, default=None)

    # Training hyperparameters (aligned with plateau study defaults)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument(
        "--best-metric",
        type=str,
        default="val_mae_case",
        choices=["val_loss", "val_mae_case", "val_mae_image", "val_rmse_image"],
    )
    p.add_argument("--rnd-train-weight", type=float, default=1.0)

    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true")

    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Base output folder (default: ml/runs/pai_cnn_case_sweep/<timestamp>)",
    )

    p.add_argument("--dry-run", action="store_true", help="Print commands without executing training.")
    return p.parse_args()


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    repo_root = get_repo_root_from_any_ml_file(__file__)
    train_script = repo_root / "ml" / "pai" / "train_cnn_pai.py"

    index_csv = (
        Path(args.index_csv)
        if args.index_csv
        else (repo_root / "shared" / "dataset_index" / "image_dataset_index.csv")
    )
    rows = read_index_csv(
        index_csv=index_csv,
        target_col="truth_PAI",
        orientation=args.orientation,
        simulation_set=args.simulation_set,
    )

    cases = sorted({r.case_norm for r in rows})
    if len(cases) < 2:
        raise RuntimeError("Need at least 2 distinct cases to run the sweep.")

    import random

    rng = random.Random(args.seed)
    rng.shuffle(cases)

    if args.train_cases_list:
        train_counts: list[int] = []
        for tok in args.train_cases_list.split(","):
            tok = tok.strip()
            if not tok:
                continue
            n = int(tok)
            if n <= 0:
                raise ValueError("Train case counts must be positive")
            if n >= len(cases):
                # Skip counts that would leave no validation cases.
                continue
            if n not in train_counts:
                train_counts.append(n)
        if not train_counts:
            raise ValueError("No usable train counts found in --train-cases-list")
        max_train = max(train_counts)
    else:
        max_train = min(int(args.max_train_cases), len(cases) - 1)
        train_counts = list(range(1, max_train + 1))
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "ml" / "runs" / "pai_cnn_case_sweep" / ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable
    results: list[SweepResult] = []

    for n_train in train_counts:
        train_cases = cases[:n_train]
        val_cases = cases[n_train:]
        if not val_cases:
            break

        run_dir = out_dir / f"train_{n_train:03d}_cases"
        val_file = run_dir / "val_cases.txt"
        _write_lines(val_file, val_cases)

        cmd = [
            python_exe,
            str(train_script),
            "--run-dir",
            str(run_dir),
            "--val-cases-file",
            str(val_file),
            "--img-size",
            str(args.img_size),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--seed",
            str(args.seed),
            "--num-workers",
            str(args.num_workers),
            "--patience",
            str(args.patience),
            "--best-metric",
            str(args.best_metric),
            "--rnd-train-weight",
            str(args.rnd_train_weight),
            "--target-col",
            "truth_PAI",
        ]

        if args.index_csv:
            cmd += ["--index-csv", str(index_csv)]
        if args.orientation:
            cmd += ["--orientation", str(args.orientation)]
        if args.simulation_set:
            cmd += ["--simulation-set", str(args.simulation_set)]
        if args.pretrained:
            cmd += ["--pretrained"]
        if args.cpu:
            cmd += ["--cpu"]
        if args.amp:
            cmd += ["--amp"]

        print(f"\n=== Train cases: {n_train}/{max_train} (val={len(val_cases)}) ===")
        print(" ".join([f'\"{c}\"' if " " in c else c for c in cmd]))

        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)

        best = _read_best(run_dir / "metrics.csv", best_metric=str(args.best_metric))
        # Annotate counts for clarity in the summary
        best = SweepResult(
            n_train_cases=n_train,
            n_val_cases=len(val_cases),
            best_epoch=best.best_epoch,
            train_loss=best.train_loss,
            val_loss=best.val_loss,
            val_mae_image=best.val_mae_image,
            val_rmse_image=best.val_rmse_image,
            val_mae_case=best.val_mae_case,
            lr=best.lr,
            run_dir=best.run_dir,
        )
        results.append(best)

    if args.dry_run:
        return 0

    summary_csv = out_dir / "case_sweep_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    agg = {
        "best_metric": str(args.best_metric),
        "seed": args.seed,
        "train_counts": train_counts,
        "n_total_cases": len(cases),
        "val_mae_case_min": min(r.val_mae_case for r in results),
        "val_mae_case_mean": mean(r.val_mae_case for r in results),
        "val_mae_case_argmin_train_cases": min(results, key=lambda r: r.val_mae_case).n_train_cases,
    }
    (out_dir / "case_sweep_aggregate.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    print(f"\nWrote: {summary_csv}")
    print(f"Wrote: {out_dir / 'case_sweep_aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
