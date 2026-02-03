from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev


@dataclass(frozen=True)
class FoldBest:
    fold: int
    best_epoch: int
    train_loss: float
    val_loss: float
    val_mae_image: float
    val_rmse_image: float
    val_mae_case: float
    lr: float
    run_dir: str


def _repo_root_from_this_file() -> Path:
    # .../repo/ml/clumping_cnn/run_kfold_train_cnn_clumping_truth.py
    return Path(__file__).resolve().parents[2]


def _read_best_from_metrics(metrics_csv: Path, fold: int, run_dir: Path) -> FoldBest:
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"No rows in metrics.csv: {metrics_csv}")

    best = min(rows, key=lambda r: float(r["val_loss"]))

    return FoldBest(
        fold=fold,
        best_epoch=int(best["epoch"]),
        train_loss=float(best["train_loss"]),
        val_loss=float(best["val_loss"]),
        val_mae_image=float(best["val_mae_image"]),
        val_rmse_image=float(best["val_rmse_image"]),
        val_mae_case=float(best["val_mae_case"]),
        lr=float(best["lr"]),
        run_dir=run_dir.as_posix(),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run case-wise k-fold training for the clumping CNN (truth_Clumping) and summarize fold metrics."
        )
    )

    p.add_argument("--k", type=int, default=5, help="Number of folds (default: 5)")
    p.add_argument("--seed", type=int, default=42)

    # Pass-through args for training
    p.add_argument("--index-csv", type=str, default=None)
    p.add_argument("--orientation", type=str, default=None)
    p.add_argument("--simulation-set", type=str, default=None)

    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--patience", type=int, default=7)

    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Base output folder (default: ml/runs/clumping_cnn_truth_kfold/<timestamp>)",
    )

    p.add_argument("--dry-run", action="store_true", help="Print commands without executing training.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.k < 2:
        raise ValueError("--k must be >= 2")

    repo_root = _repo_root_from_this_file()
    train_script = repo_root / "ml" / "clumping_cnn" / "train_cnn_clumping_truth.py"

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (repo_root / "ml" / "runs" / "clumping_cnn_truth_kfold" / ts)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable
    fold_bests: list[FoldBest] = []

    for fold in range(args.k):
        fold_dir = out_dir / f"fold_{fold}"

        cmd = [
            python_exe,
            str(train_script),
            "--kfold",
            str(args.k),
            "--fold",
            str(fold),
            "--run-dir",
            str(fold_dir),
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
        ]

        if args.index_csv:
            cmd += ["--index-csv", str(args.index_csv)]
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

        print(f"\n=== Fold {fold}/{args.k - 1} ===")
        print(" ".join([f'\"{c}\"' if " " in c else c for c in cmd]))

        if not args.dry_run:
            subprocess.run(cmd, check=True)
            best = _read_best_from_metrics(fold_dir / "metrics.csv", fold=fold, run_dir=fold_dir)
            fold_bests.append(best)

    summary_csv = out_dir / "kfold_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(FoldBest(0, 0, 0, 0, 0, 0, 0, 0, "")).keys()))
        writer.writeheader()
        for b in fold_bests:
            writer.writerow(asdict(b))

    if fold_bests:
        agg = {
            "k": args.k,
            "seed": args.seed,
            "val_mae_case_mean": mean([b.val_mae_case for b in fold_bests]),
            "val_mae_case_std": pstdev([b.val_mae_case for b in fold_bests]),
            "val_rmse_image_mean": mean([b.val_rmse_image for b in fold_bests]),
            "val_rmse_image_std": pstdev([b.val_rmse_image for b in fold_bests]),
            "val_loss_mean": mean([b.val_loss for b in fold_bests]),
            "val_loss_std": pstdev([b.val_loss for b in fold_bests]),
        }
        (out_dir / "kfold_aggregate.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

        print("\n=== Aggregate (best val_loss per fold) ===")
        print(f"val_mae_case: {agg['val_mae_case_mean']:.4f} ± {agg['val_mae_case_std']:.4f}")
        print(f"val_rmse_image: {agg['val_rmse_image_mean']:.4f} ± {agg['val_rmse_image_std']:.4f}")
        print(f"Wrote: {summary_csv}")
        print(f"Wrote: {out_dir / 'kfold_aggregate.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
