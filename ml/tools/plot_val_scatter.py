from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "pai").resolve()))
from pai_cnn.common import (  # noqa: E402
    PaiIndexDataset,
    build_model,
    build_transforms,
    collate_keep_meta,
    get_repo_root_from_any_ml_file,
    read_index_csv,
    split_cases_kfold,
    split_cases_stratified,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Load a checkpoint, run it on a validation split, and plot predicted vs truth with R2 or RMSE."
        )
    )
    p.add_argument("--checkpoint", required=True, type=str, help="Path to model checkpoint (.pt)")
    p.add_argument("--index-csv", type=str, default=None, help="Path to image_dataset_index.csv (optional)")
    p.add_argument("--target-col", type=str, default="truth_PAI", help="Column to predict (e.g., truth_PAI, truth_Clumping)")
    p.add_argument("--orientation", type=str, default=None, help="Optional filter, e.g., RND")
    p.add_argument("--simulation-set", type=str, default=None, help="Optional filter, e.g., RND")
    p.add_argument("--case-range-min", type=int, default=None)
    p.add_argument("--case-range-max", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=0.2, help="Used when kfold is not set")
    p.add_argument("--val-min-cases-in-range", type=int, default=0, help="Min validation cases inside case range (non-kfold)")
    p.add_argument("--kfold", type=int, default=None, help="Optional k for case-wise k-fold")
    p.add_argument("--fold", type=int, default=0, help="Fold index when --kfold is set")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metric", choices=["r2", "rmse"], default="r2")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--out", type=str, default=None, help="Output PNG path (default: auto in run dir)")
    p.add_argument("--csv-out", type=str, default=None, help="Optional CSV path for per-image truth/pred; default auto")
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(truth: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    # Returns (r2, rmse)
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = math.sqrt(ss_res / max(1, truth.size))
    return r2, rmse


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if (args.case_range_min is None) ^ (args.case_range_max is None):
        raise ValueError("Provide both --case-range-min and --case-range-max (or neither).")
    case_range = None
    if args.case_range_min is not None and args.case_range_max is not None:
        case_range = (int(args.case_range_min), int(args.case_range_max))

    repo_root = get_repo_root_from_any_ml_file(__file__)
    index_csv = Path(args.index_csv) if args.index_csv else (repo_root / "shared" / "dataset_index" / "image_dataset_index.csv")

    rows = read_index_csv(
        index_csv=index_csv,
        target_col=args.target_col,
        orientation=args.orientation,
        simulation_set=args.simulation_set,
    )

    if args.kfold is not None:
        train_rows, val_rows = split_cases_kfold(
            rows,
            k=int(args.kfold),
            fold=int(args.fold),
            seed=args.seed,
            case_range=case_range,
        )
    else:
        train_rows, val_rows = split_cases_stratified(
            rows,
            val_fraction=args.val_fraction,
            seed=args.seed,
            case_range=case_range,
            min_val_cases_in_range=int(args.val_min_cases_in_range),
        )
    # Only use validation rows for scoring/plotting

    val_tf = build_transforms(args.img_size, train=False)
    val_ds = PaiIndexDataset(val_rows, repo_root=repo_root, transform=val_tf)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not args.cpu,
        collate_fn=collate_keep_meta,
    )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = build_model(pretrained=False).to(device)

    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    state = ckpt.get("model_state") if isinstance(ckpt, dict) else None
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must be a dict with 'model_state'")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"Checkpoint load_state_dict had missing={missing}, unexpected={unexpected}")

    model.eval()
    preds: list[float] = []
    truths: list[float] = []

    with torch.no_grad():
        for xb, yb, _meta in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(xb).squeeze(1)
            preds.append(out.detach().cpu().numpy())
            truths.append(yb.detach().cpu().numpy())

    if not preds:
        raise RuntimeError("No validation samples found after filtering.")

    pred_arr = np.concatenate(preds).astype(float).reshape(-1)
    truth_arr = np.concatenate(truths).astype(float).reshape(-1)

    r2, rmse = compute_metrics(truth_arr, pred_arr)
    metric_value = r2 if args.metric == "r2" else rmse

    # Write per-image truth/pred to CSV for easier inspection
    csv_path = Path(args.csv_out) if args.csv_out else Path(
        f"val_truth_pred_{args.target_col}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "case_norm", "truth", "pred", "error", "abs_error"])
        # Need meta info; rebuild loader with meta kept
        # The earlier val_loader already yields meta; reuse
        for xb, yb, meta in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(xb).squeeze(1)
            preds_np = out.detach().cpu().numpy().astype(float).reshape(-1)
            truths_np = yb.detach().cpu().numpy().astype(float).reshape(-1)
            for p, t, m in zip(preds_np, truths_np, meta):
                pf = float(p)
                tf = float(t)
                err = pf - tf
                writer.writerow([m.image_path, m.case_norm, tf, pf, err, abs(err)])
        writer.writerow(["__summary__", "", f"r2={r2:.4f}", f"rmse={rmse:.4f}", "", ""])
    print(f"Saved CSV to: {csv_path}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(truth_arr, pred_arr, s=12, alpha=0.7, edgecolors="none")

    lo = float(np.min([truth_arr.min(), pred_arr.min()]))
    hi = float(np.max([truth_arr.max(), pred_arr.max()]))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)

    ax.set_xlabel(f"Truth ({args.target_col})")
    ax.set_ylabel("Prediction")
    title_metric = "R2" if args.metric == "r2" else "RMSE"
    ax.set_title(f"Val scatter — {title_metric}: {metric_value:.4f}")
    ax.grid(True, alpha=0.3)

    out_path = Path(args.out) if args.out else Path(f"scatter_{args.target_col}_{time.strftime('%Y%m%d-%H%M%S')}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved scatter to: {out_path}")
    print(f"R2={r2:.4f}, RMSE={rmse:.4f}, N={truth_arr.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
