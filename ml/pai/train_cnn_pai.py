from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

# Allow `from pai_cnn...` when invoked as `python ml/pai/train_cnn_pai.py`
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pai_cnn.common import (  # noqa: E402
    PaiIndexDataset,
    build_model,
    build_transforms,
    collate_keep_meta,
    evaluate,
    get_repo_root_from_any_ml_file,
    read_index_csv,
    save_json,
    split_cases,
    split_cases_kfold,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a CNN regressor to predict PAI from simulated images.")
    p.add_argument(
        "--index-csv",
        type=str,
        default=None,
        help="Path to shared/dataset_index/image_dataset_index.csv (default: repo shared path)",
    )
    p.add_argument("--target-col", type=str, default="truth_PAI")
    p.add_argument("--orientation", type=str, default=None, help="Optional filter, e.g. ERECT")
    p.add_argument("--simulation-set", type=str, default=None)

    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument(
        "--kfold",
        type=int,
        default=None,
        help="Optional case-wise k-fold CV (e.g. 5). If set, --val-fraction is ignored.",
    )
    p.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Which fold index to use as validation (0..k-1) when --kfold is set.",
    )

    p.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained ResNet18 backbone")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only)")
    p.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs)")

    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Output folder for this run (default: ml/runs/pai_cnn/<timestamp>)",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    repo_root = get_repo_root_from_any_ml_file(__file__)
    index_csv = (
        Path(args.index_csv)
        if args.index_csv
        else (repo_root / "shared" / "dataset_index" / "image_dataset_index.csv")
    )

    rows = read_index_csv(
        index_csv=index_csv,
        target_col=args.target_col,
        orientation=args.orientation,
        simulation_set=args.simulation_set,
    )

    if args.kfold is not None:
        train_rows, val_rows = split_cases_kfold(rows, k=int(args.kfold), fold=int(args.fold), seed=args.seed)
    else:
        train_rows, val_rows = split_cases(rows, val_fraction=args.val_fraction, seed=args.seed)

    train_tf = build_transforms(args.img_size, train=True)
    val_tf = build_transforms(args.img_size, train=False)

    train_ds = PaiIndexDataset(train_rows, repo_root=repo_root, transform=train_tf)
    val_ds = PaiIndexDataset(val_rows, repo_root=repo_root, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=not args.cpu,
        collate_fn=collate_keep_meta,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not args.cpu,
        collate_fn=collate_keep_meta,
    )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    if device.type == "cuda":
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: cpu")

    model = build_model(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    loss_fn: nn.Module = nn.SmoothL1Loss(beta=0.5)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else (repo_root / "ml" / "runs" / "pai_cnn" / ts)
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        run_dir / "config.json",
        {
            "index_csv": str(index_csv.as_posix()),
            "target_col": args.target_col,
            "orientation": args.orientation,
            "simulation_set": args.simulation_set,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "val_fraction": args.val_fraction,
            "kfold": args.kfold,
            "fold": args.fold,
            "seed": args.seed,
            "pretrained": args.pretrained,
            "device": device.type,
            "amp": use_amp,
        },
    )

    (run_dir / "splits").mkdir(exist_ok=True)
    (run_dir / "splits" / "train_cases.txt").write_text(
        "\n".join(sorted({r.case_norm for r in train_rows})), encoding="utf-8"
    )
    (run_dir / "splits" / "val_cases.txt").write_text(
        "\n".join(sorted({r.case_norm for r in val_rows})), encoding="utf-8"
    )

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_mae_image",
                "val_rmse_image",
                "val_mae_case",
                "lr",
            ],
        )
        writer.writeheader()

        best_val = float("inf")
        bad_epochs = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            n = 0

            if tqdm is not None:
                it = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
            else:
                it = train_loader

            for step, (xb, yb, _meta) in enumerate(it, start=1):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(xb)
                    loss = loss_fn(pred, yb)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running += loss.item() * yb.numel()
                n += yb.numel()

                if tqdm is not None:
                    it.set_postfix(loss=float(loss.item()))
                elif step % 50 == 0:
                    print(f"  train step {step}: loss={loss.item():.4f}")

            train_loss = running / max(1, n)

            # Validation
            model.eval()
            val_running = 0.0
            val_n = 0
            with torch.no_grad():
                for xb, yb, _meta in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    val_running += loss.item() * yb.numel()
                    val_n += yb.numel()

            val_loss = val_running / max(1, val_n)
            scheduler.step(val_loss)

            eval_metrics = evaluate(model, val_loader, device=device)
            lr = optimizer.param_groups[0]["lr"]

            print(
                " | ".join(
                    [
                        f"epoch {epoch}",
                        f"train_loss {train_loss:.4f}",
                        f"val_loss {val_loss:.4f}",
                        f"val_mae_case {eval_metrics['mae_case']:.4f}",
                        f"val_rmse_image {eval_metrics['rmse_image']:.4f}",
                        f"lr {lr:.2e}",
                    ]
                )
            )

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_mae_image": f"{eval_metrics['mae_image']:.6f}",
                    "val_rmse_image": f"{eval_metrics['rmse_image']:.6f}",
                    "val_mae_case": f"{eval_metrics['mae_case']:.6f}",
                    "lr": f"{lr:.8f}",
                }
            )
            f.flush()

            # Checkpointing
            ckpt = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": {
                    "img_size": args.img_size,
                    "target_col": args.target_col,
                    "orientation": args.orientation,
                    "simulation_set": args.simulation_set,
                    "pretrained": args.pretrained,
                },
            }
            torch.save(ckpt, run_dir / "model_last.pt")

            if val_loss < best_val:
                best_val = val_loss
                bad_epochs = 0
                torch.save(ckpt, run_dir / "model_best.pt")
            else:
                bad_epochs += 1

            if bad_epochs >= args.patience:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
