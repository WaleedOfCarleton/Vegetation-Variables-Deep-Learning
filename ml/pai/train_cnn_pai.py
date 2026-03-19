from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

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

RND_SIMULATION_SETS = {"RND", "Sunny Hemiphotos"}


def _is_rnd_simulation_set(sim_set: str | None) -> bool:
    return (sim_set or "") in RND_SIMULATION_SETS


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
    p.add_argument(
        "--case-range-min",
        type=int,
        default=None,
        help="Optional case-number range minimum (e.g., 1 for Case 001). Used for stratified splitting.",
    )
    p.add_argument(
        "--case-range-max",
        type=int,
        default=None,
        help="Optional case-number range maximum (e.g., 10 for Case 010). Used for stratified splitting.",
    )
    p.add_argument(
        "--val-min-cases-in-range",
        type=int,
        default=0,
        help=(
            "When not using --kfold, ensure validation includes at least this many cases from the given "
            "--case-range-min/--case-range-max. Default: 0 (no constraint)."
        ),
    )
    p.add_argument(
        "--val-cases-file",
        type=str,
        default=None,
        help=(
            "Optional path to a text file listing case_norm values (one per line) to use for validation. "
            "If set, overrides --val-fraction/--kfold."
        ),
    )
    p.add_argument(
        "--train-cases-file",
        type=str,
        default=None,
        help="Optional text file with training case_norm values, one per line. If provided with --val-cases-file, overlap is not allowed.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--rnd-train-weight",
        type=float,
        default=1.0,
        help=(
            "Oversample training images in the RND/Sunny Hemiphotos domain by this multiplicative weight. "
            "1.0 disables weighting. Example: 5.0 makes that domain ~5x more likely to be sampled."
        ),
    )

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
    p.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional path to a .pt checkpoint produced by this trainer (e.g. model_best.pt). "
            "If provided, the model weights are initialized from the checkpoint before training."
        ),
    )
    p.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="If set, freeze all layers except the final fc regression head.",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only)")
    p.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs)")
    p.add_argument(
        "--best-metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_mae_case", "val_mae_image", "val_rmse_image"],
        help=(
            "Which validation metric to minimize for saving model_best.pt and early stopping. "
            "Default: val_loss."
        ),
    )

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

    case_range = None
    if (args.case_range_min is None) ^ (args.case_range_max is None):
        raise ValueError("Provide both --case-range-min and --case-range-max (or neither).")
    if args.case_range_min is not None and args.case_range_max is not None:
        case_range = (int(args.case_range_min), int(args.case_range_max))

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

    if args.val_cases_file and args.kfold is not None:
        raise ValueError("--val-cases-file cannot be combined with --kfold")

    train_set = None
    if args.train_cases_file:
        train_cases = [ln.strip() for ln in Path(args.train_cases_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not train_cases:
            raise ValueError(f"No case names found in --train-cases-file: {args.train_cases_file}")
        train_set = set(train_cases)

    if args.val_cases_file:
        val_cases = [ln.strip() for ln in Path(args.val_cases_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not val_cases:
            raise ValueError(f"No case names found in --val-cases-file: {args.val_cases_file}")
        val_set = set(val_cases)
        if train_set and (train_set & val_set):
            overlap = sorted(train_set & val_set)
            raise ValueError(f"Train/val case sets overlap: {overlap}")

        if train_set is not None:
            train_rows = [r for r in rows if r.case_norm in train_set]
        else:
            train_rows = [r for r in rows if r.case_norm not in val_set]
        val_rows = [r for r in rows if r.case_norm in val_set]
        if not val_rows:
            raise ValueError("Validation set is empty after applying --val-cases-file")
        if not train_rows:
            raise ValueError("Training set is empty after applying provided train/val splits")
    elif args.kfold is not None:
        train_rows, val_rows = split_cases_kfold(
            rows,
            k=int(args.kfold),
            fold=int(args.fold),
            seed=args.seed,
            case_range=case_range,
        )
    else:
        # Case-wise split, optionally forcing some validation cases from a particular range.
        from pai_cnn.common import split_cases_stratified  # local import to keep API stable

        train_rows, val_rows = split_cases_stratified(
            rows,
            val_fraction=args.val_fraction,
            seed=args.seed,
            case_range=case_range,
            min_val_cases_in_range=int(args.val_min_cases_in_range),
        )

    train_tf = build_transforms(args.img_size, train=True)
    val_tf = build_transforms(args.img_size, train=False)

    train_ds = PaiIndexDataset(train_rows, repo_root=repo_root, transform=train_tf)
    val_ds = PaiIndexDataset(val_rows, repo_root=repo_root, transform=val_tf)

    # Optional per-domain validation loaders (only meaningful when mixing simulation sets).
    val_loader_rnd = None
    val_loader_non_rnd = None
    val_rows_rnd = [r for r in val_rows if _is_rnd_simulation_set(r.simulation_set)]
    val_rows_non_rnd = [r for r in val_rows if not _is_rnd_simulation_set(r.simulation_set)]
    if len(val_rows_rnd) > 0 and len(val_rows_non_rnd) > 0:
        val_ds_rnd = PaiIndexDataset(val_rows_rnd, repo_root=repo_root, transform=val_tf)
        val_ds_non_rnd = PaiIndexDataset(val_rows_non_rnd, repo_root=repo_root, transform=val_tf)
        val_loader_rnd = DataLoader(
            val_ds_rnd,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=not args.cpu,
            collate_fn=collate_keep_meta,
        )
        val_loader_non_rnd = DataLoader(
            val_ds_non_rnd,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=not args.cpu,
            collate_fn=collate_keep_meta,
        )

    sampler = None
    if float(args.rnd_train_weight) != 1.0:
        if float(args.rnd_train_weight) < 1.0:
            raise ValueError("--rnd-train-weight must be >= 1.0")

        weights = [float(args.rnd_train_weight) if _is_rnd_simulation_set(r.simulation_set) else 1.0 for r in train_rows]
        gen = torch.Generator()
        gen.manual_seed(int(args.seed))
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True, generator=gen)
        print(f"Using WeightedRandomSampler for training (RND weight={args.rnd_train_weight:g})")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
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

    model = build_model(pretrained=args.pretrained)
    if args.init_checkpoint:
        ckpt = torch.load(str(args.init_checkpoint), map_location="cpu")
        state = ckpt.get("model_state") if isinstance(ckpt, dict) else None
        if not isinstance(state, dict):
            raise ValueError("--init-checkpoint must be a checkpoint dict with a 'model_state' key")
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise ValueError(f"Checkpoint load_state_dict had missing={missing}, unexpected={unexpected}")
        print(f"Initialized model weights from: {args.init_checkpoint}")

    if bool(args.freeze_backbone):
        for n, p in model.named_parameters():
            if not n.startswith("fc."):
                p.requires_grad = False
        print("Freezing backbone (training fc head only)")

    model = model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
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
            "init_checkpoint": args.init_checkpoint,
            "freeze_backbone": bool(args.freeze_backbone),
            "device": device.type,
            "amp": use_amp,
            "best_metric": args.best_metric,
            "rnd_train_weight": float(args.rnd_train_weight),
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
                "val_mae_image_rnd",
                "val_mae_case_rnd",
                "val_mae_image_non_rnd",
                "val_mae_case_non_rnd",
                "lr",
            ],
        )
        writer.writeheader()

        best_score = float("inf")
        best_by: dict[str, float] = {
            "val_loss": float("inf"),
            "val_mae_case": float("inf"),
        }
        bad_epochs = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            n = 0

            if tqdm is not None:
                it = tqdm(
                    train_loader,
                    desc=f"Ep {epoch}/{args.epochs}",
                    leave=False,
                    dynamic_ncols=True,
                )
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

            if tqdm is not None:
                it.close()

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
            eval_rnd = None
            eval_non_rnd = None
            if val_loader_rnd is not None and val_loader_non_rnd is not None:
                eval_rnd = evaluate(model, val_loader_rnd, device=device)
                eval_non_rnd = evaluate(model, val_loader_non_rnd, device=device)
            lr = optimizer.param_groups[0]["lr"]

            metric_map = {
                "val_loss": float(val_loss),
                "val_mae_case": float(eval_metrics["mae_case"]),
                "val_mae_image": float(eval_metrics["mae_image"]),
                "val_rmse_image": float(eval_metrics["rmse_image"]),
            }
            score = metric_map[str(args.best_metric)]

            parts = [
                f"epoch {epoch}",
                f"train_loss {train_loss:.4f}",
                f"val_loss {val_loss:.4f}",
                f"val_mae_case {eval_metrics['mae_case']:.4f}",
                f"val_rmse_image {eval_metrics['rmse_image']:.4f}",
            ]
            if eval_rnd is not None and eval_non_rnd is not None:
                parts += [
                    f"val_mae_case_rnd {eval_rnd['mae_case']:.4f}",
                    f"val_mae_case_non_rnd {eval_non_rnd['mae_case']:.4f}",
                ]
            parts.append(f"lr {lr:.2e}")
            print(" | ".join(parts))

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_mae_image": f"{eval_metrics['mae_image']:.6f}",
                    "val_rmse_image": f"{eval_metrics['rmse_image']:.6f}",
                    "val_mae_case": f"{eval_metrics['mae_case']:.6f}",
                    "val_mae_image_rnd": (
                        f"{eval_rnd['mae_image']:.6f}" if eval_rnd is not None else ""
                    ),
                    "val_mae_case_rnd": (
                        f"{eval_rnd['mae_case']:.6f}" if eval_rnd is not None else ""
                    ),
                    "val_mae_image_non_rnd": (
                        f"{eval_non_rnd['mae_image']:.6f}" if eval_non_rnd is not None else ""
                    ),
                    "val_mae_case_non_rnd": (
                        f"{eval_non_rnd['mae_case']:.6f}" if eval_non_rnd is not None else ""
                    ),
                    "lr": f"{lr:.8f}",
                }
            )
            f.flush()

            # Checkpointing
            ckpt = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_mae_case": float(eval_metrics["mae_case"]),
                "val_mae_image": float(eval_metrics["mae_image"]),
                "val_rmse_image": float(eval_metrics["rmse_image"]),
                "best_metric": str(args.best_metric),
                "best_metric_value": float(score),
                "config": {
                    "img_size": args.img_size,
                    "target_col": args.target_col,
                    "orientation": args.orientation,
                    "simulation_set": args.simulation_set,
                    "pretrained": args.pretrained,
                    "init_checkpoint": args.init_checkpoint,
                    "freeze_backbone": bool(args.freeze_backbone),
                    "best_metric": args.best_metric,
                },
            }
            torch.save(ckpt, run_dir / "model_last.pt")

            # Always keep these two around for analysis/comparison.
            if float(val_loss) < best_by["val_loss"]:
                best_by["val_loss"] = float(val_loss)
                torch.save(ckpt, run_dir / "model_best_val_loss.pt")
            if float(eval_metrics["mae_case"]) < best_by["val_mae_case"]:
                best_by["val_mae_case"] = float(eval_metrics["mae_case"])
                torch.save(ckpt, run_dir / "model_best_val_mae_case.pt")

            if score < best_score:
                best_score = score
                bad_epochs = 0
                torch.save(ckpt, run_dir / "model_best.pt")
            else:
                bad_epochs += 1

            if bad_epochs >= args.patience:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
