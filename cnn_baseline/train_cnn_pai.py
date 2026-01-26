from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
# from zmq import device

try:
    from PIL import Image
except Exception as e:
    raise SystemExit("Missing dependency: Pillow. Install with: pip install pillow") from e

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
except Exception as e:
    raise SystemExit(
        "Missing dependency: PyTorch/torchvision.\n"
        "Install (CPU): pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
        "Install (CUDA): follow https://pytorch.org/get-started/locally/"
    ) from e


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_INDEX = REPO_ROOT / "dataset_index" / "image_dataset_index.csv"

DEFAULT_OUT_METRICS = HERE / "cnn_baseline_metrics.csv"
DEFAULT_OUT_PER_IMAGE = HERE / "cnn_per_image_predictions.csv"
DEFAULT_OUT_PER_SITE = HERE / "cnn_per_site_predictions.csv"
DEFAULT_PROGRESS_CSV = HERE / "cnn_training_progress.csv"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _archive_existing_file(path: Path, archive_dir: Path) -> Path | None:
    if not path.exists():
        return None

    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    dest = archive_dir / f"{path.stem}_{stamp}{path.suffix}"
    # Extremely unlikely, but avoid overwriting.
    if dest.exists():
        dest = archive_dir / f"{path.stem}_{stamp}_{os.getpid()}{path.suffix}"

    shutil.move(str(path), str(dest))
    return dest


def _copy_state_dict_to_cpu(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class ProgressWriter:
    def __init__(self, path: Path | None):
        self.path = path
        self._fp = None
        self._writer = None

    def __enter__(self):
        if self.path is None:
            return self

        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self._fp = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fp,
            fieldnames=[
                "timestamp_utc",
                "device",
                "kfold",
                "fold",
                "epoch",
                "event",
                "n_train",
                "n_test",
                "train_mse",
                "val_rmse",
                "val_mae",
                "val_r",
                "val_r2",
                "best_epoch",
                "best_val_rmse",
                "early_stop",
                "batch_size",
                "num_workers",
                "image_size",
                "backbone",
                "freeze_backbone",
                "nonnegative_head",
                "aug",
                "aug_scale_min",
                "aug_hflip",
                "aug_color_jitter",
                "lr",
                "weight_decay",
            ],
        )
        if is_new:
            self._writer.writeheader()
            self._fp.flush()
        return self

    def write(self, row: dict) -> None:
        if self._writer is None or self._fp is None:
            return
        self._writer.writerow(row)
        self._fp.flush()
        try:
            os.fsync(self._fp.fileno())
        except Exception:
            # Some environments/filesystems may not support fsync; flush is still helpful.
            pass

    def __exit__(self, exc_type, exc, tb):
        if self._fp is not None:
            self._fp.close()
        self._fp = None
        self._writer = None
        return False


# ---------------------------
# Metrics helpers
# ---------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = int(y_true.size)
    if n == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "r": np.nan, "r2": np.nan}

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    r = np.nan
    if n >= 2:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])

    sse = float(np.sum((y_pred - y_true) ** 2))
    sst = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
    return {"n": n, "mae": mae, "rmse": rmse, "bias": bias, "r": r, "r2": r2}


def make_case_folds(case_series: pd.Series, kfold: int, seed: int) -> pd.Series:
    cases = case_series.astype(str)
    uniq = sorted([c for c in cases.dropna().unique().tolist() if c.strip() != ""])
    if len(uniq) < kfold:
        raise ValueError(f"Not enough unique cases ({len(uniq)}) for kfold={kfold}")
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_map = {c: int(i % kfold) for i, c in enumerate(uniq)}
    return cases.map(fold_map).astype(int)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# Dataset
# ---------------------------
class HemiImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: Path, transform):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel = str(row["image_path"])
        fp = (self.image_root / rel).resolve()

        img = Image.open(fp).convert("RGB")
        x = self.transform(img)

        y = float(row["truth_PAI"])
        y = torch.tensor([y], dtype=torch.float32)

        meta = {
            "image_path": rel,
            "case_norm": str(row["case_norm"]),
            "orientation": str(row.get("orientation", "")),
        }
        return x, y, meta


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    num_workers: int
    image_size: int
    backbone: str
    freeze_backbone: bool
    nonnegative_head: bool


def build_model(backbone: str, freeze_backbone: bool, nonnegative_head: bool) -> nn.Module:
    backbone = backbone.lower().strip()

    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = m.fc.in_features
        m.fc = (
            nn.Sequential(nn.Linear(in_features, 1), nn.Softplus())
            if nonnegative_head
            else nn.Linear(in_features, 1)
        )
    elif backbone == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        in_features = m.fc.in_features
        m.fc = (
            nn.Sequential(nn.Linear(in_features, 1), nn.Softplus())
            if nonnegative_head
            else nn.Linear(in_features, 1)
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Use resnet18 or resnet34.")

    if freeze_backbone:
        for name, p in m.named_parameters():
            p.requires_grad = name.startswith("fc.")
    return m


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    losses = []

    for xb, yb, _meta in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else math.nan


@torch.no_grad()
def predict(
    model,
    loader,
    device,
    *,
    pred_min: float | None = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    model.eval()
    y_true = []
    y_pred = []
    metas: list[dict] = []

    for xb, yb, meta in loader:
        xb = xb.to(device, non_blocking=True)
        pred = model(xb).detach().cpu().numpy().reshape(-1)
        if pred_min is not None:
            pred = np.clip(pred, pred_min, None)

        yb_np = yb.detach().cpu().numpy().reshape(-1)
        y_true.append(yb_np)
        y_pred.append(pred)
        metas.extend(meta)

    return np.concatenate(y_true), np.concatenate(y_pred), metas


def _site_metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray, metas: list[dict]) -> dict:
    df = pd.DataFrame(
        {
            "case_norm": [m.get("case_norm", "") for m in metas],
            "truth_PAI": y_true,
            "pred_PAI_cnn": y_pred,
        }
    )
    # True PAI is invariant across leaf orientations, so evaluate at the case/site level.
    site = df.groupby(["case_norm"], as_index=False).agg(
        truth_PAI=("truth_PAI", "mean"),
        pred_PAI_cnn=("pred_PAI_cnn", "mean"),
    )
    return regression_metrics(site["truth_PAI"].to_numpy(), site["pred_PAI_cnn"].to_numpy())

# from torch import device
from torch.utils.data._utils.collate import default_collate

def collate_keep_meta(batch):
    xs, ys, metas = zip(*batch)
    return default_collate(xs), default_collate(ys), list(metas)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)

    p.add_argument("--out-metrics", type=Path, default=DEFAULT_OUT_METRICS)
    p.add_argument("--out-per-image", type=Path, default=DEFAULT_OUT_PER_IMAGE)
    p.add_argument("--out-per-site", type=Path, default=DEFAULT_OUT_PER_SITE)

    p.add_argument(
        "--archive-existing",
        action="store_true",
        default=True,
        help="Archive (move) existing output CSVs before writing new ones.",
    )
    p.add_argument(
        "--no-archive-existing",
        action="store_false",
        dest="archive_existing",
        help="Disable auto-archiving of existing output CSVs.",
    )
    p.add_argument(
        "--archive-dir",
        type=Path,
        default=HERE / "archive",
        help="Where to move archived output files.",
    )

    p.add_argument(
        "--progress-csv",
        type=Path,
        default=DEFAULT_PROGRESS_CSV,
        help="Write live training progress rows to this CSV (updated during training).",
    )
    p.add_argument(
        "--no-progress-csv",
        action="store_true",
        help="Disable writing the live progress CSV.",
    )

    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)  # windows-friendly default
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument(
        "--orientations",
        nargs="*",
        default=None,
        help="Optional list of orientations to include (e.g., ERECT PLANO RND). Default: include all.",
    )

    p.add_argument(
        "--aug",
        action="store_true",
        help="Enable train-time augmentation (RandomResizedCrop + optional hflip/color jitter).",
    )
    p.add_argument(
        "--aug-scale-min",
        type=float,
        default=0.7,
        help="Min scale for RandomResizedCrop when --aug is enabled (e.g., 0.7).",
    )
    p.add_argument(
        "--aug-hflip",
        action="store_true",
        help="Add RandomHorizontalFlip(p=0.5) when --aug is enabled.",
    )
    p.add_argument(
        "--aug-color-jitter",
        type=float,
        default=0.0,
        help="ColorJitter strength when --aug is enabled (0 disables; try 0.05-0.15).",
    )

    p.add_argument(
        "--pred-min",
        type=float,
        default=0.0,
        help="Minimum allowed prediction value (default 0.0 to prevent negative PAI predictions).",
    )
    p.add_argument(
        "--allow-negative-preds",
        action="store_true",
        help="Disable --pred-min clamping and allow negative predictions.",
    )

    p.add_argument(
        "--eval-each-epoch",
        action="store_true",
        help="Compute validation metrics after each epoch (slower but enables early stopping).",
    )
    p.add_argument(
        "--early-stopping",
        action="store_true",
        help="Stop training a fold when site-level validation RMSE stops improving.",
    )
    p.add_argument("--patience", type=int, default=4, help="Early-stopping patience (epochs without improvement).")
    p.add_argument("--min-delta", type=float, default=0.0, help="Minimum RMSE improvement to reset patience.")

    p.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    p.add_argument("--freeze-backbone", action="store_true", help="Train only the final FC layer.")
    p.add_argument(
        "--nonnegative-head",
        action="store_true",
        help="Use a nonnegative model head (Softplus) so predictions are intrinsically >= 0.",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for quick tests.")
    args = p.parse_args()

    set_seed(int(args.seed))

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", device)

    pred_min: float | None = None if bool(args.allow_negative_preds) else float(args.pred_min)
    if pred_min is None:
        print("Predictions: unclipped (negative values allowed)")
    else:
        print(f"Predictions: clipped to >= {pred_min:g}")

    do_aug = bool(args.aug)
    aug_scale_min = float(args.aug_scale_min)
    aug_hflip = bool(args.aug_hflip)
    aug_color_jitter = float(args.aug_color_jitter)
    if do_aug:
        print(
            "Augmentation: ON"
            f" (scale_min={aug_scale_min:g}, hflip={aug_hflip}, color_jitter={aug_color_jitter:g})"
        )
    else:
        print("Augmentation: OFF")

    df = pd.read_csv(args.index)
    # Keep labeled examples only
    df = df.dropna(subset=["image_path", "case_norm", "truth_PAI"]).copy()

    # Optional orientation filtering (true PAI does not depend on orientation).
    if args.orientations and "orientation" in df.columns:
        wanted = {str(x).upper() for x in args.orientations if str(x).strip()}
        df = df[df["orientation"].astype(str).str.upper().isin(wanted)].copy()

    if args.max_images:
        n = min(int(args.max_images), len(df))
        df = df.sample(n=n, random_state=int(args.seed)).copy()

    df = df.reset_index(drop=True)

    df["fold"] = make_case_folds(df["case_norm"], kfold=int(args.kfold), seed=int(args.seed))

    # Transforms
    # - Validation/test: deterministic resize
    # - Training: optional augmentation (RandomResizedCrop, etc.)
    image_size = int(args.image_size)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if do_aug:
        aug_ops: list = [
            transforms.RandomResizedCrop(image_size, scale=(aug_scale_min, 1.0)),
        ]
        if aug_hflip:
            aug_ops.append(transforms.RandomHorizontalFlip(p=0.5))
        if aug_color_jitter > 0:
            aug_ops.append(
                transforms.ColorJitter(
                    brightness=aug_color_jitter,
                    contrast=aug_color_jitter,
                    saturation=aug_color_jitter,
                )
            )
        train_tfm = transforms.Compose([*aug_ops, transforms.ToTensor(), normalize])
    else:
        train_tfm = test_tfm

    cfg = TrainConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.num_workers),
        image_size=int(args.image_size),
        backbone=str(args.backbone),
        freeze_backbone=bool(args.freeze_backbone),
        nonnegative_head=bool(args.nonnegative_head),
    )

    metric_rows: list[dict] = []
    per_image_rows: list[dict] = []

    # OOF prediction arrays (image-level)
    oof_pred = np.full(len(df), np.nan, dtype=np.float64)

    progress_path: Path | None = None
    if not args.no_progress_csv and args.progress_csv:
        progress_path = Path(args.progress_csv)

    # Keep the working folder clean: if the destination output files already exist,
    # move them to archive before writing new results.
    if bool(args.archive_existing):
        for pth in [Path(args.out_metrics), Path(args.out_per_image), Path(args.out_per_site)]:
            moved = _archive_existing_file(pth, Path(args.archive_dir))
            if moved is not None:
                print(f"Archived existing output: {pth.name} -> {moved}")
        if progress_path is not None:
            moved = _archive_existing_file(progress_path, Path(args.archive_dir))
            if moved is not None:
                print(f"Archived existing output: {progress_path.name} -> {moved}")

    with ProgressWriter(progress_path) as progress:
        for fold in range(int(args.kfold)):
            train_df = df[df["fold"] != fold].copy()
            test_df = df[df["fold"] == fold].copy()

            train_ds = HemiImageDataset(train_df, image_root=REPO_ROOT, transform=train_tfm)
            test_ds = HemiImageDataset(test_df, image_root=REPO_ROOT, transform=test_tfm)

            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=collate_keep_meta,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=collate_keep_meta,
            )

            model = build_model(cfg.backbone, cfg.freeze_backbone, cfg.nonnegative_head).to(device)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )

            print(f"\nFold {fold}: train={len(train_df)} test={len(test_df)}")
            progress.write(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "device": str(device),
                    "kfold": int(args.kfold),
                    "fold": int(fold),
                    "epoch": "",
                    "event": "fold_start",
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "train_mse": "",
                    "val_rmse": "",
                    "val_mae": "",
                    "val_r": "",
                    "val_r2": "",
                    "best_epoch": "",
                    "best_val_rmse": "",
                    "early_stop": "",
                    "batch_size": int(cfg.batch_size),
                    "num_workers": int(cfg.num_workers),
                    "image_size": int(cfg.image_size),
                    "backbone": str(cfg.backbone),
                    "freeze_backbone": bool(cfg.freeze_backbone),
                    "nonnegative_head": bool(cfg.nonnegative_head),
                    "aug": bool(do_aug),
                    "aug_scale_min": float(aug_scale_min) if do_aug else "",
                    "aug_hflip": bool(aug_hflip) if do_aug else "",
                    "aug_color_jitter": float(aug_color_jitter) if do_aug else "",
                    "lr": float(cfg.lr),
                    "weight_decay": float(cfg.weight_decay),
                }
            )

            do_val_each_epoch = bool(args.eval_each_epoch) or bool(args.early_stopping)
            best_val_rmse = float("inf")
            best_epoch = 0
            best_state: dict | None = None
            bad_epochs = 0
            stopped_early = False

            for epoch in range(cfg.epochs):
                loss = train_one_epoch(model, train_loader, optimizer, device)
                print(f"  epoch {epoch+1}/{cfg.epochs}  train_mse={loss:.4f}")
                progress.write(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "device": str(device),
                        "kfold": int(args.kfold),
                        "fold": int(fold),
                        "epoch": int(epoch + 1),
                        "event": "epoch_end",
                        "n_train": int(len(train_df)),
                        "n_test": int(len(test_df)),
                        "train_mse": float(loss),
                        "val_rmse": "",
                        "val_mae": "",
                        "val_r": "",
                        "val_r2": "",
                        "best_epoch": int(best_epoch) if best_epoch else "",
                        "best_val_rmse": float(best_val_rmse) if math.isfinite(best_val_rmse) else "",
                        "early_stop": "",
                        "batch_size": int(cfg.batch_size),
                        "num_workers": int(cfg.num_workers),
                        "image_size": int(cfg.image_size),
                        "backbone": str(cfg.backbone),
                        "freeze_backbone": bool(cfg.freeze_backbone),
                        "nonnegative_head": bool(cfg.nonnegative_head),
                        "aug": bool(do_aug),
                        "aug_scale_min": float(aug_scale_min) if do_aug else "",
                        "aug_hflip": bool(aug_hflip) if do_aug else "",
                        "aug_color_jitter": float(aug_color_jitter) if do_aug else "",
                        "lr": float(cfg.lr),
                        "weight_decay": float(cfg.weight_decay),
                    }
                )

                if do_val_each_epoch:
                    y_true_val, y_hat_val, metas_val = predict(model, test_loader, device, pred_min=pred_min)
                    m_site_epoch = _site_metrics_from_preds(y_true_val, y_hat_val, metas_val)
                    val_rmse = float(m_site_epoch.get("rmse", float("nan")))
                    print(
                        f"           val_site_rmse={val_rmse:.4f}  val_site_mae={float(m_site_epoch.get('mae', float('nan'))):.4f}"
                    )
                    progress.write(
                        {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "device": str(device),
                            "kfold": int(args.kfold),
                            "fold": int(fold),
                            "epoch": int(epoch + 1),
                            "event": "epoch_val_end",
                            "n_train": int(len(train_df)),
                            "n_test": int(len(test_df)),
                            "train_mse": "",
                            "val_rmse": float(m_site_epoch.get("rmse", float("nan"))),
                            "val_mae": float(m_site_epoch.get("mae", float("nan"))),
                            "val_r": float(m_site_epoch.get("r", float("nan"))),
                            "val_r2": float(m_site_epoch.get("r2", float("nan"))),
                            "best_epoch": int(best_epoch) if best_epoch else "",
                            "best_val_rmse": float(best_val_rmse) if math.isfinite(best_val_rmse) else "",
                            "early_stop": "",
                            "batch_size": int(cfg.batch_size),
                            "num_workers": int(cfg.num_workers),
                            "image_size": int(cfg.image_size),
                            "backbone": str(cfg.backbone),
                            "freeze_backbone": bool(cfg.freeze_backbone),
                            "nonnegative_head": bool(cfg.nonnegative_head),
                            "aug": bool(do_aug),
                            "aug_scale_min": float(aug_scale_min) if do_aug else "",
                            "aug_hflip": bool(aug_hflip) if do_aug else "",
                            "aug_color_jitter": float(aug_color_jitter) if do_aug else "",
                            "lr": float(cfg.lr),
                            "weight_decay": float(cfg.weight_decay),
                        }
                    )

                    if bool(args.early_stopping) and math.isfinite(val_rmse):
                        if val_rmse <= (best_val_rmse - float(args.min_delta)):
                            best_val_rmse = val_rmse
                            best_epoch = int(epoch + 1)
                            best_state = _copy_state_dict_to_cpu(model)
                            bad_epochs = 0
                        else:
                            bad_epochs += 1

                        if bad_epochs >= int(args.patience):
                            stopped_early = True
                            print(
                                f"           early_stop: no improvement for {int(args.patience)} epochs (best_epoch={best_epoch}, best_val_rmse={best_val_rmse:.4f})"
                            )
                            break

            # If early stopping was enabled, restore the best weights before final fold predictions.
            if best_state is not None:
                model.load_state_dict(best_state)
                model.to(device)

            y_true, y_hat, metas = predict(model, test_loader, device, pred_min=pred_min)

            # Store OOF predictions aligned to df index
            oof_pred[test_df.index.to_numpy()] = y_hat

            # Fold metrics (image-level)
            m_img = regression_metrics(y_true, y_hat)
            metric_rows.append(
                {"cv": "KFold", "fold": str(fold), "level": "image", "target": "truth_PAI", "pred": "cnn", **m_img}
            )

            # Per-image rows
            for i in range(len(y_hat)):
                row = {
                    "cv": "KFold",
                    "fold": str(fold),
                    "image_path": metas[i]["image_path"],
                    "case_norm": metas[i]["case_norm"],
                    "orientation": metas[i]["orientation"],
                    "truth_PAI": float(y_true[i]),
                    "pred_PAI_cnn": float(y_hat[i]),
                }
                row["error"] = row["pred_PAI_cnn"] - row["truth_PAI"]
                row["abs_error"] = abs(row["error"])
                per_image_rows.append(row)

            # Fold metrics (site/case mean across all images; true PAI is orientation-invariant)
            tmp = pd.DataFrame(per_image_rows)
            fold_img = tmp[tmp["fold"] == str(fold)].copy()
            site = fold_img.groupby(["case_norm"], as_index=False).agg(
                truth_PAI=("truth_PAI", "mean"),
                pred_PAI_cnn=("pred_PAI_cnn", "mean"),
            )
            m_site = regression_metrics(site["truth_PAI"].to_numpy(), site["pred_PAI_cnn"].to_numpy())
            metric_rows.append(
                {
                    "cv": "KFold",
                    "fold": str(fold),
                    "level": "case_mean",
                    "target": "truth_PAI",
                    "pred": "cnn",
                    **m_site,
                }
            )

            progress.write(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "device": str(device),
                    "kfold": int(args.kfold),
                    "fold": int(fold),
                    "epoch": "",
                    "event": "fold_end",
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "train_mse": "",
                    "val_rmse": float(m_site.get("rmse", float("nan"))),
                    "val_mae": float(m_site.get("mae", float("nan"))),
                    "val_r": float(m_site.get("r", float("nan"))),
                    "val_r2": float(m_site.get("r2", float("nan"))),
                    "best_epoch": int(best_epoch) if best_epoch else "",
                    "best_val_rmse": float(best_val_rmse) if math.isfinite(best_val_rmse) else "",
                    "early_stop": bool(stopped_early),
                    "batch_size": int(cfg.batch_size),
                    "num_workers": int(cfg.num_workers),
                    "image_size": int(cfg.image_size),
                    "backbone": str(cfg.backbone),
                    "freeze_backbone": bool(cfg.freeze_backbone),
                    "nonnegative_head": bool(cfg.nonnegative_head),
                    "aug": bool(do_aug),
                    "aug_scale_min": float(aug_scale_min) if do_aug else "",
                    "aug_hflip": bool(aug_hflip) if do_aug else "",
                    "aug_color_jitter": float(aug_color_jitter) if do_aug else "",
                    "lr": float(cfg.lr),
                    "weight_decay": float(cfg.weight_decay),
                }
            )

    # OOF metrics (image-level)
    df_out = df.copy()
    df_out["pred_PAI_cnn"] = oof_pred
    m_oof_img = regression_metrics(df_out["truth_PAI"].to_numpy(), df_out["pred_PAI_cnn"].to_numpy())
    metric_rows.append(
        {"cv": "KFold", "fold": "OOF", "level": "image", "target": "truth_PAI", "pred": "cnn", **m_oof_img}
    )

    # OOF metrics (site-level average, case mean)
    per_image_df = pd.DataFrame(per_image_rows)
    site_oof = per_image_df.groupby(["case_norm"], as_index=False).agg(
        truth_PAI=("truth_PAI", "mean"),
        pred_PAI_cnn=("pred_PAI_cnn", "mean"),
    )
    m_oof_site = regression_metrics(site_oof["truth_PAI"].to_numpy(), site_oof["pred_PAI_cnn"].to_numpy())
    metric_rows.append(
        {"cv": "KFold", "fold": "OOF", "level": "case_mean", "target": "truth_PAI", "pred": "cnn", **m_oof_site}
    )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.out_metrics, index=False, float_format="%.4f")

    per_image_df.to_csv(args.out_per_image, index=False, float_format="%.4f")
    site_oof.to_csv(args.out_per_site, index=False, float_format="%.4f")

    print("\nWrote metrics:", args.out_metrics)
    print("Wrote per-image predictions:", args.out_per_image)
    print("Wrote per-site predictions:", args.out_per_site)
    print("\nOOF (case_mean) metrics:", m_oof_site)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())