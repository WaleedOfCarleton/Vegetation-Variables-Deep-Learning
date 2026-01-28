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

DEFAULT_OUT_METRICS = HERE / "cnn_hinge_metrics.csv"
DEFAULT_OUT_PER_IMAGE = HERE / "cnn_hinge_per_image_predictions.csv"
DEFAULT_OUT_PER_SITE = HERE / "cnn_hinge_per_site_predictions.csv"
DEFAULT_PROGRESS_CSV = HERE / "cnn_hinge_training_progress.csv"

TARGET_COL = "truth_PAIe_hinge"
PRED_COL = "pred_PAIe_hinge_cnn"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _archive_existing_file(path: Path, archive_dir: Path) -> Path | None:
    if not path.exists():
        return None

    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    dest = archive_dir / f"{path.stem}_{stamp}{path.suffix}"
    if dest.exists():
        dest = archive_dir / f"{path.stem}_{stamp}_{os.getpid()}{path.suffix}"
    shutil.move(str(path), str(dest))
    return dest


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
                "run_id",
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
            pass

    def __exit__(self, exc_type, exc, tb):
        if self._fp is not None:
            self._fp.close()
        self._fp = None
        self._writer = None
        return False


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
    rmse = float(np.sqrt(np.mean(err**2)))
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

        y = float(row[TARGET_COL])
        y = torch.tensor([y], dtype=torch.float32)

        meta = {
            "image_path": rel,
            "case_norm": str(row.get("case_norm", "")),
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
        m.fc = nn.Sequential(nn.Linear(in_features, 1), nn.Softplus()) if nonnegative_head else nn.Linear(in_features, 1)
    elif backbone == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        in_features = m.fc.in_features
        m.fc = nn.Sequential(nn.Linear(in_features, 1), nn.Softplus()) if nonnegative_head else nn.Linear(in_features, 1)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Use resnet18 or resnet34.")

    if freeze_backbone:
        for name, p in m.named_parameters():
            p.requires_grad = name.startswith("fc.")
    return m


def train_one_epoch(model, loader, optimizer, device, *, on_batch=None) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    losses = []

    n_batches = len(loader) if hasattr(loader, "__len__") else None
    log_every = 0
    if n_batches and n_batches > 0:
        log_every = max(1, int(n_batches // 10))

    for batch_idx, (xb, yb, _meta) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

        if on_batch is not None and n_batches and n_batches > 0:
            if batch_idx == 0 or (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_batches:
                try:
                    on_batch(int(batch_idx), int(n_batches), float(np.mean(losses)))
                except Exception:
                    pass

    return float(np.mean(losses)) if losses else math.nan


@torch.no_grad()
def predict(model, loader, device, *, pred_min: float | None = 0.0) -> tuple[np.ndarray, np.ndarray, list[dict]]:
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


def _case_orientation_metrics(y_true: np.ndarray, y_pred: np.ndarray, metas: list[dict]) -> dict:
    df = pd.DataFrame(
        {
            "case_norm": [m.get("case_norm", "") for m in metas],
            "orientation": [m.get("orientation", "") for m in metas],
            TARGET_COL: y_true,
            PRED_COL: y_pred,
        }
    )
    site = df.groupby(["case_norm", "orientation"], as_index=False).agg(
        **{TARGET_COL: (TARGET_COL, "mean"), PRED_COL: (PRED_COL, "mean")}
    )
    return regression_metrics(site[TARGET_COL].to_numpy(), site[PRED_COL].to_numpy())


def main() -> int:
    p = argparse.ArgumentParser(description="Train CNN to predict hinge-region PAIe (VZA ~57.3°).")
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)

    p.add_argument("--out-metrics", type=Path, default=DEFAULT_OUT_METRICS)
    p.add_argument("--out-per-image", type=Path, default=DEFAULT_OUT_PER_IMAGE)
    p.add_argument("--out-per-site", type=Path, default=DEFAULT_OUT_PER_SITE)

    p.add_argument("--archive-existing", action="store_true", default=True)
    p.add_argument("--no-archive-existing", action="store_false", dest="archive_existing")
    p.add_argument("--archive-dir", type=Path, default=HERE / "archive")

    p.add_argument("--progress-csv", type=Path, default=DEFAULT_PROGRESS_CSV)
    p.add_argument("--no-progress-csv", action="store_true")

    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--orientations", nargs="*", default=None, help="Optional list of orientations to include.")

    p.add_argument("--backbone", type=str, default="resnet18")
    p.add_argument("--freeze-backbone", action="store_true")

    p.add_argument("--nonnegative-head", action="store_true", help="Use Softplus output head (>=0).")
    p.add_argument("--pred-min", type=float, default=0.0, help="Clamp predictions to be >= this value.")
    p.add_argument("--allow-negative-preds", action="store_true", help="Disable pred clamping.")

    p.add_argument("--aug", action="store_true", help="Enable train-time augmentation.")
    p.add_argument("--aug-scale-min", type=float, default=0.9)
    p.add_argument("--aug-hflip", action="store_true")
    p.add_argument("--aug-color-jitter", type=float, default=0.0)

    p.add_argument("--early-stop", type=int, default=6, help="Patience in epochs; 0 disables.")

    p.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional run id to suffix outputs (e.g. 20260127_120000_paie_hinge).",
    )

    args = p.parse_args()

    set_seed(args.seed)

    index_path = Path(args.index)
    df = pd.read_csv(index_path)

    required = {"image_path", "case_norm", "orientation", TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"Index missing columns {sorted(missing)}. "
            f"Rebuild dataset index to include {TARGET_COL} (run dataset_index/build_image_dataset_index.py)."
        )

    if args.orientations:
        want = {str(x).upper().strip() for x in args.orientations}
        df = df[df["orientation"].astype(str).str.upper().isin(want)].copy()

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=["image_path", "case_norm", "orientation", TARGET_COL]).copy()

    df["fold"] = make_case_folds(df["case_norm"], kfold=int(args.kfold), seed=int(args.seed))

    # Naming
    run_id = args.run_id.strip()
    suffix = f"_{run_id}" if run_id else ""
    out_metrics = Path(args.out_metrics)
    out_per_image = Path(args.out_per_image)
    out_per_site = Path(args.out_per_site)

    if suffix:
        out_metrics = out_metrics.with_name(out_metrics.stem + suffix + out_metrics.suffix)
        out_per_image = out_per_image.with_name(out_per_image.stem + suffix + out_per_image.suffix)
        out_per_site = out_per_site.with_name(out_per_site.stem + suffix + out_per_site.suffix)

    if args.archive_existing:
        for pth in [out_metrics, out_per_image, out_per_site]:
            _archive_existing_file(pth, Path(args.archive_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    if args.aug:
        aug_ops = [
            transforms.RandomResizedCrop(args.image_size, scale=(float(args.aug_scale_min), 1.0)),
        ]
        if args.aug_hflip:
            aug_ops.append(transforms.RandomHorizontalFlip(p=0.5))
        if float(args.aug_color_jitter) > 0:
            cj = float(args.aug_color_jitter)
            aug_ops.append(transforms.ColorJitter(brightness=cj, contrast=cj, saturation=cj, hue=min(0.5, cj)))
        aug_ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_tfm = transforms.Compose(aug_ops)
    else:
        train_tfm = transforms.Compose(
            [
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    test_tfm = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    from torch.utils.data._utils.collate import default_collate

    def collate_keep_meta(batch):
        xs, ys, metas = zip(*batch)
        return default_collate(xs), default_collate(ys), list(metas)

    # Keep a single progress CSV by default (much easier to watch live).
    # We include run_id in each row so multiple runs can share the file.
    progress_path = None if args.no_progress_csv else Path(args.progress_csv)

    metrics_rows: list[dict] = []
    per_image_rows: list[dict] = []

    pred_min = None if args.allow_negative_preds else float(args.pred_min)

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

    with ProgressWriter(progress_path) as pw:
        for fold in range(int(args.kfold)):
            tr = df[df["fold"] != fold].copy()
            te = df[df["fold"] == fold].copy()

            train_ds = HemiImageDataset(tr, image_root=REPO_ROOT, transform=train_tfm)
            test_ds = HemiImageDataset(te, image_root=REPO_ROOT, transform=test_tfm)

            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=collate_keep_meta,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=collate_keep_meta,
            )

            model = build_model(cfg.backbone, cfg.freeze_backbone, cfg.nonnegative_head).to(device)
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )

            best_rmse = math.inf
            best_state = None
            best_epoch = -1
            patience = int(args.early_stop)
            bad = 0

            if pw.path is not None:
                pw.write(
                    {
                        "run_id": run_id,
                        "timestamp_utc": _utc_stamp(),
                        "device": str(device),
                        "kfold": int(args.kfold),
                        "fold": int(fold),
                        "epoch": -1,
                        "event": "fold_start",
                        "n_train": int(len(tr)),
                        "n_test": int(len(te)),
                        "train_mse": math.nan,
                        "val_rmse": math.nan,
                        "val_mae": math.nan,
                        "val_r": math.nan,
                        "val_r2": math.nan,
                        "best_epoch": int(best_epoch),
                        "best_val_rmse": float(best_rmse) if math.isfinite(best_rmse) else math.nan,
                        "early_stop": int(patience),
                        "batch_size": cfg.batch_size,
                        "num_workers": cfg.num_workers,
                        "image_size": cfg.image_size,
                        "backbone": cfg.backbone,
                        "freeze_backbone": cfg.freeze_backbone,
                        "nonnegative_head": cfg.nonnegative_head,
                        "aug": bool(args.aug),
                        "aug_scale_min": float(args.aug_scale_min),
                        "aug_hflip": bool(args.aug_hflip),
                        "aug_color_jitter": float(args.aug_color_jitter),
                        "lr": cfg.lr,
                        "weight_decay": cfg.weight_decay,
                    }
                )

            for epoch in range(cfg.epochs):
                if pw.path is not None:
                    pw.write(
                        {
                            "run_id": run_id,
                            "timestamp_utc": _utc_stamp(),
                            "device": str(device),
                            "kfold": int(args.kfold),
                            "fold": int(fold),
                            "epoch": int(epoch),
                            "event": "epoch_start",
                            "n_train": int(len(tr)),
                            "n_test": int(len(te)),
                            "train_mse": math.nan,
                            "val_rmse": math.nan,
                            "val_mae": math.nan,
                            "val_r": math.nan,
                            "val_r2": math.nan,
                            "best_epoch": int(best_epoch),
                            "best_val_rmse": float(best_rmse) if math.isfinite(best_rmse) else math.nan,
                            "early_stop": int(patience),
                            "batch_size": cfg.batch_size,
                            "num_workers": cfg.num_workers,
                            "image_size": cfg.image_size,
                            "backbone": cfg.backbone,
                            "freeze_backbone": cfg.freeze_backbone,
                            "nonnegative_head": cfg.nonnegative_head,
                            "aug": bool(args.aug),
                            "aug_scale_min": float(args.aug_scale_min),
                            "aug_hflip": bool(args.aug_hflip),
                            "aug_color_jitter": float(args.aug_color_jitter),
                            "lr": cfg.lr,
                            "weight_decay": cfg.weight_decay,
                        }
                    )

                def _on_batch(batch_idx: int, n_batches: int, avg_loss: float) -> None:
                    if pw.path is None:
                        return
                    pw.write(
                        {
                            "run_id": run_id,
                            "timestamp_utc": _utc_stamp(),
                            "device": str(device),
                            "kfold": int(args.kfold),
                            "fold": int(fold),
                            "epoch": int(epoch),
                            "event": "batch",
                            "n_train": int(len(tr)),
                            "n_test": int(len(te)),
                            "train_mse": float(avg_loss),
                            "val_rmse": math.nan,
                            "val_mae": math.nan,
                            "val_r": math.nan,
                            "val_r2": math.nan,
                            "best_epoch": int(best_epoch),
                            "best_val_rmse": float(best_rmse) if math.isfinite(best_rmse) else math.nan,
                            "early_stop": int(patience),
                            "batch_size": cfg.batch_size,
                            "num_workers": cfg.num_workers,
                            "image_size": cfg.image_size,
                            "backbone": cfg.backbone,
                            "freeze_backbone": cfg.freeze_backbone,
                            "nonnegative_head": cfg.nonnegative_head,
                            "aug": bool(args.aug),
                            "aug_scale_min": float(args.aug_scale_min),
                            "aug_hflip": bool(args.aug_hflip),
                            "aug_color_jitter": float(args.aug_color_jitter),
                            "lr": cfg.lr,
                            "weight_decay": cfg.weight_decay,
                        }
                    )

                train_mse = train_one_epoch(model, train_loader, optimizer, device, on_batch=_on_batch)
                y_true, y_hat, metas = predict(model, test_loader, device, pred_min=pred_min)

                m_img = regression_metrics(y_true, y_hat)
                m_site = _case_orientation_metrics(y_true, y_hat, metas)

                if pw.path is not None:
                    pw.write(
                        {
                            "run_id": run_id,
                            "timestamp_utc": _utc_stamp(),
                            "device": str(device),
                            "kfold": int(args.kfold),
                            "fold": int(fold),
                            "epoch": int(epoch),
                            "event": "epoch",
                            "n_train": int(len(tr)),
                            "n_test": int(len(te)),
                            "train_mse": float(train_mse),
                            "val_rmse": float(m_site.get("rmse", math.nan)),
                            "val_mae": float(m_site.get("mae", math.nan)),
                            "val_r": float(m_site.get("r", math.nan)),
                            "val_r2": float(m_site.get("r2", math.nan)),
                            "best_epoch": int(best_epoch),
                            "best_val_rmse": float(best_rmse) if math.isfinite(best_rmse) else math.nan,
                            "early_stop": int(patience),
                            "batch_size": cfg.batch_size,
                            "num_workers": cfg.num_workers,
                            "image_size": cfg.image_size,
                            "backbone": cfg.backbone,
                            "freeze_backbone": cfg.freeze_backbone,
                            "nonnegative_head": cfg.nonnegative_head,
                            "aug": bool(args.aug),
                            "aug_scale_min": float(args.aug_scale_min),
                            "aug_hflip": bool(args.aug_hflip),
                            "aug_color_jitter": float(args.aug_color_jitter),
                            "lr": cfg.lr,
                            "weight_decay": cfg.weight_decay,
                        }
                    )

                val_rmse = float(m_site.get("rmse", math.nan))
                if math.isfinite(val_rmse) and val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1

                if patience > 0 and bad >= patience:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            y_true, y_hat, metas = predict(model, test_loader, device, pred_min=pred_min)
            m_img = regression_metrics(y_true, y_hat)
            m_site = _case_orientation_metrics(y_true, y_hat, metas)

            metrics_rows.append({"cv": "KFold", "fold": str(fold), "level": "image", "target": TARGET_COL, "pred": "cnn", **m_img})
            metrics_rows.append(
                {
                    "cv": "KFold",
                    "fold": str(fold),
                    "level": "case_orientation_mean",
                    "target": TARGET_COL,
                    "pred": "cnn",
                    **m_site,
                }
            )

            for i, meta in enumerate(metas):
                row = {
                    "cv": "KFold",
                    "fold": int(fold),
                    "image_path": meta.get("image_path", ""),
                    "case_norm": meta.get("case_norm", ""),
                    "orientation": meta.get("orientation", ""),
                    TARGET_COL: float(y_true[i]),
                    PRED_COL: float(y_hat[i]),
                }
                row["error"] = row[PRED_COL] - row[TARGET_COL]
                row["abs_error"] = abs(row["error"])
                per_image_rows.append(row)

    per_image_df = pd.DataFrame(per_image_rows)
    per_site_df = (
        per_image_df.groupby(["case_norm", "orientation"], as_index=False)
        .agg(**{TARGET_COL: (TARGET_COL, "mean"), PRED_COL: (PRED_COL, "mean")})
        .copy()
    )

    # OOF metrics
    m_oof_img = regression_metrics(per_image_df[TARGET_COL].to_numpy(), per_image_df[PRED_COL].to_numpy())
    m_oof_site = regression_metrics(per_site_df[TARGET_COL].to_numpy(), per_site_df[PRED_COL].to_numpy())
    metrics_rows.append({"cv": "KFold", "fold": "OOF", "level": "image", "target": TARGET_COL, "pred": "cnn", **m_oof_img})
    metrics_rows.append(
        {"cv": "KFold", "fold": "OOF", "level": "case_orientation_mean", "target": TARGET_COL, "pred": "cnn", **m_oof_site}
    )

    metrics_df = pd.DataFrame(metrics_rows)

    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_per_image.parent.mkdir(parents=True, exist_ok=True)
    out_per_site.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(out_metrics, index=False, float_format="%.6f")
    per_image_df.to_csv(out_per_image, index=False, float_format="%.4f")
    per_site_df.to_csv(out_per_site, index=False, float_format="%.4f")

    print("Wrote:")
    print("  metrics  :", out_metrics)
    print("  per-image:", out_per_image)
    print("  per-site :", out_per_site)
    print("\nOOF (case_orientation_mean) metrics:", m_oof_site)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
