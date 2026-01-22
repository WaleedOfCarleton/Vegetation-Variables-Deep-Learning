from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
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


def build_model(backbone: str, freeze_backbone: bool) -> nn.Module:
    backbone = backbone.lower().strip()

    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 1)
    elif backbone == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 1)
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
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    model.eval()
    y_true = []
    y_pred = []
    metas: list[dict] = []

    for xb, yb, meta in loader:
        xb = xb.to(device, non_blocking=True)
        pred = model(xb).detach().cpu().numpy().reshape(-1)

        yb_np = yb.detach().cpu().numpy().reshape(-1)
        y_true.append(yb_np)
        y_pred.append(pred)
        metas.extend(meta)

    return np.concatenate(y_true), np.concatenate(y_pred), metas

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

    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)  # windows-friendly default
    p.add_argument("--image-size", type=int, default=224)

    p.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    p.add_argument("--freeze-backbone", action="store_true", help="Train only the final FC layer.")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for quick tests.")
    args = p.parse_args()

    set_seed(int(args.seed))

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", device)

    df = pd.read_csv(args.index)
    # Keep labeled examples only
    df = df.dropna(subset=["image_path", "case_norm", "truth_PAI"]).copy()

    # Keep only ERECT/PLANO by default if present (matches your truth coverage)
    if "orientation" in df.columns:
        df = df[df["orientation"].isin(["ERECT", "PLANO"])].copy()

    if args.max_images:
        n = min(int(args.max_images), len(df))
        df = df.sample(n=n, random_state=int(args.seed)).copy()

    df = df.reset_index(drop=True)

    df["fold"] = make_case_folds(df["case_norm"], kfold=int(args.kfold), seed=int(args.seed))

    # Transforms: resize down from 4000x4000 to something trainable
    # Note: for later, you can try RandomResizedCrop / augmentation.
    tfm = transforms.Compose(
        [
            transforms.Resize((int(args.image_size), int(args.image_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    cfg = TrainConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.num_workers),
        image_size=int(args.image_size),
        backbone=str(args.backbone),
        freeze_backbone=bool(args.freeze_backbone),
    )

    metric_rows: list[dict] = []
    per_image_rows: list[dict] = []

    # OOF prediction arrays (image-level)
    oof_pred = np.full(len(df), np.nan, dtype=np.float64)

    for fold in range(int(args.kfold)):
        train_df = df[df["fold"] != fold].copy()
        test_df = df[df["fold"] == fold].copy()

        train_ds = HemiImageDataset(train_df, image_root=HERE, transform=tfm)
        test_ds = HemiImageDataset(test_df, image_root=HERE, transform=tfm)

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

        model = build_model(cfg.backbone, cfg.freeze_backbone).to(device)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        print(f"\nFold {fold}: train={len(train_df)} test={len(test_df)}")
        for epoch in range(cfg.epochs):
            loss = train_one_epoch(model, train_loader, optimizer, device)
            print(f"  epoch {epoch+1}/{cfg.epochs}  train_mse={loss:.4f}")

        y_true, y_hat, metas = predict(model, test_loader, device)

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

        # Fold metrics (site/case+orientation mean of 10 photos)
        tmp = pd.DataFrame(per_image_rows)
        fold_img = tmp[tmp["fold"] == str(fold)].copy()
        site = (
            fold_img.groupby(["case_norm", "orientation"], as_index=False)
            .agg(truth_PAI=("truth_PAI", "mean"), pred_PAI_cnn=("pred_PAI_cnn", "mean"))
        )
        m_site = regression_metrics(site["truth_PAI"].to_numpy(), site["pred_PAI_cnn"].to_numpy())
        metric_rows.append(
            {"cv": "KFold", "fold": str(fold), "level": "case_orientation_mean", "target": "truth_PAI", "pred": "cnn", **m_site}
        )

    # OOF metrics (image-level)
    df_out = df.copy()
    df_out["pred_PAI_cnn"] = oof_pred
    m_oof_img = regression_metrics(df_out["truth_PAI"].to_numpy(), df_out["pred_PAI_cnn"].to_numpy())
    metric_rows.append(
        {"cv": "KFold", "fold": "OOF", "level": "image", "target": "truth_PAI", "pred": "cnn", **m_oof_img}
    )

    # OOF metrics (site-level average)
    per_image_df = pd.DataFrame(per_image_rows)
    site_oof = (
        per_image_df.groupby(["case_norm", "orientation"], as_index=False)
        .agg(truth_PAI=("truth_PAI", "mean"), pred_PAI_cnn=("pred_PAI_cnn", "mean"))
    )
    m_oof_site = regression_metrics(site_oof["truth_PAI"].to_numpy(), site_oof["pred_PAI_cnn"].to_numpy())
    metric_rows.append(
        {"cv": "KFold", "fold": "OOF", "level": "case_orientation_mean", "target": "truth_PAI", "pred": "cnn", **m_oof_site}
    )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.out_metrics, index=False, float_format="%.4f")

    per_image_df.to_csv(args.out_per_image, index=False, float_format="%.4f")
    site_oof.to_csv(args.out_per_site, index=False, float_format="%.4f")

    print("\nWrote metrics:", args.out_metrics)
    print("Wrote per-image predictions:", args.out_per_image)
    print("Wrote per-site predictions:", args.out_per_site)
    print("\nOOF (case_orientation_mean) metrics:", m_oof_site)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())