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


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "Simulations").exists() and (p / "shared").exists() and (p / "ml").exists():
            return p
        if (p / ".git").exists():
            return p
    return start


REPO_ROOT = _find_repo_root(HERE)
DEFAULT_INDEX = REPO_ROOT / "shared" / "dataset_index" / "image_dataset_index.csv"

DEFAULT_OUT_METRICS = HERE / "cnn_baseline_metrics.csv"
DEFAULT_OUT_PER_IMAGE = HERE / "cnn_per_image_predictions.csv"
DEFAULT_OUT_PER_SITE = HERE / "cnn_per_site_predictions.csv"
DEFAULT_PROGRESS_CSV = HERE / "cnn_training_progress.csv"


def _path_is_default(value: Path, default: Path) -> bool:
    try:
        return Path(value).resolve() == Path(default).resolve()
    except Exception:
        return str(value) == str(default)


def _apply_run_dir_defaults(args: argparse.Namespace) -> None:
    """If --run-id is provided, redirect default outputs into working/<run-id>/... .

    We only override paths that are still set to their defaults, so explicit CLI
    overrides win.
    """

    run_id = str(getattr(args, "run_id", "") or "").strip()
    if not run_id:
        return

    run_root = str(getattr(args, "run_root", "working") or "working").strip().lower()
    if run_root not in {"working", "archive"}:
        raise ValueError("--run-root must be one of: working, archive")

    run_dir = HERE / run_root / run_id
    raw_dir = run_dir / "raw"
    models_dir = run_dir / "models"
    raw_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if _path_is_default(Path(args.out_metrics), DEFAULT_OUT_METRICS):
        args.out_metrics = raw_dir / "cnn_baseline_metrics.csv"
    if _path_is_default(Path(args.out_per_image), DEFAULT_OUT_PER_IMAGE):
        args.out_per_image = raw_dir / "cnn_per_image_predictions.csv"
    if _path_is_default(Path(args.out_per_site), DEFAULT_OUT_PER_SITE):
        args.out_per_site = raw_dir / "cnn_per_site_predictions.csv"
    if (not bool(args.no_progress_csv)) and _path_is_default(Path(args.progress_csv), DEFAULT_PROGRESS_CSV):
        args.progress_csv = raw_dir / "cnn_training_progress.csv"

    if args.models_dir is None:
        args.models_dir = models_dir

    if args.split_out is None:
        args.split_out = raw_dir / "case_split.csv"

    # Keep rerun archives within the run folder (instead of polluting cnn_baseline/archive).
    if _path_is_default(Path(args.archive_dir), HERE / "archive"):
        args.archive_dir = run_dir / "_old"


def _write_latest_run_baseline(path: Path, *, run_id: str, out_metrics: Path, out_per_image: Path, out_per_site: Path, progress_csv: Path | None, models_dir: Path) -> None:
    """Write cnn_baseline/latest_run.txt in a stable, click-friendly format."""
    lines: list[str] = []
    lines.append(f"runId={run_id}")
    lines.append("")
    lines.append("[raw]")
    if progress_csv is not None:
        lines.append(f"progress={Path(progress_csv).resolve()}")
    lines.append(f"metrics={Path(out_metrics).resolve()}")
    lines.append(f"per_image={Path(out_per_image).resolve()}")
    lines.append(f"per_site={Path(out_per_site).resolve()}")
    lines.append(f"plots_dir={Path(out_metrics).resolve().parent}")
    lines.append(f"models_dir={Path(models_dir).resolve()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_run_dir(out_metrics: Path) -> Path:
    """Choose a stable folder to place auxiliary outputs (models/splits).

    We anchor to the folder containing the primary outputs so that archived runs
    keep everything together.
    """
    return Path(out_metrics).resolve().parent


def _pick_holdout_cases(
    cases: list[str],
    *,
    holdout_fraction: float,
    holdout_n_cases: int | None,
    seed: int,
) -> set[str]:
    if holdout_n_cases is not None and holdout_n_cases < 0:
        raise ValueError("--holdout-n-cases must be >= 0")

    frac = float(holdout_fraction)
    if frac < 0.0 or frac >= 1.0:
        raise ValueError("--holdout-fraction must be in [0, 1)")

    if holdout_n_cases is None:
        n = int(round(len(cases) * frac))
    else:
        n = int(holdout_n_cases)

    if n <= 0:
        return set()
    if n >= len(cases):
        raise ValueError("Holdout would consume all cases; reduce holdout size.")

    rng = np.random.default_rng(int(seed))
    picked = rng.choice(np.asarray(cases, dtype=object), size=n, replace=False)
    return {str(x) for x in picked}


def _read_case_list(path: Path) -> set[str]:
    """Reads a list of case_norm values from a txt/csv (first column)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Holdout cases file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if df.empty:
            return set()
        col = "case_norm" if "case_norm" in df.columns else df.columns[0]
        return {str(x).strip() for x in df[col].dropna().astype(str).tolist() if str(x).strip()}

    # txt/other: one case per line
    cases = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            cases.add(s)
    return cases


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


def _save_checkpoint(
    path: Path,
    *,
    state_dict: dict,
    backbone: str,
    image_size: int,
    nonnegative_head: bool,
    pred_min: float | None,
    seed: int,
    fold: int | None,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": state_dict,
        "backbone": str(backbone),
        "image_size": int(image_size),
        "nonnegative_head": bool(nonnegative_head),
        "pred_min": pred_min,
        "seed": int(seed),
        "fold": fold,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))


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

    p.add_argument(
        "--run-id",
        type=str,
        default="",
        help=(
            "Optional run id. If set, default outputs go under cnn_baseline/<run_root>/<run_id>/. "
            "Example: --run-id 20260129_pai_ep25_holdout10"
        ),
    )
    p.add_argument(
        "--run-root",
        type=str,
        default="working",
        choices=["working", "archive"],
        help="Where to place run folders when --run-id is used. Default: working",
    )

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

    p.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help=(
            "Optional case-level holdout fraction (e.g., 0.1 = hold out ~10%% of unique case_norm). "
            "Holdout cases are excluded from all CV folds and can be used for manual testing later."
        ),
    )
    p.add_argument(
        "--holdout-n-cases",
        type=int,
        default=None,
        help="Optional absolute number of unique case_norm to hold out (overrides --holdout-fraction).",
    )
    p.add_argument(
        "--holdout-cases-file",
        type=Path,
        default=None,
        help="Optional txt/csv listing case_norm values to hold out (one per line, or CSV with case_norm column).",
    )
    p.add_argument(
        "--split-out",
        type=Path,
        default=None,
        help="Optional path to write the case split CSV (case_norm, split). Default: next to --out-metrics.",
    )

    p.add_argument(
        "--save-models",
        action="store_true",
        default=True,
        help="Save fold checkpoints (and an optional final model) for later inference.",
    )
    p.add_argument(
        "--no-save-models",
        action="store_false",
        dest="save_models",
        help="Disable saving checkpoints.",
    )
    p.add_argument(
        "--save-final-model",
        action="store_true",
        default=True,
        help="Also train a final model on all non-holdout data and save it.",
    )
    p.add_argument(
        "--no-save-final-model",
        action="store_false",
        dest="save_final_model",
        help="Disable training/saving a final model on all non-holdout data.",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Where to save model checkpoints/split files. Default: <out-metrics folder>/models/.",
    )

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

    # If a run id is provided, redirect default outputs into working/<run-id>/...
    _apply_run_dir_defaults(args)

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

    # ---------------------------
    # Case-level holdout split
    # ---------------------------
    uniq_cases = sorted(
        [c for c in df["case_norm"].astype(str).dropna().unique().tolist() if str(c).strip()]
    )
    holdout_cases: set[str] = set()
    if args.holdout_cases_file is not None:
        holdout_cases = _read_case_list(Path(args.holdout_cases_file))
    else:
        holdout_cases = _pick_holdout_cases(
            uniq_cases,
            holdout_fraction=float(args.holdout_fraction),
            holdout_n_cases=(int(args.holdout_n_cases) if args.holdout_n_cases is not None else None),
            seed=int(args.seed),
        )

    df["split"] = np.where(df["case_norm"].astype(str).isin(holdout_cases), "holdout", "train")

    split_out = Path(args.split_out) if args.split_out is not None else (_default_run_dir(Path(args.out_metrics)) / "case_split.csv")
    split_out.parent.mkdir(parents=True, exist_ok=True)
    (
        df[["case_norm", "split"]]
        .drop_duplicates()
        .sort_values(["split", "case_norm"], ascending=[True, True])
        .to_csv(split_out, index=False)
    )
    if holdout_cases:
        print(f"Holdout: {len(holdout_cases)} case(s) excluded from training. Split file: {split_out}")
    else:
        print(f"Holdout: none (split file written): {split_out}")

    # Only assign folds over the training split (prevents leakage)
    train_mask = df["split"].astype(str) == "train"
    df.loc[train_mask, "fold"] = make_case_folds(df.loc[train_mask, "case_norm"], kfold=int(args.kfold), seed=int(args.seed))
    df.loc[~train_mask, "fold"] = -1

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

    # Holdout prediction accumulator (ensemble average over fold models)
    holdout_df = df[df["split"].astype(str) == "holdout"].copy()
    holdout_pred_sum: np.ndarray | None = None
    holdout_pred_count: int = 0
    if len(holdout_df) > 0:
        holdout_pred_sum = np.zeros(len(holdout_df), dtype=np.float64)

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
            # CV is done only within the train split.
            train_df = df[(df["split"].astype(str) == "train") & (df["fold"].astype(int) != fold)].copy()
            test_df = df[(df["split"].astype(str) == "train") & (df["fold"].astype(int) == fold)].copy()

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

            # Save fold checkpoint for later inference/testing.
            if bool(args.save_models):
                run_dir = _default_run_dir(Path(args.out_metrics))
                models_dir = Path(args.models_dir) if args.models_dir is not None else (run_dir / "models")
                ckpt_path = models_dir / f"pai_model_fold{fold}.pth"
                _save_checkpoint(
                    ckpt_path,
                    state_dict=_copy_state_dict_to_cpu(model),
                    backbone=cfg.backbone,
                    image_size=cfg.image_size,
                    nonnegative_head=cfg.nonnegative_head,
                    pred_min=pred_min,
                    seed=int(args.seed),
                    fold=int(fold),
                    extra={"target": "truth_PAI"},
                )

            y_true, y_hat, metas = predict(model, test_loader, device, pred_min=pred_min)

            # Store OOF predictions aligned to df index
            oof_pred[test_df.index.to_numpy()] = y_hat

            # Optionally also predict the holdout set with this fold model (for an ensemble).
            if holdout_pred_sum is not None and len(holdout_df) > 0:
                holdout_ds = HemiImageDataset(holdout_df, image_root=REPO_ROOT, transform=test_tfm)
                holdout_loader = DataLoader(
                    holdout_ds,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                    pin_memory=(device.type == "cuda"),
                    collate_fn=collate_keep_meta,
                )
                _y_true_h, _y_hat_h, _metas_h = predict(model, holdout_loader, device, pred_min=pred_min)
                holdout_pred_sum += _y_hat_h.astype(np.float64)
                holdout_pred_count += 1

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
                    "split": "train",
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
    train_only = df_out[df_out["split"].astype(str) == "train"].copy()
    m_oof_img = regression_metrics(train_only["truth_PAI"].to_numpy(), train_only["pred_PAI_cnn"].to_numpy())
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

    # Append holdout predictions (ensemble over folds) if configured.
    if holdout_pred_sum is not None and holdout_pred_count > 0:
        holdout_df_out = holdout_df.copy().reset_index(drop=True)
        holdout_df_out["pred_PAI_cnn"] = (holdout_pred_sum / float(holdout_pred_count)).astype(np.float64)
        # Add to per-image table with fold='HOLDOUT' (predictions are ensemble outputs).
        for i in range(len(holdout_df_out)):
            per_image_rows.append(
                {
                    "cv": "Holdout",
                    "fold": "ENS",
                    "image_path": str(holdout_df_out.loc[i, "image_path"]),
                    "case_norm": str(holdout_df_out.loc[i, "case_norm"]),
                    "orientation": str(holdout_df_out.loc[i, "orientation"]) if "orientation" in holdout_df_out.columns else "",
                    "split": "holdout",
                    "truth_PAI": float(holdout_df_out.loc[i, "truth_PAI"]),
                    "pred_PAI_cnn": float(holdout_df_out.loc[i, "pred_PAI_cnn"]),
                    "error": float(holdout_df_out.loc[i, "pred_PAI_cnn"] - holdout_df_out.loc[i, "truth_PAI"]),
                    "abs_error": float(abs(holdout_df_out.loc[i, "pred_PAI_cnn"] - holdout_df_out.loc[i, "truth_PAI"])),
                }
            )

        # Holdout metrics at case_mean level
        holdout_site = (
            holdout_df_out.groupby(["case_norm"], as_index=False)
            .agg(truth_PAI=("truth_PAI", "mean"), pred_PAI_cnn=("pred_PAI_cnn", "mean"))
            .copy()
        )
        m_hold_site = regression_metrics(holdout_site["truth_PAI"].to_numpy(), holdout_site["pred_PAI_cnn"].to_numpy())
        metric_rows.append(
            {"cv": "Holdout", "fold": "ENS", "level": "case_mean", "target": "truth_PAI", "pred": "cnn", **m_hold_site}
        )

        m_hold_img = regression_metrics(
            holdout_df_out["truth_PAI"].to_numpy(),
            holdout_df_out["pred_PAI_cnn"].to_numpy(),
        )
        metric_rows.append(
            {"cv": "Holdout", "fold": "ENS", "level": "image", "target": "truth_PAI", "pred": "cnn", **m_hold_img}
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.out_metrics, index=False, float_format="%.4f")

    per_image_df = pd.DataFrame(per_image_rows)
    per_image_df.to_csv(args.out_per_image, index=False, float_format="%.4f")
    # Per-site output for the CV (OOF) predictions only (train split)
    site_oof.to_csv(args.out_per_site, index=False, float_format="%.4f")

    # Save a final model trained on all non-holdout data (optional)
    if bool(args.save_models) and bool(args.save_final_model):
        run_dir = _default_run_dir(Path(args.out_metrics))
        models_dir = Path(args.models_dir) if args.models_dir is not None else (run_dir / "models")

        final_train_df = df[df["split"].astype(str) == "train"].copy()
        final_ds = HemiImageDataset(final_train_df, image_root=REPO_ROOT, transform=train_tfm)
        final_loader = DataLoader(
            final_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_keep_meta,
        )

        final_model = build_model(cfg.backbone, cfg.freeze_backbone, cfg.nonnegative_head).to(device)
        final_opt = torch.optim.AdamW(
            [p for p in final_model.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        print(f"\nTraining final model on all train data: n={len(final_train_df)}")
        for epoch in range(cfg.epochs):
            loss = train_one_epoch(final_model, final_loader, final_opt, device)
            print(f"  final epoch {epoch+1}/{cfg.epochs}  train_mse={loss:.4f}")

        final_ckpt = models_dir / "pai_model_final_train.pth"
        _save_checkpoint(
            final_ckpt,
            state_dict=_copy_state_dict_to_cpu(final_model),
            backbone=cfg.backbone,
            image_size=cfg.image_size,
            nonnegative_head=cfg.nonnegative_head,
            pred_min=pred_min,
            seed=int(args.seed),
            fold=None,
            extra={"target": "truth_PAI", "trained_on": "train_split_only"},
        )
        print(f"Saved final model: {final_ckpt}")

    print("\nWrote metrics:", args.out_metrics)
    print("Wrote per-image predictions:", args.out_per_image)
    print("Wrote per-site predictions:", args.out_per_site)
    print("\nOOF (case_mean) metrics:", m_oof_site)

    # Update latest_run pointer for convenience
    run_id = str(args.run_id).strip()
    if run_id:
        progress_path_out: Path | None = None
        if not bool(args.no_progress_csv) and args.progress_csv:
            progress_path_out = Path(args.progress_csv)
        models_dir = Path(args.models_dir) if args.models_dir is not None else (_default_run_dir(Path(args.out_metrics)) / "models")
        _write_latest_run_baseline(
            HERE / "latest_run.txt",
            run_id=run_id,
            out_metrics=Path(args.out_metrics),
            out_per_image=Path(args.out_per_image),
            out_per_site=Path(args.out_per_site),
            progress_csv=progress_path_out,
            models_dir=models_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())