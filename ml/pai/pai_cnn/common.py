from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

try:
    from torchvision import transforms
    from torchvision.models import ResNet18_Weights, resnet18
except Exception as exc:  # pragma: no cover
    raise ImportError("torchvision is required for the CNN scripts. Install torch + torchvision.") from exc


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class IndexRow:
    image_path: str
    case_norm: str
    orientation: str
    simulation_set: str
    truth_value: float


def get_repo_root_from_any_ml_file(ml_file: str | Path) -> Path:
    """Find the repo root starting from any file under ml/.

    This is intentionally robust to nested folders like ml/pai/... .
    """

    p = Path(ml_file).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "Simulations").exists() and (parent / "shared").exists() and (parent / "ml").exists():
            return parent
        if (parent / ".git").exists():
            return parent
    return p.parents[-1]


def read_index_csv(
    index_csv: Path,
    target_col: str = "truth_PAI",
    orientation: Optional[str] = None,
    simulation_set: Optional[str] = None,
) -> list[IndexRow]:
    rows: list[IndexRow] = []

    with index_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_cols = {"image_path", "case_norm", "orientation", "simulation_set", target_col} - set(
            reader.fieldnames or []
        )
        if missing_cols:
            raise ValueError(f"Index CSV is missing columns: {sorted(missing_cols)}. Found: {reader.fieldnames}")

        for r in reader:
            if orientation is not None and r["orientation"] != orientation:
                continue
            if simulation_set is not None and r["simulation_set"] != simulation_set:
                continue

            raw = (r.get(target_col) or "").strip()
            if raw == "":
                continue
            try:
                y = float(raw)
            except ValueError:
                token = raw.split("+")[0].strip()
                y = float(token)

            rows.append(
                IndexRow(
                    image_path=r["image_path"],
                    case_norm=r["case_norm"],
                    orientation=r["orientation"],
                    simulation_set=r["simulation_set"],
                    truth_value=y,
                )
            )

    if not rows:
        raise ValueError("No usable rows found after filtering; check orientation/simulation_set/target_col.")
    return rows


def split_cases(rows: list[IndexRow], val_fraction: float, seed: int) -> tuple[list[IndexRow], list[IndexRow]]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1")

    cases = sorted({r.case_norm for r in rows})
    rng = random.Random(seed)
    rng.shuffle(cases)

    n_val = max(1, int(round(len(cases) * val_fraction)))
    val_cases = set(cases[:n_val])

    train_rows = [r for r in rows if r.case_norm not in val_cases]
    val_rows = [r for r in rows if r.case_norm in val_cases]
    return train_rows, val_rows


def split_cases_kfold(rows: list[IndexRow], k: int, fold: int, seed: int) -> tuple[list[IndexRow], list[IndexRow]]:
    if k < 2:
        raise ValueError("k must be >= 2")
    if not (0 <= fold < k):
        raise ValueError("fold must be in [0, k-1]")

    cases = sorted({r.case_norm for r in rows})
    rng = random.Random(seed)
    rng.shuffle(cases)

    folds: list[list[str]] = [[] for _ in range(k)]
    for i, c in enumerate(cases):
        folds[i % k].append(c)

    val_cases = set(folds[fold])
    train_rows = [r for r in rows if r.case_norm not in val_cases]
    val_rows = [r for r in rows if r.case_norm in val_cases]
    return train_rows, val_rows


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class PaiIndexDataset(Dataset):
    def __init__(self, rows: list[IndexRow], repo_root: Path, transform: transforms.Compose) -> None:
        self._rows = rows
        self._repo_root = repo_root
        self._transform = transform

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int):
        r = self._rows[idx]
        image_path = (self._repo_root / r.image_path).resolve()
        img = Image.open(image_path).convert("RGB")
        x = self._transform(img)
        y = torch.tensor([r.truth_value], dtype=torch.float32)
        return x, y, r


def collate_keep_meta(batch):
    xs, ys, metas = zip(*batch)
    return default_collate(xs), default_collate(ys), list(metas)


def build_model(pretrained: bool) -> nn.Module:
    if pretrained:
        try:
            weights = ResNet18_Weights.DEFAULT
        except Exception:
            weights = None
    else:
        weights = None

    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: Iterable, device: torch.device) -> dict:
    model.eval()

    sum_abs = 0.0
    sum_sq = 0.0
    n = 0

    per_case_pred_sum: dict[str, float] = {}
    per_case_count: dict[str, int] = {}
    per_case_truth: dict[str, float] = {}

    for xb, yb, meta in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        err = pred - yb

        sum_abs += err.abs().sum().item()
        sum_sq += (err * err).sum().item()
        n += yb.numel()

        preds = pred.squeeze(1).detach().cpu().tolist()
        truths = yb.squeeze(1).detach().cpu().tolist()
        metas = list(meta)
        for p, t, m in zip(preds, truths, metas):
            per_case_pred_sum[m.case_norm] = per_case_pred_sum.get(m.case_norm, 0.0) + float(p)
            per_case_count[m.case_norm] = per_case_count.get(m.case_norm, 0) + 1
            per_case_truth[m.case_norm] = float(t)

    mae = sum_abs / max(1, n)
    rmse = math.sqrt(sum_sq / max(1, n))

    case_abs = 0.0
    case_n = 0
    for case, s in per_case_pred_sum.items():
        mean_pred = s / per_case_count[case]
        case_abs += abs(mean_pred - per_case_truth[case])
        case_n += 1

    return {
        "mae_image": mae,
        "rmse_image": rmse,
        "mae_case": (case_abs / max(1, case_n)),
        "n_images": n,
        "n_cases": case_n,
    }


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
