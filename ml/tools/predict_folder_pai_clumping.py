from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

# Reuse PAI/Clumping CNN utilities.
sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "pai").resolve()))
from pai_cnn.common import build_model, build_transforms  # noqa: E402


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    rel_path: str
    case: str


class ImageFolderDataset(Dataset):
    def __init__(self, items: list[ImageInfo], transform) -> None:
        self._items = items
        self._transform = transform

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        info = self._items[idx]
        img = Image.open(info.path).convert("RGB")
        x = self._transform(img)
        return x, info


def collate_keep_info(batch):
    xs, infos = zip(*batch)
    return default_collate(xs), list(infos)


def _list_images(root: Path) -> list[ImageInfo]:
    items: list[ImageInfo] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name.lower() == "thumbs.db":
            continue
        if p.suffix.lower() not in _IMAGE_EXTS:
            continue
        rel = p.relative_to(root)
        # If there are nested folders, group by the immediate parent folder; otherwise, use the root name.
        case = rel.parent.as_posix() if rel.parent.as_posix() not in {"", "."} else root.name
        items.append(ImageInfo(path=p, rel_path=rel.as_posix(), case=case))
    if not items:
        raise FileNotFoundError(f"No images found under {root} (looked for: {sorted(_IMAGE_EXTS)})")
    return items


def _load_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    img_size = int(cfg.get("img_size", 224))
    pretrained = bool(cfg.get("pretrained", False))

    model = build_model(pretrained=pretrained)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, img_size


@torch.no_grad()
def _predict(model: torch.nn.Module, img_size: int, items: list[ImageInfo], device: torch.device, *, batch_size: int, num_workers: int) -> dict[str, float]:
    tf = build_transforms(img_size, train=False)
    ds = ImageFolderDataset(items, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_keep_info)

    preds: dict[str, float] = {}
    for xb, infos in loader:
        xb = xb.to(device)
        out = model(xb).squeeze(1).detach().cpu().tolist()
        for pred, info in zip(out, infos):
            preds[info.rel_path] = float(pred)
    return preds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PAI and Clumping checkpoints on all images in a folder (recursively).")
    p.add_argument("--images-dir", required=True, help="Folder containing images or case subfolders")
    p.add_argument("--pai-checkpoint", required=True, help="Checkpoint (.pt) for PAI prediction")
    p.add_argument("--clumping-checkpoint", default=None, help="Optional checkpoint (.pt) for clumping prediction")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--out-csv", default=None, help="Output CSV path (default: <images-dir>/predictions_pai_clumping.csv)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    images_dir = Path(args.images_dir).resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"images-dir not found: {images_dir}")

    items = _list_images(images_dir)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: cpu")

    print(f"Discovered {len(items)} images across {len(set(i.case for i in items))} case(s)")

    pai_model, pai_img_size = _load_model(Path(args.pai_checkpoint), device)
    pai_preds = _predict(pai_model, pai_img_size, items, device, batch_size=args.batch_size, num_workers=args.num_workers)

    clumping_preds: dict[str, float] = {}
    if args.clumping_checkpoint:
        cl_model, cl_img_size = _load_model(Path(args.clumping_checkpoint), device)
        clumping_preds = _predict(cl_model, cl_img_size, items, device, batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        print("No clumping checkpoint provided; clumping column will be empty.")

    out_csv = Path(args.out_csv) if args.out_csv else (images_dir / "predictions_pai_clumping.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    import csv

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "image_path", "pred_pai", "pred_clumping"],
        )
        writer.writeheader()
        for info in items:
            writer.writerow(
                {
                    "case": info.case,
                    "image_path": info.rel_path,
                    "pred_pai": pai_preds.get(info.rel_path),
                    "pred_clumping": clumping_preds.get(info.rel_path) if clumping_preds else None,
                }
            )

    print(f"Wrote predictions to: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
