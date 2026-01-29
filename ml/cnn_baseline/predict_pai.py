from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except Exception as e:
    raise SystemExit("Missing dependency: Pillow. Install with: pip install pillow") from e

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision import models
except Exception as e:
    raise SystemExit(
        "Missing dependency: PyTorch/torchvision.\n"
        "Install (CPU): pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
        "Install (CUDA): follow https://pytorch.org/get-started/locally/"
    ) from e

def build_model(backbone: str, freeze_backbone: bool, nonnegative_head: bool) -> torch.nn.Module:
    backbone = str(backbone).lower().strip()

    if backbone == "resnet18":
        net = models.resnet18(weights=None)
    elif backbone == "resnet34":
        net = models.resnet34(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Use resnet18 or resnet34.")

    if freeze_backbone:
        for p in net.parameters():
            p.requires_grad = False

    in_features = int(net.fc.in_features)
    if nonnegative_head:
        net.fc = nn.Sequential(nn.Linear(in_features, 1), nn.Softplus())
    else:
        net.fc = nn.Linear(in_features, 1)

    return net


def _resolve_images(args: argparse.Namespace) -> list[Path]:
    images: list[Path] = []

    if args.image is not None:
        images.append(Path(args.image))

    if args.images:
        images.extend([Path(p) for p in args.images])

    if args.image_dir is not None:
        root = Path(args.image_dir)
        if not root.exists():
            raise FileNotFoundError(f"image-dir not found: {root}")
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                images.append(p)

    if not images:
        raise SystemExit("No images provided. Use --image, --images, or --image-dir.")

    # De-dup while keeping order.
    seen = set()
    out: list[Path] = []
    for p in images:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def _load_checkpoints(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Returns (payloads, cfg) where cfg is the shared model config."""

    paths: list[Path] = []
    if args.model is not None:
        paths.append(Path(args.model))

    if args.models_dir is not None:
        d = Path(args.models_dir)
        if not d.exists():
            raise FileNotFoundError(f"models-dir not found: {d}")
        paths.extend(sorted(d.glob("*.pth")))

    if not paths:
        raise SystemExit("Provide --model <checkpoint.pth> or --models-dir <folder_with_pth_files>.")

    payloads: list[dict] = []
    ref: dict | None = None

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        # Explicit weights_only=False avoids a FutureWarning in newer PyTorch.
        payload = torch.load(str(p), map_location="cpu", weights_only=False)
        if "state_dict" not in payload:
            raise ValueError(f"Not a valid checkpoint (missing state_dict): {p}")

        cfg = {
            "backbone": str(payload.get("backbone", "resnet18")),
            "image_size": int(payload.get("image_size", 224)),
            "nonnegative_head": bool(payload.get("nonnegative_head", False)),
            "pred_min": payload.get("pred_min", 0.0),
        }
        if ref is None:
            ref = cfg
        else:
            # Ensure we aren't accidentally averaging incompatible models.
            for k in ["backbone", "image_size", "nonnegative_head"]:
                if cfg.get(k) != ref.get(k):
                    raise ValueError(
                        f"Checkpoint config mismatch for key '{k}'.\n"
                        f"  ref={ref.get(k)}\n  this={cfg.get(k)}\n  path={p}"
                    )

        payloads.append({"path": str(p), **payload})

    assert ref is not None
    return payloads, ref


def _predict_one_model(model: torch.nn.Module, device: torch.device, batch: torch.Tensor, *, pred_min: float | None) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        y = model(batch.to(device))
        y = y.detach().cpu().numpy().reshape(-1)
    if pred_min is not None:
        y = np.clip(y, float(pred_min), None)
    return y


def main() -> int:
    p = argparse.ArgumentParser(description="Predict PAI from image(s) using saved CNN checkpoint(s).")

    p.add_argument("--model", type=Path, default=None, help="Path to a single .pth checkpoint.")
    p.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Folder containing multiple .pth checkpoints (ensemble average).",
    )

    p.add_argument("--image", type=Path, default=None, help="Single image path.")
    p.add_argument("--images", nargs="*", default=None, help="One or more image paths.")
    p.add_argument("--image-dir", type=Path, default=None, help="Recursively predict all images in a folder.")

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")

    p.add_argument(
        "--pred-min",
        type=float,
        default=None,
        help="Optional override for pred_min clamping (default: use checkpoint setting).",
    )

    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV output path. If omitted, prints to stdout.",
    )

    args = p.parse_args()

    images = _resolve_images(args)
    payloads, cfg = _load_checkpoints(args)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    image_size = int(cfg["image_size"])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    pred_min = cfg.get("pred_min", 0.0)
    if args.pred_min is not None:
        pred_min = float(args.pred_min)

    # pred_min can be None in some checkpoints; keep that behavior.
    if pred_min is None:
        pred_min_effective: float | None = None
    else:
        pred_min_effective = float(pred_min)

    # Build/load models
    models: list[torch.nn.Module] = []
    for payload in payloads:
        m = build_model(cfg["backbone"], freeze_backbone=False, nonnegative_head=cfg["nonnegative_head"]).to(device)
        m.load_state_dict(payload["state_dict"], strict=True)
        m.eval()
        models.append(m)

    # Predict in batches
    rows: list[dict] = []
    bs = int(args.batch_size)

    for start in range(0, len(images), bs):
        batch_paths = images[start : start + bs]
        batch_tensors = []
        for ip in batch_paths:
            if not ip.exists():
                raise FileNotFoundError(f"Image not found: {ip}")
            im = Image.open(ip).convert("RGB")
            batch_tensors.append(tfm(im))
        batch = torch.stack(batch_tensors, dim=0)

        preds = []
        for m in models:
            preds.append(_predict_one_model(m, device, batch, pred_min=pred_min_effective))
        pred_mean = np.mean(np.stack(preds, axis=0), axis=0)

        for ip, yhat in zip(batch_paths, pred_mean.tolist()):
            rows.append(
                {
                    "image_path": str(ip),
                    "pred_PAI_cnn": float(yhat),
                    "n_models": int(len(models)),
                    "backbone": str(cfg["backbone"]),
                    "image_size": int(image_size),
                    "pred_min": ("" if pred_min_effective is None else float(pred_min_effective)),
                }
            )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        pd.DataFrame(rows).to_csv(args.out, index=False, float_format="%.6f")
        print(f"Wrote: {args.out}")
    else:
        for r in rows:
            print(f"{r['pred_PAI_cnn']:.6f}\t{r['image_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
