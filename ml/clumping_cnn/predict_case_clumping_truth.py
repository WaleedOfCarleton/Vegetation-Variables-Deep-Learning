from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys

# Reuse PAI CNN utilities.
sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "pai").resolve()))

from pai_cnn.common import (  # noqa: E402
    PaiIndexDataset,
    build_model,
    build_transforms,
    collate_keep_meta,
    get_repo_root_from_any_ml_file,
    read_index_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict Excel truth clumping (truth_Clumping) for a Case.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--case", type=str, required=True)
    p.add_argument("--index-csv", type=str, default=None)
    p.add_argument("--orientation", type=str, default=None)
    p.add_argument("--simulation-set", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out-csv", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()

    repo_root = get_repo_root_from_any_ml_file(__file__)
    index_csv = (
        Path(args.index_csv)
        if args.index_csv
        else (repo_root / "shared" / "dataset_index" / "image_dataset_index.csv")
    )

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    img_size = int(cfg.get("img_size", 224))
    pretrained = bool(cfg.get("pretrained", False))

    rows = read_index_csv(
        index_csv=index_csv,
        target_col="truth_Clumping",
        orientation=args.orientation or cfg.get("orientation"),
        simulation_set=args.simulation_set or cfg.get("simulation_set"),
    )

    case_rows = [r for r in rows if r.case_norm == args.case]
    if not case_rows:
        raise ValueError(f"No rows found for case='{args.case}'. Check spelling, e.g. 'Case 001'.")

    tf = build_transforms(img_size, train=False)
    ds = PaiIndexDataset(case_rows, repo_root=repo_root, transform=tf)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_keep_meta,
    )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: cpu")

    model = build_model(pretrained=pretrained)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    preds = []
    for xb, yb, meta in loader:
        xb = xb.to(device)
        pred = model(xb).squeeze(1).detach().cpu().tolist()
        y = yb.squeeze(1).detach().cpu().tolist()
        metas = list(meta)
        for p, t, m in zip(pred, y, metas):
            preds.append(
                {
                    "case_norm": m.case_norm,
                    "simulation_set": m.simulation_set,
                    "orientation": m.orientation,
                    "image_path": m.image_path,
                    "truth_clumping": float(t),
                    "pred_clumping": float(p),
                }
            )

    mean_pred = sum(r["pred_clumping"] for r in preds) / len(preds)
    truth = preds[0]["truth_clumping"]

    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else (ckpt_path.parent / f"pred_clumping_truth_{args.case.replace(' ', '_')}.csv")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(preds[0].keys()))
        writer.writeheader()
        writer.writerows(preds)

    print(f"Case: {args.case}")
    print(f"Truth clumping: {truth:.4f}")
    print(f"Pred (mean over {len(preds)} images): {mean_pred:.4f}")
    print(f"Abs error: {abs(mean_pred - truth):.4f}")
    print(f"Wrote: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
