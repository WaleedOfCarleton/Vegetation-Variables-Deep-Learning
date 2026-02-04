from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    p = argparse.ArgumentParser(description="Print useful metadata from a saved .pt checkpoint.")
    p.add_argument("checkpoint", type=str, help="Path to a .pt checkpoint (e.g., model_best.pt)")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        print(f"Checkpoint is not a dict (type={type(ckpt)}).")
        return 0

    info = {
        "path": str(ckpt_path.as_posix()),
        "epoch": ckpt.get("epoch"),
        "best_metric": ckpt.get("best_metric"),
        "best_metric_value": ckpt.get("best_metric_value"),
        "val_loss": ckpt.get("val_loss"),
        "val_mae_case": ckpt.get("val_mae_case"),
        "val_mae_image": ckpt.get("val_mae_image"),
        "val_rmse_image": ckpt.get("val_rmse_image"),
        "config": ckpt.get("config"),
    }

    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
