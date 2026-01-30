from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Allow `from pai_cnn...` when invoked as `python ml/predict_all_cases_pai.py`
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pai_cnn.common import (  # noqa: E402
    PaiIndexDataset,
    build_model,
    build_transforms,
    collate_keep_meta,
    get_repo_root_from_ml_file,
    read_index_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict PAI for all cases in the dataset index using a trained checkpoint."
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model_best.pt or model_last.pt")
    p.add_argument(
        "--index-csv",
        type=str,
        default=None,
        help="Path to shared/dataset_index/image_dataset_index.csv (default: repo shared path)",
    )
    p.add_argument("--target-col", type=str, default="truth_PAI")
    p.add_argument("--orientation", type=str, default=None)
    p.add_argument("--simulation-set", type=str, default=None)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true")

    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output folder (default: alongside checkpoint)",
    )
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()

    repo_root = get_repo_root_from_ml_file(__file__)
    index_csv = (
        Path(args.index_csv) if args.index_csv else (repo_root / "shared" / "dataset_index" / "image_dataset_index.csv")
    )

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    img_size = int(cfg.get("img_size", 224))
    pretrained = bool(cfg.get("pretrained", False))

    rows = read_index_csv(
        index_csv=index_csv,
        target_col=args.target_col,
        orientation=args.orientation or cfg.get("orientation"),
        simulation_set=args.simulation_set or cfg.get("simulation_set"),
    )

    tf = build_transforms(img_size, train=False)
    ds = PaiIndexDataset(rows, repo_root=repo_root, transform=tf)
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

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image_csv = out_dir / "pred_all_images.csv"
    per_case_csv = out_dir / "pred_all_cases.csv"

    per_case_sum = defaultdict(float)
    per_case_n = defaultdict(int)
    per_case_truth = {}

    abs_err_sum = 0.0
    sq_err_sum = 0.0
    n_images = 0

    with per_image_csv.open("w", newline="", encoding="utf-8") as f_img:
        writer = csv.DictWriter(
            f_img,
            fieldnames=[
                "case_norm",
                "simulation_set",
                "orientation",
                "image_path",
                "truth",
                "pred",
            ],
        )
        writer.writeheader()

        for xb, yb, metas in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).squeeze(1)

            preds = pred.detach().cpu().tolist()
            truths = yb.squeeze(1).detach().cpu().tolist()

            for p, t, m in zip(preds, truths, metas):
                writer.writerow(
                    {
                        "case_norm": m.case_norm,
                        "simulation_set": m.simulation_set,
                        "orientation": m.orientation,
                        "image_path": m.image_path,
                        "truth": float(t),
                        "pred": float(p),
                    }
                )

                per_case_sum[m.case_norm] += float(p)
                per_case_n[m.case_norm] += 1
                per_case_truth[m.case_norm] = float(t)

                e = float(p) - float(t)
                abs_err_sum += abs(e)
                sq_err_sum += e * e
                n_images += 1

    # Write per-case means
    case_abs_sum = 0.0
    n_cases = 0

    with per_case_csv.open("w", newline="", encoding="utf-8") as f_case:
        writer = csv.DictWriter(f_case, fieldnames=["case_norm", "truth", "pred_mean", "abs_err"])
        writer.writeheader()

        for case in sorted(per_case_sum.keys()):
            mean_pred = per_case_sum[case] / per_case_n[case]
            truth = per_case_truth[case]
            ae = abs(mean_pred - truth)
            writer.writerow(
                {
                    "case_norm": case,
                    "truth": truth,
                    "pred_mean": mean_pred,
                    "abs_err": ae,
                }
            )
            case_abs_sum += ae
            n_cases += 1

    mae_image = abs_err_sum / max(1, n_images)
    rmse_image = (sq_err_sum / max(1, n_images)) ** 0.5
    mae_case = case_abs_sum / max(1, n_cases)

    print(f"Images: {n_images} | Cases: {n_cases}")
    print(f"MAE (image): {mae_image:.4f}")
    print(f"RMSE (image): {rmse_image:.4f}")
    print(f"MAE (case mean): {mae_case:.4f}")
    print(f"Wrote: {per_image_csv}")
    print(f"Wrote: {per_case_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
