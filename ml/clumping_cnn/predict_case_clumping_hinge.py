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
    IndexRow,
    PaiIndexDataset,
    build_model,
    build_transforms,
    collate_keep_meta,
    get_repo_root_from_any_ml_file,
)


def _parse_float(raw: str) -> float | None:
    s = (raw or "").strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        token = s.split("+")[0].strip()
        try:
            return float(token)
        except Exception:
            return None


def _read_case_rows(index_csv: Path, case: str) -> list[IndexRow]:
    import csv as _csv

    rows: list[IndexRow] = []
    with index_csv.open("r", newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        required = {"image_path", "case_norm", "orientation", "simulation_set", "truth_PAI", "truth_PAIe_hinge"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Index CSV missing required columns: {sorted(missing)}")

        for r in reader:
            if r.get("case_norm") != case:
                continue
            pai = _parse_float(r.get("truth_PAI", ""))
            paie_hinge = _parse_float(r.get("truth_PAIe_hinge", ""))
            if pai is None or paie_hinge is None or pai <= 0:
                continue
            omega = paie_hinge / pai
            rows.append(
                IndexRow(
                    image_path=r["image_path"],
                    case_norm=r["case_norm"],
                    orientation=r["orientation"],
                    simulation_set=r["simulation_set"],
                    truth_value=float(omega),
                )
            )

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict hinge clumping (PAIe_hinge/PAI) for a Case.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--case", type=str, required=True)
    p.add_argument("--index-csv", type=str, default=None)
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

    rows = _read_case_rows(index_csv=index_csv, case=args.case)
    if not rows:
        raise ValueError(f"No usable clumping-hinge rows found for case='{args.case}'.")

    tf = build_transforms(img_size, train=False)
    ds = PaiIndexDataset(rows, repo_root=repo_root, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_keep_meta)

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
                    "truth_omega_hinge": float(t),
                    "pred_omega_hinge": float(p),
                }
            )

    mean_pred = sum(r["pred_omega_hinge"] for r in preds) / len(preds)
    truth = preds[0]["truth_omega_hinge"]

    out_csv = Path(args.out_csv) if args.out_csv else (ckpt_path.parent / f"pred_clumping_{args.case.replace(' ', '_')}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(preds[0].keys()))
        writer.writeheader()
        writer.writerows(preds)

    print(f"Case: {args.case}")
    print(f"Truth omega_hinge: {truth:.4f}")
    print(f"Pred (mean over {len(preds)} images): {mean_pred:.4f}")
    print(f"Abs error: {abs(mean_pred - truth):.4f}")
    print(f"Wrote: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
