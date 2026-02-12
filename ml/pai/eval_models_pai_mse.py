from __future__ import annotations

import argparse
import csv
import fnmatch
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Allow `from pai_cnn...` when invoked as `python ml/pai/eval_models_pai_mse.py`
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pai_cnn.common import (  # noqa: E402
    PaiIndexDataset,
    build_model,
    build_transforms,
    collate_keep_meta,
    get_repo_root_from_any_ml_file,
    read_index_csv,
)


@dataclass(frozen=True)
class CaseAggKey:
    simulation_set: str
    case_norm: str


def _discover_checkpoints(runs_dirs: list[Path], patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()

    for root in runs_dirs:
        if not root.exists():
            continue
        for fp in root.rglob("*.pt"):
            name = fp.name
            if not any(fnmatch.fnmatch(name, pat) for pat in patterns):
                continue
            r = fp.resolve()
            if r not in seen:
                out.append(r)
                seen.add(r)

    out.sort()
    return out


def parse_args() -> argparse.Namespace:
    repo_root = get_repo_root_from_any_ml_file(__file__)

    default_runs = [
        repo_root / "ml" / "runs" / "pai_cnn",
        repo_root / "ml" / "runs" / "pai_cnn_kfold",
    ]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_out_dir = repo_root / "shared" / "estimations_eval" / "pai_model_eval" / ts

    p = argparse.ArgumentParser(
        description=(
            "Evaluate many PAI CNN checkpoints and compute per-case mean-squared-error (squared loss) per simulation folder. "
            "Also writes an across-model average squared loss per case."
        )
    )

    p.add_argument(
        "--runs-dir",
        type=Path,
        action="append",
        default=default_runs,
        help=(
            "Folder(s) to search for checkpoints (*.pt). Can be passed multiple times. "
            "Default: ml/runs/pai_cnn and ml/runs/pai_cnn_kfold"
        ),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=None,
        help="Explicit checkpoint path(s) to evaluate (skips discovery). Can be passed multiple times.",
    )
    p.add_argument(
        "--checkpoint-glob",
        type=str,
        action="append",
        default=["model_best*.pt"],
        help="Filename glob(s) to match within runs-dir. Default: model_best*.pt",
    )

    p.add_argument(
        "--index-csv",
        type=Path,
        default=None,
        help="Path to shared/dataset_index/image_dataset_index.csv (default: repo shared path)",
    )
    p.add_argument("--target-col", type=str, default="truth_PAI")

    p.add_argument(
        "--simulation-set",
        type=str,
        default=None,
        help="Optional filter, e.g. 'RND' or 'DHP - ERECT - 4000x4000' (matches index column simulation_set exactly).",
    )
    p.add_argument(
        "--orientation",
        type=str,
        default=None,
        help="Optional filter, e.g. ERECT/PLANO/RND (matches index column orientation exactly).",
    )

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Default 0 for Windows stability.",
    )
    p.add_argument("--cpu", action="store_true")

    p.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir,
        help="Output directory (default: shared/estimations_eval/pai_model_eval/<timestamp>)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list discovered checkpoints and exit (no model inference).",
    )

    return p.parse_args()


@torch.no_grad()
def _eval_one_checkpoint(
    *,
    checkpoint: Path,
    index_csv: Path,
    target_col: str,
    simulation_set: str | None,
    orientation: str | None,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[list[dict], dict]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})

    img_size = int(cfg.get("img_size", 224))
    pretrained = bool(cfg.get("pretrained", False))

    rows = read_index_csv(
        index_csv=index_csv,
        target_col=target_col,
        orientation=orientation,
        simulation_set=simulation_set,
    )

    repo_root = get_repo_root_from_any_ml_file(__file__)

    tf = build_transforms(img_size, train=False)
    ds = PaiIndexDataset(rows, repo_root=repo_root, transform=tf)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_keep_meta,
    )

    model = build_model(pretrained=pretrained)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    # Per-case aggregation (using mean prediction across images in that case)
    pred_sum: dict[CaseAggKey, float] = defaultdict(float)
    pred_n: dict[CaseAggKey, int] = defaultdict(int)
    truth_by_case: dict[CaseAggKey, float] = {}

    # Optional overall image-level MSE
    img_sq_sum = 0.0
    img_n = 0

    for xb, yb, metas in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb).squeeze(1)

        preds = pred.detach().cpu().tolist()
        truths = yb.squeeze(1).detach().cpu().tolist()

        for p, t, m in zip(preds, truths, metas):
            key = CaseAggKey(simulation_set=str(m.simulation_set), case_norm=str(m.case_norm))
            pred_sum[key] += float(p)
            pred_n[key] += 1
            truth_by_case.setdefault(key, float(t))

            e = float(p) - float(t)
            img_sq_sum += e * e
            img_n += 1

    per_case_rows: list[dict] = []
    case_sq_sum = 0.0
    case_n = 0

    for key in sorted(pred_sum.keys(), key=lambda k: (k.simulation_set, k.case_norm)):
        mean_pred = pred_sum[key] / max(1, pred_n[key])
        truth = truth_by_case.get(key)
        if truth is None:
            continue

        sq_err = (mean_pred - float(truth)) ** 2
        per_case_rows.append(
            {
                "checkpoint": checkpoint.as_posix(),
                "simulation_set": key.simulation_set,
                "case_norm": key.case_norm,
                "truth": float(truth),
                "pred_mean": float(mean_pred),
                "sq_err_case_mean": float(sq_err),
                "n_images": int(pred_n[key]),
            }
        )
        case_sq_sum += float(sq_err)
        case_n += 1

    summary = {
        "checkpoint": checkpoint.as_posix(),
        "img_mse": (img_sq_sum / max(1, img_n)),
        "img_n": int(img_n),
        "case_mse": (case_sq_sum / max(1, case_n)),
        "case_n": int(case_n),
        "img_size": int(img_size),
        "pretrained": bool(pretrained),
    }

    return per_case_rows, summary


def main() -> int:
    args = parse_args()

    repo_root = get_repo_root_from_any_ml_file(__file__)
    index_csv = (
        Path(args.index_csv)
        if args.index_csv
        else (repo_root / "shared" / "dataset_index" / "image_dataset_index.csv")
    )

    if args.checkpoint:
        checkpoints = [p.resolve() for p in args.checkpoint]
    else:
        checkpoints = _discover_checkpoints([Path(p) for p in args.runs_dir], patterns=list(args.checkpoint_glob))

    if not checkpoints:
        raise FileNotFoundError(
            "No checkpoints found. Try passing --runs-dir <path> or --checkpoint <path>, "
            "or broaden --checkpoint-glob (e.g. --checkpoint-glob '*.pt')."
        )

    print(f"Checkpoints: {len(checkpoints)}")
    for p in checkpoints[:10]:
        print(" -", p)
    if len(checkpoints) > 10:
        print(f" ... (+{len(checkpoints) - 10} more)")

    if args.dry_run:
        return 0

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: cpu")

    out_dir: Path = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_per_case = out_dir / "per_model_per_case.csv"
    out_per_model = out_dir / "per_model_summary.csv"
    out_avg_models = out_dir / "avg_sq_loss_across_models_per_case.csv"

    all_case_rows: list[dict] = []
    model_summaries: list[dict] = []

    for i, ckpt in enumerate(checkpoints, start=1):
        print(f"[{i}/{len(checkpoints)}] Evaluating: {ckpt}")
        case_rows, summary = _eval_one_checkpoint(
            checkpoint=ckpt,
            index_csv=index_csv,
            target_col=str(args.target_col),
            simulation_set=args.simulation_set,
            orientation=args.orientation,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
        )
        all_case_rows.extend(case_rows)
        model_summaries.append(summary)

    # Write per-model-per-case
    with out_per_case.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "checkpoint",
            "simulation_set",
            "case_norm",
            "truth",
            "pred_mean",
            "sq_err_case_mean",
            "n_images",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_case_rows:
            w.writerow(r)

    # Write per-model summary
    with out_per_model.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["checkpoint", "img_mse", "img_n", "case_mse", "case_n", "img_size", "pretrained"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in model_summaries:
            w.writerow(r)

    # Average squared loss across models for each (simulation_set, case_norm)
    acc_sq: dict[tuple[str, str], float] = defaultdict(float)
    acc_n: dict[tuple[str, str], int] = defaultdict(int)
    truth_map: dict[tuple[str, str], float] = {}

    for r in all_case_rows:
        key = (str(r["simulation_set"]), str(r["case_norm"]))
        acc_sq[key] += float(r["sq_err_case_mean"])
        acc_n[key] += 1
        truth_map.setdefault(key, float(r["truth"]))

    avg_rows: list[dict] = []
    for (sim_set, case_norm), sq_sum in sorted(acc_sq.items()):
        n = max(1, acc_n[(sim_set, case_norm)])
        avg_rows.append(
            {
                "simulation_set": sim_set,
                "case_norm": case_norm,
                "truth": float(truth_map[(sim_set, case_norm)]),
                "avg_sq_loss_models": float(sq_sum / n),
                "n_models": int(acc_n[(sim_set, case_norm)]),
            }
        )

    with out_avg_models.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["simulation_set", "case_norm", "truth", "avg_sq_loss_models", "n_models"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in avg_rows:
            w.writerow(r)

    print("Wrote:", out_per_case)
    print("Wrote:", out_per_model)
    print("Wrote:", out_avg_models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
