from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except Exception as e:
    raise SystemExit("Missing dependency: pandas") from e

try:
    from PIL import Image
except Exception as e:
    raise SystemExit("Missing dependency: Pillow") from e

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
except Exception as e:
    raise SystemExit("Missing dependency: PyTorch/torchvision") from e


def _read_text_strip_bom(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return raw.lstrip("\ufeff").strip()


def _parse_kv_file(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("["):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _resolve_models_dir(repo_root: Path, models_dir: Path | None, run_id: str | None) -> Path:
    if models_dir is not None:
        return (repo_root / models_dir).resolve() if not models_dir.is_absolute() else models_dir

    if run_id is None:
        latest = repo_root / "ml" / "cnn_baseline" / "latest_run.txt"
        if not latest.exists():
            raise SystemExit(
                "--models-dir was not provided and ml/cnn_baseline/latest_run.txt was not found. "
                "Provide --models-dir or --run-id."
            )
        latest_text = _read_text_strip_bom(latest)
        if not latest_text:
            raise SystemExit("ml/cnn_baseline/latest_run.txt is empty. Provide --models-dir or --run-id.")

        kv = _parse_kv_file(latest_text)
        if "models_dir" in kv and kv["models_dir"]:
            md = Path(kv["models_dir"])
            if md.exists():
                return md.resolve()

        run_id = kv.get("runId") or kv.get("run_id")
        if not run_id:
            run_id = next((ln.strip() for ln in latest_text.splitlines() if ln.strip()), "")
        if not run_id:
            raise SystemExit("Could not parse run id from ml/cnn_baseline/latest_run.txt.")

    candidates = [
        repo_root / "ml" / "cnn_baseline" / "working" / run_id / "models",
        repo_root / "ml" / "cnn_baseline" / "archive" / run_id / "models",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise SystemExit("Could not resolve models directory. Tried:\n" + "\n".join(str(c) for c in candidates))


def _load_checkpoints(models_dir: Path) -> tuple[list[dict], dict]:
    ckpts = sorted(Path(models_dir).glob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No .pth checkpoints found in: {models_dir}")

    payloads: list[dict] = []
    ref: dict | None = None

    for p in ckpts:
        payload = torch.load(str(p), map_location="cpu", weights_only=False)
        if "state_dict" not in payload:
            raise ValueError(f"Not a valid checkpoint (missing state_dict): {p}")

        cfg = {
            "backbone": str(payload.get("backbone", "resnet18")).lower().strip(),
            "image_size": int(payload.get("image_size", 224)),
            "nonnegative_head": bool(payload.get("nonnegative_head", False)),
            "pred_min": payload.get("pred_min", 0.0),
        }
        if ref is None:
            ref = cfg
        else:
            for k in ["backbone", "image_size", "nonnegative_head"]:
                if cfg.get(k) != ref.get(k):
                    raise ValueError(
                        f"Checkpoint config mismatch for key '{k}'.\n"
                        f"  ref={ref.get(k)}\n  this={cfg.get(k)}\n  path={p}"
                    )

        payloads.append({"path": str(p), **payload})

    assert ref is not None
    return payloads, ref


def _build_model(backbone: str, nonnegative_head: bool) -> torch.nn.Module:
    backbone = str(backbone).lower().strip()
    if backbone == "resnet18":
        net = models.resnet18(weights=None)
    elif backbone == "resnet34":
        net = models.resnet34(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    in_features = int(net.fc.in_features)
    if nonnegative_head:
        net.fc = nn.Sequential(nn.Linear(in_features, 1), nn.Softplus())
    else:
        net.fc = nn.Linear(in_features, 1)
    return net


def _predict_images(
    image_paths: list[Path],
    models_list: list[torch.nn.Module],
    device: torch.device,
    tfm: transforms.Compose,
    pred_min_effective: float | None,
    batch_size: int,
) -> np.ndarray:
    preds: list[float] = []
    bs = int(batch_size)
    with torch.no_grad():
        for start in range(0, len(image_paths), bs):
            batch_paths = image_paths[start : start + bs]
            batch = torch.stack([tfm(Image.open(p).convert("RGB")) for p in batch_paths], dim=0).to(device)

            ens: list[np.ndarray] = []
            for m in models_list:
                y = m(batch).detach().cpu().numpy().reshape(-1)
                if pred_min_effective is not None:
                    y = np.clip(y, pred_min_effective, None)
                ens.append(y)
            yhat = np.mean(np.stack(ens, axis=0), axis=0)
            preds.extend([float(x) for x in yhat.tolist()])

    return np.asarray(preds, dtype=float)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate the PAI CNN on ALL holdout cases and compare to truth_PAI.\n"
            "Loads the saved checkpoints once, predicts every holdout image, then summarizes per-case."
        )
    )
    p.add_argument(
        "--index",
        type=Path,
        default=Path("shared/dataset_index/image_dataset_index.csv"),
        help="Path to image_dataset_index.csv",
    )
    p.add_argument(
        "--case-split",
        type=Path,
        default=None,
        help="Path to case_split.csv. If omitted, inferred from the models_dir run folder.",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Folder containing .pth checkpoints. If omitted, uses ml/cnn_baseline/latest_run.txt.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id to locate models in ml/cnn_baseline/working/<run_id>/models.",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Base path used to resolve relative image_path entries.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path for per-case summary. Default: <run>/raw/holdout_case_metrics.csv",
    )
    p.add_argument(
        "--print-max",
        type=int,
        default=50,
        help="Print up to N cases to console (sorted by abs error).",
    )

    args = p.parse_args()

    print("Starting holdout evaluation...", flush=True)

    repo_root = Path(args.repo_root).resolve()
    index_path = (repo_root / args.index).resolve() if not args.index.is_absolute() else args.index

    models_dir = _resolve_models_dir(repo_root, args.models_dir, args.run_id)
    run_dir = models_dir.parent
    raw_dir = run_dir / "raw"

    case_split_path = args.case_split
    if case_split_path is None:
        candidate = raw_dir / "case_split.csv"
        if not candidate.exists():
            raise SystemExit(
                "--case-split not provided and could not find case_split.csv at: "
                f"{candidate}\nProvide --case-split explicitly."
            )
        case_split_path = candidate
    case_split_path = (repo_root / case_split_path).resolve() if not case_split_path.is_absolute() else case_split_path

    out_path = args.out
    if out_path is None:
        out_path = raw_dir / "holdout_case_metrics.csv"
    out_path = (repo_root / out_path).resolve() if not out_path.is_absolute() else out_path

    print(f"Index: {index_path}", flush=True)
    print(f"Models: {models_dir}", flush=True)
    print(f"Case split: {case_split_path}", flush=True)

    idx = pd.read_csv(index_path)
    split = pd.read_csv(case_split_path)

    holdout_cases = (
        split.loc[split["split"].astype(str) == "holdout", "case_norm"].astype(str).dropna().unique().tolist()
    )
    if not holdout_cases:
        raise SystemExit(f"No holdout cases found in {case_split_path}")

    print("Loading checkpoints...", flush=True)
    payloads, cfg = _load_checkpoints(models_dir)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    image_size = int(cfg["image_size"])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tfm = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize])

    pred_min = cfg.get("pred_min", 0.0)
    pred_min_effective: float | None = None if pred_min is None else float(pred_min)

    print(f"Device: {device}", flush=True)
    print(f"Holdout cases: {len(holdout_cases)}", flush=True)
    print(f"Checkpoints: {len(payloads)}", flush=True)

    print("Building models...", flush=True)
    models_list: list[torch.nn.Module] = []
    for payload in payloads:
        m = _build_model(cfg["backbone"], nonnegative_head=cfg["nonnegative_head"]).to(device)
        m.load_state_dict(payload["state_dict"], strict=True)
        m.eval()
        models_list.append(m)

    rows: list[dict] = []

    for i, case in enumerate(holdout_cases, start=1):
        print(f"Case {i}/{len(holdout_cases)}: {case}", flush=True)
        df_case = idx[idx["case_norm"].astype(str) == str(case)].dropna(subset=["image_path"]).copy()
        if len(df_case) == 0:
            rows.append({"case_norm": case, "status": "missing_in_index"})
            continue

        truth_vals = df_case.get("truth_PAI")
        if truth_vals is None:
            rows.append({"case_norm": case, "status": "missing_truth_PAI"})
            continue

        truth_unique = df_case["truth_PAI"].dropna().unique().astype(float)
        truth = float(np.mean(truth_unique)) if len(truth_unique) else float("nan")

        image_paths = [(repo_root / Path(p)).resolve() for p in df_case["image_path"].astype(str).tolist()]
        missing = [p for p in image_paths if not p.exists()]
        if missing:
            rows.append(
                {
                    "case_norm": case,
                    "status": "missing_images",
                    "n_images": int(len(image_paths)),
                    "n_missing_images": int(len(missing)),
                    "truth_PAI": truth,
                }
            )
            continue

        yhat = _predict_images(
            image_paths,
            models_list=models_list,
            device=device,
            tfm=tfm,
            pred_min_effective=pred_min_effective,
            batch_size=int(args.batch_size),
        )

        pred_mean = float(np.mean(yhat))
        pred_std = float(np.std(yhat))
        err = pred_mean - truth
        abs_err = float(abs(err))
        pct_err = float(100.0 * err / truth) if truth and np.isfinite(truth) else float("nan")
        rmse = _rmse(np.full_like(yhat, truth, dtype=float), yhat) if np.isfinite(truth) else float("nan")
        mae = float(np.mean(np.abs(yhat - truth))) if np.isfinite(truth) else float("nan")

        rows.append(
            {
                "case_norm": case,
                "status": "ok",
                "n_images": int(len(image_paths)),
                "truth_PAI": float(truth),
                "pred_mean": float(pred_mean),
                "pred_std": float(pred_std),
                "error": float(err),
                "abs_error": float(abs_err),
                "percent_error": float(pct_err),
                "rmse_vs_truth": float(rmse),
                "mae_vs_truth": float(mae),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, float_format="%.6f")

    ok = out_df[out_df["status"] == "ok"].copy()
    if len(ok) > 0:
        # Overall metrics across cases
        y = ok["truth_PAI"].astype(float).to_numpy()
        yhat = ok["pred_mean"].astype(float).to_numpy()
        overall_rmse = _rmse(y, yhat)
        overall_mae = float(np.mean(np.abs(yhat - y)))
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        corr = float(np.corrcoef(y, yhat)[0, 1]) if len(y) >= 2 else float("nan")

        print("\nOverall holdout (case-level mean predictions):")
        print(f"n_cases_ok={len(ok)}")
        print(f"rmse={overall_rmse:.6f}")
        print(f"mae={overall_mae:.6f}")
        print(f"r2={r2:.6f}")
        print(f"corr={corr:.6f}")

        show = ok.sort_values("abs_error", ascending=False)
        max_rows = int(args.print_max) if args.print_max is not None else len(show)
        show = show.head(max_rows)

        print("\nWorst cases by abs_error:")
        for _, r in show.iterrows():
            print(
                f"{r['case_norm']}: truth={float(r['truth_PAI']):.6f} "
                f"pred_mean={float(r['pred_mean']):.6f} "
                f"error={float(r['error']):+.6f} "
                f"abs_error={float(r['abs_error']):.6f} "
                f"pct={float(r['percent_error']):+.2f}%"
            )

    print(f"\nWrote summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
