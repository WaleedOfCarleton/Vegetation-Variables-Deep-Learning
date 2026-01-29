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


def _parse_latest_run_file(text: str) -> dict[str, str]:
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
                "--models-dir was not provided and latest_run.txt was not found. "
                "Provide --models-dir or --run-id."
            )
        latest_text = _read_text_strip_bom(latest)
        if not latest_text:
            raise SystemExit("latest_run.txt is empty. Provide --models-dir or --run-id.")

        kv = _parse_latest_run_file(latest_text)
        # If the file contains an absolute models_dir path, prefer it.
        if "models_dir" in kv and kv["models_dir"]:
            md = Path(kv["models_dir"])
            if md.exists():
                return md.resolve()

        run_id = kv.get("runId") or kv.get("run_id")
        if not run_id:
            # Fallback: use first non-empty line.
            run_id = next((ln.strip() for ln in latest_text.splitlines() if ln.strip()), "")
        if not run_id:
            raise SystemExit("Could not parse run id from latest_run.txt. Provide --models-dir or --run-id.")

    # Prefer working/, fall back to archive/.
    candidates = [
        repo_root / "ml" / "cnn_baseline" / "working" / run_id / "models",
        repo_root / "ml" / "cnn_baseline" / "archive" / run_id / "models",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise SystemExit("Could not resolve models directory. Tried:\n" + "\n".join(str(c) for c in candidates))


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


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Predict PAI for all images belonging to a case_norm (e.g., 'Case 002')\n"
            "by looking up image_path rows in the dataset index and running a saved CNN ensemble."
        )
    )
    p.add_argument("--case", required=True, help="Case identifier exactly matching case_norm (e.g., 'Case 002').")
    p.add_argument(
        "--index",
        type=Path,
        default=Path("shared/dataset_index/image_dataset_index.csv"),
        help="Path to image_dataset_index.csv",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help=(
            "Folder containing one or more .pth checkpoints (ensemble average). "
            "If omitted, uses ml/cnn_baseline/latest_run.txt to find working/<run_id>/models."
        ),
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id to locate models in ml/cnn_baseline/working/<run_id>/models (or archive/<run_id>/models).",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Base path used to resolve relative image_path entries.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV output path for per-image predictions.",
    )
    p.add_argument(
        "--no-print",
        dest="do_print",
        action="store_false",
        help="Do not print per-image predictions to the console.",
    )
    p.set_defaults(do_print=True)
    p.add_argument(
        "--print-max",
        type=int,
        default=None,
        help="If set, only print the first N per-image predictions (after sorting by image_path).",
    )
    args = p.parse_args()

    print("Starting case prediction...", flush=True)

    repo_root = Path(args.repo_root).resolve()
    index_path = (repo_root / args.index).resolve() if not args.index.is_absolute() else args.index
    resolved_models_dir = _resolve_models_dir(repo_root, args.models_dir, args.run_id)

    print(f"Loading index: {index_path}", flush=True)
    df = pd.read_csv(index_path)
    if "case_norm" not in df.columns or "image_path" not in df.columns:
        raise SystemExit("Index missing required columns: case_norm, image_path")

    case = str(args.case)
    df = df[df["case_norm"].astype(str) == case].dropna(subset=["image_path"]).copy()
    if len(df) == 0:
        raise SystemExit(f"No rows found for {case} in {index_path}")

    image_paths = [(repo_root / Path(p)).resolve() for p in df["image_path"].astype(str).tolist()]
    missing = [str(p) for p in image_paths if not p.exists()]
    if missing:
        raise SystemExit("Missing image files (first 10):\n" + "\n".join(missing[:10]))

    print(f"Loading checkpoints from: {resolved_models_dir}", flush=True)
    payloads, cfg = _load_checkpoints(resolved_models_dir)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    image_size = int(cfg["image_size"])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tfm = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize])

    pred_min = cfg.get("pred_min", 0.0)
    pred_min_effective: float | None = None if pred_min is None else float(pred_min)

    print("Building models...", flush=True)
    models_list: list[torch.nn.Module] = []
    for payload in payloads:
        m = _build_model(cfg["backbone"], nonnegative_head=cfg["nonnegative_head"]).to(device)
        m.load_state_dict(payload["state_dict"], strict=True)
        m.eval()
        models_list.append(m)

    print(f"Running inference on {len(image_paths)} images...", flush=True)
    preds: list[float] = []
    bs = int(args.batch_size)
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

    out_df = pd.DataFrame(
        {
            "case_norm": [case] * len(image_paths),
            "image_path": [str(p) for p in image_paths],
            "pred_PAI_cnn": np.array(preds, dtype=float),
            "n_models": int(len(models_list)),
            "backbone": str(cfg["backbone"]),
            "image_size": int(image_size),
            "pred_min": ("" if pred_min_effective is None else float(pred_min_effective)),
        }
    )

    mean = float(out_df["pred_PAI_cnn"].mean())
    std = float(out_df["pred_PAI_cnn"].std(ddof=0))

    print(f"Device: {device}")
    print(f"Case: {case}")
    print(f"Images: {len(out_df)}")
    print(f"Checkpoints: {len(models_list)} (from {resolved_models_dir})")
    print(f"Predicted PAI (mean +/- std): {mean:.6f} +/- {std:.6f}")

    if args.do_print:
        show = out_df[["pred_PAI_cnn", "image_path"]].copy()
        show = show.sort_values("image_path")
        if args.print_max is not None:
            show = show.head(int(args.print_max))
        print("\nPer-image predictions:")
        for _, r in show.iterrows():
            print(f"{float(r['pred_PAI_cnn']):.6f}\t{r['image_path']}")

    if args.out is not None:
        out_path = args.out
        if not out_path.is_absolute():
            out_path = (repo_root / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False, float_format="%.6f")
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
