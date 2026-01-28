from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = int(y_true.size)
    if n == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "r": np.nan, "r2": np.nan}

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))

    r = np.nan
    if n >= 2:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])

    sse = float(np.sum((y_pred - y_true) ** 2))
    sst = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
    return {"n": n, "mae": mae, "rmse": rmse, "bias": bias, "r": r, "r2": r2}


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute hinge-region clumping from PAI and PAIe_hinge predictions: clumping_hinge = PAIe_hinge / PAI"
    )
    p.add_argument(
        "--pai-per-site",
        type=Path,
        required=True,
        help="Per-site/case_mean PAI predictions CSV from cnn_baseline (has case_norm, truth_PAI, pred_PAI_cnn).",
    )
    p.add_argument(
        "--paie-per-site",
        type=Path,
        required=True,
        help="Per-site/case_orientation_mean PAIe_hinge predictions CSV from cnn_hinge (has case_norm, orientation, truth_PAIe_hinge, pred_PAIe_hinge_cnn).",
    )
    p.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--prefix", type=str, default="clumping_hinge")
    p.add_argument("--eps", type=float, default=1e-6, help="Small epsilon to avoid divide-by-zero.")
    args = p.parse_args()

    pai = pd.read_csv(args.pai_per_site)
    paie = pd.read_csv(args.paie_per_site)

    for col in ["case_norm", "truth_PAI", "pred_PAI_cnn"]:
        if col not in pai.columns:
            raise SystemExit(f"PAI per-site CSV missing required column: {col}")

    for col in ["case_norm", "orientation", "truth_PAIe_hinge", "pred_PAIe_hinge_cnn"]:
        if col not in paie.columns:
            raise SystemExit(f"PAIe per-site CSV missing required column: {col}")

    pai_small = pai[["case_norm", "truth_PAI", "pred_PAI_cnn"]].copy()
    pai_small = pai_small.rename(columns={"pred_PAI_cnn": "pred_PAI"})

    paie_small = paie[["case_norm", "orientation", "truth_PAIe_hinge", "pred_PAIe_hinge_cnn"]].copy()
    paie_small = paie_small.rename(columns={"pred_PAIe_hinge_cnn": "pred_PAIe_hinge"})

    merged = paie_small.merge(pai_small, on=["case_norm"], how="left", validate="many_to_one")

    eps = float(args.eps)
    merged["truth_clumping_hinge"] = merged["truth_PAIe_hinge"].astype(float) / np.maximum(
        merged["truth_PAI"].astype(float), eps
    )
    merged["pred_clumping_hinge"] = merged["pred_PAIe_hinge"].astype(float) / np.maximum(
        merged["pred_PAI"].astype(float), eps
    )

    m = regression_metrics(merged["truth_clumping_hinge"].to_numpy(), merged["pred_clumping_hinge"].to_numpy())

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix.strip()
    if prefix and not prefix.endswith("_"):
        prefix += "_"

    out_preds = outdir / f"{prefix}clumping_hinge_per_site_predictions.csv"
    out_metrics = outdir / f"{prefix}clumping_hinge_metrics.csv"

    merged.to_csv(out_preds, index=False, float_format="%.6f")
    pd.DataFrame(
        [
            {
                "level": "case_orientation_mean",
                "n": m["n"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "bias": m["bias"],
                "r": m["r"],
                "r2": m["r2"],
                "eps": eps,
            }
        ]
    ).to_csv(out_metrics, index=False, float_format="%.6f")

    print("Wrote:")
    print("  preds  :", out_preds)
    print("  metrics:", out_metrics)
    print("Metrics:", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
