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


def fit_linear_calibration(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        raise ValueError("Not enough finite points to fit calibration")

    # Solve y ≈ a*x + b via least squares
    A = np.column_stack([x, np.ones_like(x)])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b)


def _build_fit_df(per_image: pd.DataFrame, fit_level: str, pred_col: str = "pred_PAI_cnn") -> pd.DataFrame:
    if fit_level == "case_mean":
        return (
            per_image.groupby("case_norm", as_index=False)
            .agg(truth_PAI=("truth_PAI", "mean"), pred_PAI_cnn=(pred_col, "mean"))
            .copy()
        )

    return per_image[["truth_PAI", pred_col]].rename(columns={pred_col: "pred_PAI_cnn"}).copy()


def _apply_calibration(
    pred_raw: np.ndarray,
    a: float,
    b: float,
    clamp_min: float,
    allow_negative: bool,
) -> np.ndarray:
    pred = a * np.asarray(pred_raw, dtype=np.float64) + float(b)
    if not allow_negative:
        pred = np.maximum(pred, float(clamp_min))
    return pred


def main() -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Calibrate CNN predictions to reduce bias (OOF).")
    p.add_argument(
        "--per-image",
        type=Path,
        required=True,
        help="Input per-image predictions CSV (from train_cnn_pai.py).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=here,
        help="Output directory for calibrated CSVs/plots.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="calibrated",
        help="Prefix used for output filenames.",
    )
    p.add_argument(
        "--fit-level",
        choices=["case_mean", "image"],
        default="case_mean",
        help="Fit calibration using case_mean (recommended for true PAI) or image-level.",
    )
    p.add_argument(
        "--oof-foldwise",
        action="store_true",
        help=(
            "Perform leakage-safe OOF calibration: for each fold f, fit a,b on all other folds' OOF rows, "
            "then apply to fold f. Requires a 'fold' column in the per-image CSV."
        ),
    )
    p.add_argument(
        "--clamp-min",
        type=float,
        default=0.0,
        help="Clamp calibrated predictions to be >= this value (default 0.0).",
    )
    p.add_argument(
        "--allow-negative-calibrated",
        action="store_true",
        help="Disable clamping and allow negative calibrated predictions.",
    )
    p.add_argument(
        "--write-raw-metrics",
        action="store_true",
        help="Also include raw (uncalibrated) metric rows in the output metrics CSV.",
    )
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix.strip()
    if prefix and not prefix.endswith("_"):
        prefix = prefix + "_"

    per_image = pd.read_csv(args.per_image)

    required = {"case_norm", "truth_PAI", "pred_PAI_cnn"}
    missing = required - set(per_image.columns)
    if missing:
        raise SystemExit(f"Missing required columns in per-image CSV: {sorted(missing)}")

    out = per_image.copy()
    out["pred_PAI_cnn_raw"] = out["pred_PAI_cnn"].astype(float)

    a: float
    b: float
    if bool(args.oof_foldwise):
        if "fold" not in out.columns:
            raise SystemExit("--oof-foldwise requires a 'fold' column in the per-image CSV")

        folds = sorted(pd.Series(out["fold"]).dropna().unique().tolist())
        if len(folds) < 2:
            raise SystemExit(f"--oof-foldwise requires >= 2 folds; found: {folds}")

        out["cal_a"] = np.nan
        out["cal_b"] = np.nan
        out["pred_PAI_cnn_cal"] = np.nan

        for f in folds:
            fit_rows = out[out["fold"] != f]
            apply_mask = out["fold"] == f

            fit_df = _build_fit_df(fit_rows, args.fit_level, pred_col="pred_PAI_cnn_raw")
            a_f, b_f = fit_linear_calibration(
                fit_df["pred_PAI_cnn"].to_numpy(),
                fit_df["truth_PAI"].to_numpy(),
            )

            out.loc[apply_mask, "cal_a"] = a_f
            out.loc[apply_mask, "cal_b"] = b_f
            out.loc[apply_mask, "pred_PAI_cnn_cal"] = _apply_calibration(
                out.loc[apply_mask, "pred_PAI_cnn_raw"].to_numpy(),
                a_f,
                b_f,
                clamp_min=float(args.clamp_min),
                allow_negative=bool(args.allow_negative_calibrated),
            )

        out["pred_PAI_cnn"] = out["pred_PAI_cnn_cal"].astype(float)
        out.drop(columns=["pred_PAI_cnn_cal"], inplace=True)

        # For convenience in the metrics row
        a = float(np.nanmean(out["cal_a"].to_numpy(dtype=np.float64)))
        b = float(np.nanmean(out["cal_b"].to_numpy(dtype=np.float64)))
        print(
            f"Calibration fit ({args.fit_level}, oof_foldwise): mean a={a:.6f}, mean b={b:.6f} (per-fold values recorded in output CSV)"
        )
        pred_label = "cnn_calibrated_oof_foldwise"
        scheme = "oof_foldwise"
    else:
        fit_df = _build_fit_df(out, args.fit_level, pred_col="pred_PAI_cnn")
        a, b = fit_linear_calibration(fit_df["pred_PAI_cnn"].to_numpy(), fit_df["truth_PAI"].to_numpy())
        print(f"Calibration fit ({args.fit_level}): truth ≈ a*pred + b  with  a={a:.6f}, b={b:.6f}")
        out["pred_PAI_cnn"] = _apply_calibration(
            out["pred_PAI_cnn_raw"].to_numpy(),
            a,
            b,
            clamp_min=float(args.clamp_min),
            allow_negative=bool(args.allow_negative_calibrated),
        )
        pred_label = "cnn_calibrated"
        scheme = "global"

    out["error"] = out["pred_PAI_cnn"].to_numpy(dtype=float) - out["truth_PAI"].to_numpy(dtype=float)
    out["abs_error"] = np.abs(out["error"].to_numpy(dtype=float))

    per_site = (
        out.groupby("case_norm", as_index=False)
        .agg(truth_PAI=("truth_PAI", "mean"), pred_PAI_cnn=("pred_PAI_cnn", "mean"))
        .copy()
    )

    # Metrics
    m_img_cal = regression_metrics(out["truth_PAI"].to_numpy(), out["pred_PAI_cnn"].to_numpy())
    m_site_cal = regression_metrics(per_site["truth_PAI"].to_numpy(), per_site["pred_PAI_cnn"].to_numpy())

    metric_rows: list[dict] = []
    metric_rows.append(
        {
            "cv": "KFold",
            "fold": "OOF",
            "level": "case_mean",
            "target": "truth_PAI",
            "pred": pred_label,
            **m_site_cal,
            "cal_a": a,
            "cal_b": b,
            "cal_fit_level": args.fit_level,
            "cal_scheme": scheme,
            "cal_clamp_min": ("" if args.allow_negative_calibrated else float(args.clamp_min)),
        }
    )
    metric_rows.append(
        {
            "cv": "KFold",
            "fold": "OOF",
            "level": "image",
            "target": "truth_PAI",
            "pred": pred_label,
            **m_img_cal,
            "cal_a": a,
            "cal_b": b,
            "cal_fit_level": args.fit_level,
            "cal_scheme": scheme,
            "cal_clamp_min": ("" if args.allow_negative_calibrated else float(args.clamp_min)),
        }
    )

    if bool(args.write_raw_metrics):
        m_img_raw = regression_metrics(out["truth_PAI"].to_numpy(), out["pred_PAI_cnn_raw"].to_numpy())
        per_site_raw = (
            per_image.groupby("case_norm", as_index=False)
            .agg(truth_PAI=("truth_PAI", "mean"), pred_PAI_cnn=("pred_PAI_cnn", "mean"))
            .copy()
        )
        m_site_raw = regression_metrics(per_site_raw["truth_PAI"].to_numpy(), per_site_raw["pred_PAI_cnn"].to_numpy())
        metric_rows.append(
            {
                "cv": "KFold",
                "fold": "OOF",
                "level": "case_mean",
                "target": "truth_PAI",
                "pred": "cnn_raw",
                **m_site_raw,
                "cal_a": "",
                "cal_b": "",
                "cal_fit_level": "",
                "cal_scheme": "",
                "cal_clamp_min": "",
            }
        )
        metric_rows.append(
            {
                "cv": "KFold",
                "fold": "OOF",
                "level": "image",
                "target": "truth_PAI",
                "pred": "cnn_raw",
                **m_img_raw,
                "cal_a": "",
                "cal_b": "",
                "cal_fit_level": "",
                "cal_scheme": "",
                "cal_clamp_min": "",
            }
        )

    metrics_df = pd.DataFrame(metric_rows)

    out_per_image = outdir / f"{prefix}cnn_per_image_predictions.csv"
    out_per_site = outdir / f"{prefix}cnn_per_site_predictions.csv"
    out_metrics = outdir / f"{prefix}cnn_baseline_metrics.csv"

    out.to_csv(out_per_image, index=False, float_format="%.4f")
    per_site.to_csv(out_per_site, index=False, float_format="%.4f")
    metrics_df.to_csv(out_metrics, index=False, float_format="%.6f")

    print("Wrote:")
    print("  per-image:", out_per_image)
    print("  per-site :", out_per_site)
    print("  metrics  :", out_metrics)

    print("Calibrated OOF case_mean metrics:", m_site_cal)
    print("Calibrated OOF image metrics:", m_img_cal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
