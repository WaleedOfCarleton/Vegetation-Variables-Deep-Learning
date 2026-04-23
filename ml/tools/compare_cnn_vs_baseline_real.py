from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TRUTH_CSV = Path("shared/truth_join/truth_joined_to_hemipy.csv")


def compute_bias(df: pd.DataFrame) -> dict:
    out = {}
    # Baseline columns in truth_joined_to_hemipy
    # PAI hinge bias
    if {"PAI_Hinge_value", "truth_PAI"}.issubset(df.columns):
        diff = df["PAI_Hinge_value"] - df["truth_PAI"]
        out["pai_hinge_bias"] = diff.mean()
        out["pai_hinge_mae"] = diff.abs().mean()
    if {"Clumping_Hinge_value", "truth_Clumping"}.issubset(df.columns):
        diff = df["Clumping_Hinge_value"] - df["truth_Clumping"]
        out["clumping_hinge_bias"] = diff.mean()
        out["clumping_hinge_mae"] = diff.abs().mean()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Compare CNN predictions to baseline with bias correction from simulated truth.")
    p.add_argument("--truth-csv", default=TRUTH_CSV, help="CSV with truth and baseline columns (default: shared/truth_join/truth_joined_to_hemipy.csv)")
    p.add_argument("--baseline-real", required=True, help="Per-image baseline CSV for real photos")
    p.add_argument("--cnn-real", required=True, help="Per-image CNN predictions CSV for real photos")
    p.add_argument("--out-csv", required=True, help="Output comparison CSV path")
    args = p.parse_args()

    truth_df = pd.read_csv(args.truth_csv)
    bias = compute_bias(truth_df)
    if not bias:
        raise RuntimeError("Could not compute bias from truth CSV; missing columns.")

    base_df = pd.read_csv(args.baseline_real)
    cnn_df = pd.read_csv(args.cnn_real)

    # Normalize filename key for join
    base_df["_key"] = base_df["filename"].str.lower()
    if "image_path" in cnn_df.columns:
        cnn_df["_key"] = cnn_df["image_path"].str.split("/").str[-1].str.lower()
    elif "filename" in cnn_df.columns:
        cnn_df["_key"] = cnn_df["filename"].str.lower()
    else:
        raise RuntimeError("CNN CSV missing image path/filename column")

    merged = pd.merge(base_df, cnn_df, on="_key", suffixes=("_baseline", "_cnn"))

    merged["baseline_pai_biascorr"] = merged["pai_hinge"] - bias.get("pai_hinge_bias", 0.0)
    merged["baseline_clumping_biascorr"] = merged["clumping_hinge"] - bias.get("clumping_hinge_bias", 0.0)

    merged["cnn_vs_biascorr_pai_diff"] = merged["pred_pai"] - merged["baseline_pai_biascorr"]
    merged["cnn_vs_biascorr_clumping_diff"] = merged["pred_clumping"] - merged["baseline_clumping_biascorr"]

    merged.to_csv(args.out_csv, index=False)

    # Print quick summary
    summary = {
        "n_rows": len(merged),
        "bias_pai_hinge": bias.get("pai_hinge_bias"),
        "bias_clumping_hinge": bias.get("clumping_hinge_bias"),
        "cnn_minus_biascorr_pai_mean": merged["cnn_vs_biascorr_pai_diff"].mean(),
        "cnn_minus_biascorr_pai_mae": merged["cnn_vs_biascorr_pai_diff"].abs().mean(),
        "cnn_minus_biascorr_clumping_mean": merged["cnn_vs_biascorr_clumping_diff"].mean(),
        "cnn_minus_biascorr_clumping_mae": merged["cnn_vs_biascorr_clumping_diff"].abs().mean(),
    }
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"Wrote comparison CSV to: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
