from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "Simulations").exists() and (p / "shared").exists() and (p / "ml").exists():
            return p
        if (p / ".git").exists():
            return p
    return start


REPO_ROOT = _find_repo_root(HERE)
DEFAULT_IN = REPO_ROOT / "shared" / "truth_join" / "truth_joined_to_hemipy.csv"
DEFAULT_OUT_METRICS = HERE / "estimation_metrics_summary.csv"
DEFAULT_OUT_RESIDUALS = HERE / "estimation_residuals.csv"


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    df = pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1).dropna()
    n = int(len(df))
    if n == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "r": np.nan, "r2": np.nan}

    err = df["y_pred"] - df["y_true"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))

    r = np.nan
    if n >= 2:
        r = float(np.corrcoef(df["y_true"], df["y_pred"])[0, 1])

    y = df["y_true"].to_numpy()
    yhat = df["y_pred"].to_numpy()
    sse = float(np.sum((yhat - y) ** 2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan

    return {"n": n, "mae": mae, "rmse": rmse, "bias": bias, "r": r, "r2": r2}


def _pick_split_col(df: pd.DataFrame, preferred: str | None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in ["case_norm", "Case", "truth_sim_id"]:
        if c in df.columns:
            return c
    raise ValueError(
        "No split column found. Expected one of: case_norm, Case, truth_sim_id "
        "(or pass --split-col with a valid column name)."
    )


def _make_case_folds(df: pd.DataFrame, split_col: str, kfold: int, seed: int) -> pd.DataFrame:
    if kfold < 2:
        raise ValueError("--kfold must be >= 2")

    out = df.copy()
    out["_case_key"] = out[split_col].astype(str)

    unique_cases = out["_case_key"].dropna().unique().tolist()
    unique_cases = [c for c in unique_cases if c.strip() != ""]
    unique_cases.sort()

    if len(unique_cases) < kfold:
        raise ValueError(f"Not enough unique cases ({len(unique_cases)}) for k-fold={kfold}.")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_cases)

    fold_map = {case: int(i % kfold) for i, case in enumerate(unique_cases)}
    out["fold"] = out["_case_key"].map(fold_map).astype(int)
    return out.drop(columns=["_case_key"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_IN)
    p.add_argument("--out-metrics", type=Path, default=DEFAULT_OUT_METRICS)
    p.add_argument("--out-residuals", type=Path, default=DEFAULT_OUT_RESIDUALS)
    p.add_argument("--no-residuals", action="store_true", help="Only write metrics summary (no per-row residuals CSV).")

    p.add_argument("--kfold", type=int, default=5, help="Number of folds for case-based CV (>=2).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for fold assignment.")
    p.add_argument("--split-col", type=str, default=None, help="Column to split by (default: case_norm/Case).")

    args = p.parse_args()

    df0 = pd.read_csv(args.input)
    split_col = _pick_split_col(df0, args.split_col)
    df = _make_case_folds(df0, split_col=split_col, kfold=args.kfold, seed=args.seed)

    # What to evaluate (truth column, predicted column)
    # NOTE: We do NOT pre-filter these by df.columns, because calibrated columns
    # are created inside each fold's test_df.
    pairs = [
        ("truth_PAI", "PAI_Hinge_value"),
        ("truth_PAI", "PAI_Miller_value"),
        ("truth_PAI", "PAI_Hinge_value_biascorrected"),
        ("truth_PAI", "PAI_Miller_value_biascorrected"),
        ("truth_PAI", "PAI_Hinge_value_linearcal"),
        ("truth_PAI", "PAI_Miller_value_linearcal"),
        ("truth_Clumping", "Clumping_Hinge_value"),
        ("truth_Clumping", "Clumping_Miller_value"),
    ]

    # Sanity check: at least some base columns must exist
    base_required = [("truth_PAI", "PAI_Hinge_value"), ("truth_PAI", "PAI_Miller_value")]
    if not any((t in df.columns and pcol in df.columns) for (t, pcol) in base_required):
        raise ValueError("Expected base columns like truth_PAI + PAI_Hinge_value were not found in the input CSV.")

    metric_rows: list[dict] = []
    oof_rows_for_residuals: list[pd.DataFrame] = []

    def _eval_frame(frame: pd.DataFrame, fold_value: str) -> None:
        group_labels = ["ALL"]
        groups: dict[str, pd.DataFrame] = {"ALL": frame}
        if "orientation" in frame.columns:
            for ori, g in frame.groupby("orientation", dropna=False):
                group_labels.append(str(ori))
                groups[str(ori)] = g

        for group_name in group_labels:
            g = groups[group_name]
            for truth_col, pred_col in pairs:
                if truth_col not in g.columns or pred_col not in g.columns:
                    continue
                m = regression_metrics(g[truth_col], g[pred_col])
                metric_rows.append(
                    {
                        "cv": "KFold",
                        "fold": fold_value,  # "0".."k-1" or "OOF"
                        "group": group_name,
                        "truth_col": truth_col,
                        "pred_col": pred_col,
                        **m,
                    }
                )

    # Run folds
    for fold in range(int(args.kfold)):
        train_df = df[df["fold"] != fold].copy()
        test_df = df[df["fold"] == fold].copy()

        # Fit calibrations on TRAIN, apply to TEST only
        def _bias_correct_fit_apply(truth_col: str, pred_col: str, out_col: str) -> None:
            if truth_col not in train_df.columns or pred_col not in train_df.columns:
                return
            tmp = train_df[[truth_col, pred_col]].dropna()
            if len(tmp) == 0 or pred_col not in test_df.columns:
                return
            bias = float((tmp[pred_col] - tmp[truth_col]).mean())  # pred - truth on TRAIN
            test_df[out_col] = test_df[pred_col] - bias

        def _linear_calibrate_fit_apply(truth_col: str, pred_col: str, out_col: str) -> None:
            if truth_col not in train_df.columns or pred_col not in train_df.columns:
                return
            tmp = train_df[[truth_col, pred_col]].dropna()
            if len(tmp) < 2 or pred_col not in test_df.columns:
                return
            x = tmp[pred_col].to_numpy()
            y = tmp[truth_col].to_numpy()
            b, a = np.polyfit(x, y, 1)  # y ≈ a + b*x
            test_df[out_col] = a + b * test_df[pred_col]

        _bias_correct_fit_apply("truth_PAI", "PAI_Hinge_value", "PAI_Hinge_value_biascorrected")
        _bias_correct_fit_apply("truth_PAI", "PAI_Miller_value", "PAI_Miller_value_biascorrected")
        _linear_calibrate_fit_apply("truth_PAI", "PAI_Hinge_value", "PAI_Hinge_value_linearcal")
        _linear_calibrate_fit_apply("truth_PAI", "PAI_Miller_value", "PAI_Miller_value_linearcal")

        # Metrics for this fold (TEST fold only)
        _eval_frame(test_df, fold_value=str(fold))

        # Residual rows for this fold (TEST fold only)
        if not args.no_residuals:
            keep_id_cols = [c for c in ["case_norm", "Case", "orientation", "Root", "Plot", "Direction"] if c in test_df.columns]
            keep_id_cols = keep_id_cols + ["fold"]

            residual_parts = []
            for truth_col, pred_col in pairs:
                if truth_col not in test_df.columns or pred_col not in test_df.columns:
                    continue
                sub = test_df[keep_id_cols + [truth_col, pred_col]].copy()
                sub["truth_col"] = truth_col
                sub["pred_col"] = pred_col
                sub["error"] = sub[pred_col] - sub[truth_col]
                sub["abs_error"] = sub["error"].abs()
                residual_parts.append(sub)

            if residual_parts:
                oof_rows_for_residuals.append(pd.concat(residual_parts, ignore_index=True))

    # OOF aggregate metrics
    residuals_df = None
    if not args.no_residuals and oof_rows_for_residuals:
        residuals_df = pd.concat(oof_rows_for_residuals, ignore_index=True)

        # OOF overall + by orientation
        group_frames = [("ALL", residuals_df)]
        if "orientation" in residuals_df.columns:
            group_frames += [(str(ori), gg) for ori, gg in residuals_df.groupby("orientation", dropna=False)]

        for group_name, g in group_frames:
            for (truth_col, pred_col), gg2 in g.groupby(["truth_col", "pred_col"]):
                m = regression_metrics(gg2[truth_col], gg2[pred_col])
                metric_rows.append(
                    {
                        "cv": "KFold",
                        "fold": "OOF",
                        "group": group_name,
                        "truth_col": truth_col,
                        "pred_col": pred_col,
                        **m,
                    }
                )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["truth_col", "pred_col", "fold", "group"])
    metrics_df.to_csv(args.out_metrics, index=False, float_format="%.2f")

    if not args.no_residuals:
        if residuals_df is None:
            pd.DataFrame(columns=["fold", "truth_col", "pred_col", "error", "abs_error"]).to_csv(args.out_residuals, index=False)
        else:
            residuals_df.to_csv(args.out_residuals, index=False)

    print("CV: KFold")
    print("Split column:", split_col)
    print("Folds:", args.kfold)
    print("Fold counts:\n", df["fold"].value_counts(dropna=False).sort_index())
    print("Wrote metrics:", args.out_metrics)
    if not args.no_residuals:
        print("Wrote residuals:", args.out_residuals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())