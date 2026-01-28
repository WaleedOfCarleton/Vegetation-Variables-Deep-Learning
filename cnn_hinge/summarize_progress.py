from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_PROGRESS = HERE / "cnn_hinge_training_progress.csv"
DEFAULT_LATEST = HERE / "latest_run.txt"


def _parse_ts_utc(s: str) -> datetime | None:
    # Expected like: 20260127_165241Z
    try:
        return datetime.strptime(str(s).strip(), "%Y%m%d_%H%M%SZ")
    except Exception:
        return None


def _pick_run_id(args_run_id: str | None) -> str | None:
    def _clean(v: str) -> str:
        return str(v).strip().lstrip("\ufeff")

    if args_run_id:
        v = _clean(args_run_id)
        return v if v else None
    if DEFAULT_LATEST.exists():
        v = _clean(DEFAULT_LATEST.read_text(encoding="utf-8"))
        return v if v else None
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize cnn_hinge training progress CSV.")
    p.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    p.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run id to filter on. Defaults to cnn_hinge/latest_run.txt if present; else shows all.",
    )
    p.add_argument("--only-epoch", action="store_true", help="Only show end-of-epoch rows (event=epoch).")
    args = p.parse_args()

    progress_path = Path(args.progress)
    if not progress_path.exists():
        raise SystemExit(f"Progress CSV not found: {progress_path}")

    df = pd.read_csv(progress_path)
    if df.empty:
        print("Progress CSV is empty.")
        return 0

    if "run_id" not in df.columns:
        raise SystemExit(
            "Progress CSV does not have a 'run_id' column. "
            "Use a newer cnn_hinge/train_cnn_paie_hinge.py that logs run_id."
        )

    run_id = _pick_run_id(args.run_id)
    if not run_id:
        # Auto-pick most recent run_id from the CSV (by timestamp_utc when possible).
        if "timestamp_utc" in df.columns:
            ts = df["timestamp_utc"].map(_parse_ts_utc)
            if ts.notna().any():
                run_id = str(df.loc[ts.idxmax(), "run_id"]).strip()
        if not run_id:
            # Fallback: last row in file order
            run_id = str(df.iloc[-1]["run_id"]).strip()
        run_id = run_id if run_id else None
    if run_id:
        df = df[df["run_id"].astype(str) == run_id].copy()
        if df.empty:
            print(f"No rows found for run_id={run_id!r}.")
            return 0
        print(f"(auto) using run_id={run_id}")

    # parse timestamps for sorting
    if "timestamp_utc" in df.columns:
        df["_ts"] = df["timestamp_utc"].map(_parse_ts_utc)
        df = df.sort_values(["_ts"], na_position="last")

    if args.only_epoch and "event" in df.columns:
        df = df[df["event"].astype(str) == "epoch"].copy()

    # Latest row overall
    latest = df.iloc[-1].to_dict()
    print("=== Latest ===")
    print(f"run_id      : {latest.get('run_id')}")
    print(f"timestamp   : {latest.get('timestamp_utc')}")
    print(f"device      : {latest.get('device')}")
    print(f"fold/epoch  : {latest.get('fold')}/{latest.get('epoch')}")
    print(f"event       : {latest.get('event')}")
    if pd.notna(latest.get("val_rmse")):
        print(f"val_rmse    : {latest.get('val_rmse')}")

    # Per-fold summary (based on end-of-epoch rows)
    if not {"fold", "event", "val_rmse", "epoch"}.issubset(set(df.columns)):
        print("\nMissing expected columns for per-fold summary (fold/event/val_rmse/epoch).")
        return 0

    epoch_rows = df[df["event"].astype(str) == "epoch"].copy()
    if epoch_rows.empty:
        print("\nNo end-of-epoch rows found yet (event=epoch).")
        return 0

    # latest per fold
    epoch_rows["epoch"] = pd.to_numeric(epoch_rows["epoch"], errors="coerce")
    epoch_rows["val_rmse"] = pd.to_numeric(epoch_rows["val_rmse"], errors="coerce")

    latest_per_fold = (
        epoch_rows.sort_values(["fold", "epoch"]).groupby("fold", as_index=False).tail(1).sort_values("fold")
    )

    best_per_fold = (
        epoch_rows.dropna(subset=["val_rmse"])  # type: ignore[arg-type]
        .sort_values(["fold", "val_rmse", "epoch"])
        .groupby("fold", as_index=False)
        .head(1)
        .sort_values("fold")
    )

    print("\n=== Per-Fold (latest epoch row) ===")
    cols = ["fold", "epoch", "val_rmse", "val_mae", "val_r", "val_r2", "best_epoch", "best_val_rmse"]
    cols = [c for c in cols if c in latest_per_fold.columns]
    print(latest_per_fold[cols].to_string(index=False))

    print("\n=== Per-Fold (best val_rmse so far) ===")
    cols2 = ["fold", "epoch", "val_rmse", "val_mae", "val_r", "val_r2"]
    cols2 = [c for c in cols2 if c in best_per_fold.columns]
    print(best_per_fold[cols2].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
