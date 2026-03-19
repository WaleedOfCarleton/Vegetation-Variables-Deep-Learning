from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Iterable

from plot_case_sweep import _load_points as load_points_from_summary, _plot, _METRIC_CHOICES


def _count_lines(path: Path) -> int:
    if not path.exists():
        return -1
    text = path.read_text(encoding="utf-8")
    return len([ln for ln in text.splitlines() if ln.strip()])


def _best_rows_from_metrics(run_dir: Path, metric: str) -> list[dict]:
    rows: list[dict] = []
    for metrics_csv in sorted(run_dir.glob("train_*_cases/metrics.csv")):
        run_subdir = metrics_csv.parent
        # Prefer the intended train count encoded in the folder name (train_XXX_cases)
        try:
            n_train_from_dir = int(run_subdir.name.split("_")[1])
        except Exception:
            n_train_from_dir = -1

        with metrics_csv.open("r", newline="", encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        if not data:
            continue
        if metric not in data[0]:
            raise RuntimeError(f"Metric '{metric}' not in metrics.csv columns for {metrics_csv}")
        best = min(data, key=lambda r: float(r[metric]))

        splits_dir = metrics_csv.parent / "splits"
        n_train = n_train_from_dir if n_train_from_dir > 0 else _count_lines(splits_dir / "train_cases.txt")
        n_val = _count_lines(splits_dir / "val_cases.txt")
        rows.append(
            {
                "n_train_cases": n_train,
                "n_val_cases": n_val,
                "best_epoch": int(best["epoch"]),
                "train_loss": float(best["train_loss"]),
                "val_loss": float(best["val_loss"]),
                "val_mae_image": float(best["val_mae_image"]),
                "val_rmse_image": float(best["val_rmse_image"]),
                "val_mae_case": float(best["val_mae_case"]),
                "lr": float(best["lr"]),
                "run_dir": metrics_csv.parent.as_posix(),
            }
        )
    if not rows:
        raise RuntimeError(f"No metrics.csv files found under {run_dir}")
    return sorted(rows, key=lambda r: r["n_train_cases"])


def _write_summary(run_dir: Path, rows: Iterable[dict]) -> Path:
    rows = list(rows)
    summary_path = run_dir / "case_sweep_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def _write_aggregate(run_dir: Path, metric: str, rows: list[dict]) -> Path:
    agg = {
        "best_metric": metric,
        "train_counts": [r["n_train_cases"] for r in rows],
        "n_total_cases": max(r["n_train_cases"] + r["n_val_cases"] for r in rows),
        "val_mae_case_min": min(r["val_mae_case"] for r in rows),
        "val_mae_case_mean": mean(r["val_mae_case"] for r in rows),
        "val_mae_case_argmin_train_cases": min(rows, key=lambda r: r["val_mae_case"])["n_train_cases"],
    }
    agg_path = run_dir / "case_sweep_aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    return agg_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build case sweep summary/aggregate and plot.")
    p.add_argument("--run-dir", required=True, help="Sweep folder containing train_*_cases subfolders")
    p.add_argument("--metric", default="val_mae_case", choices=sorted(_METRIC_CHOICES))
    p.add_argument("--plot", action="store_true", help="Also write case_sweep_plot.png")
    p.add_argument("--out-plot", default=None, help="Optional override for plot path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    rows = _best_rows_from_metrics(run_dir, args.metric)
    summary_path = _write_summary(run_dir, rows)
    agg_path = _write_aggregate(run_dir, args.metric, rows)

    if args.plot:
        points = load_points_from_summary(run_dir, args.metric)
        out_path = Path(args.out_plot) if args.out_plot else (run_dir / "case_sweep_plot.png")
        _plot(points, args.metric, out_path)
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote aggregate: {agg_path}")
    if args.plot:
        print(f"Wrote plot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
