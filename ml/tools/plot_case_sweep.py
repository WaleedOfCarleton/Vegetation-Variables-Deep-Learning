from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


# Metrics available in run_sweep_train_cases metrics.csv files
_METRIC_CHOICES = {"val_loss", "val_mae_case", "val_mae_image", "val_rmse_image"}


@dataclass(frozen=True)
class Point:
    n_train: int
    n_val: int
    metric: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot case sweep results (metric vs number of training cases).")
    p.add_argument("--run-dir", required=True, type=str, help="Sweep folder containing train_*_cases subfolders.")
    p.add_argument("--metric", default="val_mae_case", choices=sorted(_METRIC_CHOICES), help="Metric to plot (best over epochs).")
    p.add_argument("--out", default=None, help="PNG output path (default: <run-dir>/case_sweep_plot.png)")
    p.add_argument("--epochs", type=int, default=None, help="Optional max epochs to show in title (overrides auto-detect).")
    p.add_argument("--patience", type=int, default=None, help="Optional patience to show in title (overrides auto-detect).")
    p.add_argument("--x-ticks", type=str, default=None, help="Comma-separated list of x tick values to force (e.g., 5,15,25).")
    p.add_argument("--y-ticks", type=str, default=None, help="Comma-separated list of y tick values to force (e.g., 0.1,0.2,0.3).")
    p.add_argument("--y-lims", type=str, default=None, help="Comma-separated y-axis limits 'ymin,ymax' to force (e.g., 0.1,0.6).")
    return p.parse_args()


def _read_summary(summary_csv: Path, metric: str) -> list[Point]:
    points: list[Point] = []
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append(
                Point(
                    n_train=int(row["n_train_cases"]),
                    n_val=int(row["n_val_cases"]),
                    metric=float(row[metric]),
                )
            )
    return points


def _best_from_metrics(metrics_csv: Path, metric: str) -> Point:
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows in {metrics_csv}")
    if metric not in rows[0]:
        raise RuntimeError(f"Metric '{metric}' not found in {metrics_csv}")
    best = min(rows, key=lambda r: float(r[metric]))
    return Point(
        n_train=int(best.get("n_train_cases", -1)),
        n_val=int(best.get("n_val_cases", -1)),
        metric=float(best[metric]),
    )


def _read_from_train_dirs(run_dir: Path, metric: str) -> list[Point]:
    points: list[Point] = []
    for sub in sorted(run_dir.glob("train_*_cases")):
        metrics_csv = sub / "metrics.csv"
        if not metrics_csv.exists():
            continue
        points.append(_best_from_metrics(metrics_csv, metric))
    if not points:
        raise RuntimeError(f"No metrics.csv files found under {run_dir}")
    return points


def _load_points(run_dir: Path, metric: str) -> list[Point]:
    summary_csv = run_dir / "case_sweep_summary.csv"
    if summary_csv.exists():
        return _read_summary(summary_csv, metric)
    return _read_from_train_dirs(run_dir, metric)


def _detect_training_meta(run_dir: Path) -> tuple[int | None, int | None]:
    # Try to read epochs/patience from the first train_*_cases config.json
    for sub in sorted(run_dir.glob("train_*_cases")):
        cfg = sub / "config.json"
        if not cfg.exists():
            continue
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            return int(data.get("epochs")) if "epochs" in data else None, int(data.get("patience")) if "patience" in data else None
        except Exception:
            continue
    return None, None


def _maybe_parse_float_list(text: str | None) -> list[float] | None:
    if text is None:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    vals: list[float] = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            raise ValueError(f"Could not parse float from '{p}' in '{text}'")
    return vals


def _plot(points: Iterable[Point], metric: str, out_path: Path, *, epochs: int | None, patience: int | None, run_dir: Path, x_ticks: list[float] | None, y_ticks: list[float] | None, y_lims: list[float] | None) -> None:
    pts = sorted(points, key=lambda p: p.n_train)
    xs = [p.n_train for p in pts]
    ys = [p.metric for p in pts]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, marker="o", lw=1.5)
    ax.set_xlabel("Number of Train Cases")
    ax.set_ylabel(metric)
    meta_epochs, meta_patience = _detect_training_meta(run_dir)
    e = epochs if epochs is not None else meta_epochs
    ptn = patience if patience is not None else meta_patience
    val_counts = sorted({p.n_val for p in pts})
    title = "Consistent Number val cases" if len(val_counts) == 1 else "Changing Number val cases"

    extras = []
    if e is not None:
        extras.append(f"epochs={e}")
    if ptn is not None:
        extras.append(f"patience={ptn}")
    if extras:
        title = f"{title} ({', '.join(extras)})"
    ax.set_title(title)

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_lims is not None:
        if len(y_lims) != 2:
            raise ValueError("--y-lims expects two comma-separated numbers: ymin,ymax")
        ax.set_ylim(y_lims[0], y_lims[1])

    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    points = _load_points(run_dir, args.metric)
    out_path = Path(args.out) if args.out else (run_dir / "case_sweep_plot.png")
    x_ticks = _maybe_parse_float_list(args.x_ticks)
    y_ticks = _maybe_parse_float_list(args.y_ticks)
    y_lims = _maybe_parse_float_list(args.y_lims)
    _plot(points, args.metric, out_path, epochs=args.epochs, patience=args.patience, run_dir=run_dir, x_ticks=x_ticks, y_ticks=y_ticks, y_lims=y_lims)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
