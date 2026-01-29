from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: matplotlib.\n"
        "Install in your env with one of:\n"
        "  conda install -n hemipy-gpu -c defaults matplotlib\n"
        "  pip install matplotlib\n"
    ) from e


HERE = Path(__file__).resolve().parent
DEFAULT_OUTDIR = HERE


def _safe_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _best_oof_metrics(metrics_df: pd.DataFrame, preferred_level: str) -> dict:
    if metrics_df.empty:
        return {}

    if not {"fold", "level"}.issubset(set(metrics_df.columns)):
        return {}

    df = metrics_df.copy()
    df = df[(df["fold"].astype(str) == "OOF") & (df["level"].astype(str) == preferred_level)]
    if df.empty:
        df = metrics_df[metrics_df["fold"].astype(str) == "OOF"].copy()
    if df.empty:
        return {}

    row = df.iloc[0].to_dict()
    keys = ["n", "mae", "rmse", "bias", "r", "r2"]
    return {k: _safe_float(row.get(k)) for k in keys}


def _scatter_truth_pred(
    df: pd.DataFrame,
    truth_col: str,
    pred_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Path,
    subtitle_metrics: dict | None,
    *,
    dpi: int,
    figsize: tuple[float, float],
    point_size: float,
    font_size: float,
) -> None:
    d = df[[truth_col, pred_col]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        raise SystemExit(f"No finite rows found for columns: {truth_col}, {pred_col}")

    x = d[truth_col].to_numpy(dtype=float)
    y = d[pred_col].to_numpy(dtype=float)

    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(x, y, s=point_size, alpha=0.55, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.2)

    if subtitle_metrics:
        rmse = subtitle_metrics.get("rmse")
        r2 = subtitle_metrics.get("r2")
        r = subtitle_metrics.get("r")
        mae = subtitle_metrics.get("mae")
        parts = []
        if np.isfinite(_safe_float(rmse)):
            parts.append(f"RMSE={rmse:.3f}")
        if np.isfinite(_safe_float(mae)):
            parts.append(f"MAE={mae:.3f}")
        if np.isfinite(_safe_float(r)):
            parts.append(f"r={r:.3f}")
        if np.isfinite(_safe_float(r2)):
            parts.append(f"R2={r2:.3f}")
        if parts:
            ax.text(
                0.02,
                0.98,
                "  ".join(parts),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=max(10.0, font_size * 0.8),
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
            )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def _residual_plot(
    df: pd.DataFrame,
    truth_col: str,
    pred_col: str,
    title: str,
    xlabel: str,
    outpath: Path,
    *,
    dpi: int,
    figsize: tuple[float, float],
    point_size: float,
) -> None:
    d = df[[truth_col, pred_col]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        raise SystemExit(f"No finite rows found for columns: {truth_col}, {pred_col}")

    x = d[truth_col].to_numpy(dtype=float)
    resid = d[pred_col].to_numpy(dtype=float) - x

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(x, resid, s=point_size, alpha=0.55, edgecolors="none")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Residual (pred - truth)")
    ax.grid(True, which="major", alpha=0.2)

    resid_finite = resid[np.isfinite(resid)]
    if resid_finite.size:
        lim = float(np.nanpercentile(np.abs(resid_finite), 99))
        if np.isfinite(lim) and lim > 0:
            ax.set_ylim(-lim, lim)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Visualize CNN hinge outputs (truth vs pred + residuals).")
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--per-image", type=Path, required=True)
    p.add_argument("--per-site", type=Path, required=True)
    p.add_argument("--truth-col", type=str, default="truth_PAIe_hinge")
    p.add_argument("--pred-col", type=str, default="pred_PAIe_hinge_cnn")
    p.add_argument("--preferred-level", type=str, default="case_orientation_mean")
    p.add_argument("--xlabel", type=str, default="Truth")
    p.add_argument("--ylabel", type=str, default="Predicted (CNN)")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--prefix", type=str, default="")
    p.add_argument("--dpi", type=int, default=250)
    p.add_argument("--font-size", type=float, default=14.0)
    p.add_argument("--point-size", type=float, default=22.0)
    p.add_argument("--scatter-figsize", type=str, default="10,8")
    p.add_argument("--resid-figsize", type=str, default="10,6")
    args = p.parse_args()

    def _parse_figsize(value: str) -> tuple[float, float]:
        try:
            w, h = value.split(",")
            return float(w.strip()), float(h.strip())
        except Exception as e:
            raise SystemExit(f"Invalid figsize '{value}'. Expected 'W,H' (e.g., 10,8).") from e

    scatter_figsize = _parse_figsize(args.scatter_figsize)
    resid_figsize = _parse_figsize(args.resid_figsize)

    plt.rcParams.update(
        {
            "font.size": args.font_size,
            "axes.titlesize": args.font_size * 1.05,
            "axes.labelsize": args.font_size,
            "xtick.labelsize": args.font_size * 0.9,
            "ytick.labelsize": args.font_size * 0.9,
        }
    )

    metrics_df = pd.read_csv(args.metrics) if args.metrics.exists() else pd.DataFrame()
    oof = _best_oof_metrics(metrics_df, preferred_level=args.preferred_level)

    per_image = pd.read_csv(args.per_image)
    per_site = pd.read_csv(args.per_site)

    for df, name in [(per_image, "per-image"), (per_site, "per-site")]:
        if args.truth_col not in df.columns or args.pred_col not in df.columns:
            raise SystemExit(
                f"Expected columns {args.truth_col} and {args.pred_col} in {name} CSV. Found: {sorted(df.columns.tolist())}"
            )

    outdir = Path(args.outdir)
    prefix = args.prefix.strip()
    if prefix and not prefix.endswith("_"):
        prefix += "_"

    _scatter_truth_pred(
        per_site,
        truth_col=args.truth_col,
        pred_col=args.pred_col,
        title=f"CNN hinge: Truth vs Pred ({args.preferred_level})",
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        outpath=outdir / f"{prefix}cnn_hinge_truth_vs_pred_site.png",
        subtitle_metrics=oof,
        dpi=args.dpi,
        figsize=scatter_figsize,
        point_size=args.point_size,
        font_size=args.font_size,
    )
    _residual_plot(
        per_site,
        truth_col=args.truth_col,
        pred_col=args.pred_col,
        title=f"CNN hinge: Residuals vs Truth ({args.preferred_level})",
        xlabel=args.xlabel,
        outpath=outdir / f"{prefix}cnn_hinge_residuals_site.png",
        dpi=args.dpi,
        figsize=resid_figsize,
        point_size=args.point_size,
    )

    _scatter_truth_pred(
        per_image,
        truth_col=args.truth_col,
        pred_col=args.pred_col,
        title="CNN hinge: Truth vs Pred (image-level)",
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        outpath=outdir / f"{prefix}cnn_hinge_truth_vs_pred_image.png",
        subtitle_metrics=None,
        dpi=args.dpi,
        figsize=scatter_figsize,
        point_size=args.point_size,
        font_size=args.font_size,
    )
    _residual_plot(
        per_image,
        truth_col=args.truth_col,
        pred_col=args.pred_col,
        title="CNN hinge: Residuals vs Truth (image-level)",
        xlabel=args.xlabel,
        outpath=outdir / f"{prefix}cnn_hinge_residuals_image.png",
        dpi=args.dpi,
        figsize=resid_figsize,
        point_size=args.point_size,
    )

    print("Wrote plots to:", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
