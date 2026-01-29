from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from PIL import Image
except Exception as e:
    raise SystemExit("Missing dependency: Pillow. Install with: pip install pillow") from e


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
DEFAULT_INDEX = REPO_ROOT / "shared" / "dataset_index" / "image_dataset_index.csv"
DEFAULT_OUT = HERE / "image_baseline_metrics.csv"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = int(y_true.size)
    if n == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "r": np.nan, "r2": np.nan}

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    r = np.nan
    if n >= 2:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])

    sse = float(np.sum((y_pred - y_true) ** 2))
    sst = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
    return {"n": n, "mae": mae, "rmse": rmse, "bias": bias, "r": r, "r2": r2}


def make_case_folds(case_series: pd.Series, kfold: int, seed: int) -> pd.Series:
    cases = case_series.astype(str)
    uniq = sorted([c for c in cases.dropna().unique().tolist() if c.strip() != ""])
    if len(uniq) < kfold:
        raise ValueError(f"Not enough unique cases ({len(uniq)}) for kfold={kfold}")
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_map = {c: int(i % kfold) for i, c in enumerate(uniq)}
    return cases.map(fold_map).astype(int)


def extract_features(image_path: Path) -> dict:
    im = Image.open(image_path).convert("L")  # grayscale
    arr = np.asarray(im, dtype=np.float32) / 255.0

    mean_int = float(arr.mean())
    std_int = float(arr.std())
    dark_frac = float((arr < 0.5).mean())

    # Simple edge-ish measure (no extra deps)
    dx = np.abs(arr[:, 1:] - arr[:, :-1])
    dy = np.abs(arr[1:, :] - arr[:-1, :])
    edge_mean = float(0.5 * (dx.mean() + dy.mean()))

    return {
        "mean_int": mean_int,
        "std_int": std_int,
        "dark_frac": dark_frac,
        "edge_mean": edge_mean,
    }


def fit_linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Adds intercept automatically
    X2 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    beta, *_ = np.linalg.lstsq(X2, y, rcond=None)
    return beta


def predict_linear_regression(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    X2 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    return X2 @ beta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for quick tests.")
    args = p.parse_args()

    df = pd.read_csv(args.index)
    df = df.dropna(subset=["image_path", "case_norm", "orientation", "truth_PAI"]).copy()

    if args.max_images:
        df = df.head(int(args.max_images)).copy()

    # Extract features
    feats = []
    for rel in df["image_path"].tolist():
        fp = (HERE / rel).resolve()
        f = extract_features(fp)
        feats.append(f)

    feat_df = pd.DataFrame(feats)
    df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

    df["fold"] = make_case_folds(df["case_norm"], kfold=int(args.kfold), seed=int(args.seed))

    feature_cols = ["mean_int", "std_int", "dark_frac", "edge_mean"]

    # Case-based CV, evaluated at (a) image-level and (b) case+orientation averaged
    oof_preds = np.full(len(df), np.nan, dtype=np.float64)

    for fold in range(int(args.kfold)):
        train = df[df["fold"] != fold]
        test = df[df["fold"] == fold]

        Xtr = train[feature_cols].to_numpy(dtype=np.float64)
        ytr = train["truth_PAI"].to_numpy(dtype=np.float64)
        Xte = test[feature_cols].to_numpy(dtype=np.float64)

        beta = fit_linear_regression(Xtr, ytr)
        yhat = predict_linear_regression(beta, Xte)
        oof_preds[test.index.to_numpy()] = yhat

    df["pred_PAI_linear"] = oof_preds

    # Metrics (image-level)
    m_img = regression_metrics(df["truth_PAI"].to_numpy(), df["pred_PAI_linear"].to_numpy())

    # Metrics (average predictions per case+orientation)
    g = df.groupby(["case_norm", "orientation"], as_index=False).agg(
        truth_PAI=("truth_PAI", "mean"),
        pred_PAI_linear=("pred_PAI_linear", "mean"),
    )
    m_grp = regression_metrics(g["truth_PAI"].to_numpy(), g["pred_PAI_linear"].to_numpy())

    out_rows = [
        {"level": "image", **m_img},
        {"level": "case_orientation_mean", **m_grp},
    ]
    pd.DataFrame(out_rows).to_csv(args.out, index=False, float_format="%.4f")

    print("Wrote:", args.out)
    print("Samples:", len(df), "Groups:", len(g))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())