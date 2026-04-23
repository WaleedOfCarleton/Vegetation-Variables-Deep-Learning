from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np
from uncertainties import unumpy

# Allow import of legacy hemipy baseline
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hemipy_core"))
import hemipy  # noqa: E402


DEFAULT_IMG_SIZE = np.array([3465, 5202])
DEFAULT_OPT_CEN = np.array([1754, 2595])
DEFAULT_CAL_FUN = np.array([0, 0, 0.0548543])
DEFAULT_LAT = 51.7734


def _scale_calibration(target_size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale optical centre and calibration polynomial if images are resized.

    Assumes uniform scaling from the original calibration size. For distance-based
    polynomial d -> s*d, coefficients scale as c3/s^3, c2/s^2, c1/s.
    """

    s_h = target_size[0] / DEFAULT_IMG_SIZE[0]
    s_w = target_size[1] / DEFAULT_IMG_SIZE[1]
    s = (s_h + s_w) / 2.0  # approximate uniform scaling

    opt = DEFAULT_OPT_CEN * np.array([s_h, s_w])
    cal = np.array(
        [
            DEFAULT_CAL_FUN[0] / (s**3) if s != 0 else DEFAULT_CAL_FUN[0],
            DEFAULT_CAL_FUN[1] / (s**2) if s != 0 else DEFAULT_CAL_FUN[1],
            DEFAULT_CAL_FUN[2] / s if s != 0 else DEFAULT_CAL_FUN[2],
        ]
    )
    return opt, cal


def _image_sizes(img_dir: Path) -> list[Tuple[int, int, Path]]:
    import imageio.v2 as iio  # lightweight read

    exts = (".nef", ".cr2", ".cr3", ".pef", ".raw", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff")
    sizes: list[Tuple[int, int, Path]] = []
    for p in sorted(img_dir.glob("*")):
        if p.suffix.lower() in exts and p.is_file():
            try:
                arr = iio.imread(p)
                if arr.ndim >= 2:
                    h, w = arr.shape[0], arr.shape[1]
                    sizes.append((h, w, p))
            except Exception:
                continue
    if not sizes:
        raise FileNotFoundError("No images found to infer size")
    return sizes


def _select_mode_size(sizes: Iterable[Tuple[int, int, Path]]) -> Tuple[int, int]:
    counter = Counter((h, w) for h, w, _ in sizes)
    (h, w), _ = counter.most_common(1)[0]
    return h, w


def _nom(val: Any) -> float:
    try:
        return float(unumpy.nominal_values(val))
    except Exception:
        try:
            return float(val)
        except Exception:
            return float("nan")


def run_baseline(img_dir: Path, *, date: str, lat: float, direction: str = "up") -> dict:
    sizes = _image_sizes(img_dir)
    target_hw = np.array(_select_mode_size(sizes))
    opt_cen, cal_fun = _scale_calibration(target_hw)

    # Keep only images matching the modal size to avoid broadcast errors
    matched: list[Path] = [p for (h, w, p) in sizes if (h, w) == tuple(target_hw)]
    if not matched:
        raise RuntimeError("No images matched the modal size—cannot proceed")

    tmpdir = Path(tempfile.mkdtemp(prefix="hemipy_baseline_"))
    try:
        for p in matched:
            shutil.copy2(p, tmpdir / p.name)

        zen = hemipy.zenith(target_hw, opt_cen, cal_fun, down_factor=1)
        azi = hemipy.azimuth(target_hw, opt_cen, down_factor=1)

        res = hemipy.process(
            img_dir=str(tmpdir),
            zenith=zen,
            azimuth=azi,
            date=date,
            lat=lat,
            direction=direction,
            down_factor=1,  # keep image/zenith/azimuth shapes aligned
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    out = {
        "img_dir": str(img_dir),
        "date": date,
        "lat": lat,
        "direction": direction,
        "paie_hinge": _nom(res.get("paie_hinge")),
        "pai_hinge": _nom(res.get("pai_hinge")),
        "clumping_hinge": _nom(res.get("clumping_hinge")),
        "paie_miller": _nom(res.get("paie_miller")),
        "pai_miller": _nom(res.get("pai_miller")),
        "clumping_miller": _nom(res.get("clumping_miller")),
        "fcover": _nom(res.get("fcover")),
        "fipar": _nom(res.get("fipar")),
    }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run legacy hemipy baseline on a folder of fisheye images and write CSV.")
    p.add_argument("--images-dir", required=True, help="Folder with images (all processed together)")
    p.add_argument("--date", default=_dt.date.today().isoformat(), help="Acquisition date YYYY-MM-DD (default: today)")
    p.add_argument("--lat", type=float, default=DEFAULT_LAT, help=f"Latitude in degrees (default: {DEFAULT_LAT})")
    p.add_argument("--direction", choices=["up", "down"], default="up", help="Camera direction (default: up)")
    p.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path (default: <images-dir>/baseline_hemipy_results.csv)",
    )
    p.add_argument("--per-image", action="store_true", help="Write per-image rows instead of a single folder aggregate")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    img_dir = Path(args.images_dir).resolve()
    if not img_dir.exists():
        raise FileNotFoundError(f"images-dir not found: {img_dir}")

    out_csv = Path(args.out_csv) if args.out_csv else img_dir / "baseline_hemipy_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    import csv

    if not args.per_image:
        res = run_baseline(img_dir, date=args.date, lat=float(args.lat), direction=args.direction)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(res.keys()))
            writer.writeheader()
            writer.writerow(res)
        print(f"Wrote baseline results to: {out_csv}")
        return 0

    # Per-image mode
    sizes = _image_sizes(img_dir)
    target_hw = np.array(_select_mode_size(sizes))
    opt_cen, cal_fun = _scale_calibration(target_hw)
    zen = hemipy.zenith(target_hw, opt_cen, cal_fun, down_factor=1)
    azi = hemipy.azimuth(target_hw, opt_cen, down_factor=1)

    matched: list[Path] = [p for (h, w, p) in sizes if (h, w) == tuple(target_hw)]
    if not matched:
        raise RuntimeError("No images matched the modal size—cannot proceed")

    rows = []
    tmp_root = Path(tempfile.mkdtemp(prefix="hemipy_per_image_"))
    try:
        for p in matched:
            sub = tmp_root / p.stem
            sub.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, sub / p.name)

            res = hemipy.process(
                img_dir=str(sub),
                zenith=zen,
                azimuth=azi,
                date=args.date,
                lat=float(args.lat),
                direction=args.direction,
                down_factor=1,
            )

            rows.append(
                {
                    "filename": p.name,
                    "paie_hinge": _nom(res.get("paie_hinge")),
                    "pai_hinge": _nom(res.get("pai_hinge")),
                    "clumping_hinge": _nom(res.get("clumping_hinge")),
                    "paie_miller": _nom(res.get("paie_miller")),
                    "pai_miller": _nom(res.get("pai_miller")),
                    "clumping_miller": _nom(res.get("clumping_miller")),
                    "fcover": _nom(res.get("fcover")),
                    "fipar": _nom(res.get("fipar")),
                }
            )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote per-image baseline results to: {out_csv} (rows={len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
