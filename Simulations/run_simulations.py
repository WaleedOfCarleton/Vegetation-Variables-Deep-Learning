from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hemipy


# ----------------------------
# Calibration / site metadata
# ----------------------------

img_size = np.array([4000, 4000])

# Boss info: for even-sized images, the true centre is between 4 pixels.
# HemiPy internally does `opt_cen = opt_cen - 1`, so pass the half-pixel centre here.
opt_cen = np.array([2000.5, 2000.5], dtype=float)

# Still provisional (keep your current value until you confirm the correct mapping).
cal_fun = np.array([0.0, 0.0, 90.0 / 1615.0], dtype=float)  # ≈ 0.0557 deg/px
lat = 51.7734
date = "2024-01-01"

# IMPORTANT: must match the zenith/azimuth grids you generate.
down_factor = 1

# For circular fisheye-style images (often zeros outside the lens circle), this helps thresholding.
ignore_zeros = True

# Analysis settings (leave defaults unless you have a reason to change)
direction = "up"
min_zenith = 0
max_zenith = 60
zenith_bin = 10
azimuth_bin = 10
fcover_zenith = 10
use_miller_rings = False


# ----------------------------
# Inputs / outputs
# ----------------------------

roots = [
    REPO_ROOT / r"Simulations\DHP - ERECT - 4000x4000\DHP - ERECT - 4000x4000",
    REPO_ROOT / r"Simulations\DHP - PLANO - 4000x4000\DHP - PLANO - 4000x4000",
    REPO_ROOT / r"Simulations\DHP - RND - 4000x4000\DHP - RND - 4000x4000",
]

out_path = HERE / "simulations_output.csv"
err_path = HERE / "simulations_errors.csv"

expected_images_per_plot = 10


def iter_image_files(img_dir: Path) -> list[Path]:
    exts = {
        ".nef", ".cr2", ".cr3", ".pef", ".raw",
        ".jpg", ".jpeg", ".png", ".gif", ".bmp",
        ".tif", ".tiff",
    }
    if not img_dir.exists() or not img_dir.is_dir():
        return []
    return sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def iter_plot_dirs(case_dir: Path):
    if not case_dir.exists() or not case_dir.is_dir():
        return
    for child in sorted(case_dir.iterdir()):
        if not child.is_dir():
            continue
        # Only treat Plot* as plots (skips “For presentation”, etc.)
        if not child.name.lower().startswith("plot"):
            continue
        if len(iter_image_files(child)) == 0:
            continue
        yield child


# Precompute angle grids once (must match `down_factor` used in process()).
zenith = hemipy.zenith(img_size, opt_cen, cal_fun, down_factor=down_factor)
azimuth = hemipy.azimuth(img_size, opt_cen, down_factor=down_factor)

fieldnames = [
    "Root", "Case", "Plot", "Direction", "N_Images",
    "PAIe_Hinge", "PAI_Hinge", "Clumping_Hinge",
    "PAIe_Miller", "PAI_Miller", "Clumping_Miller",
    "FIPAR", "FCOVER",
]

with out_path.open("w", newline="", encoding="utf-8") as f_out, err_path.open("w", newline="", encoding="utf-8") as f_err:
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    err_writer = csv.DictWriter(
        f_err,
        fieldnames=["Root", "Case", "Plot", "Direction", "Error"],
    )
    err_writer.writeheader()

    for root in roots:
        if not root.exists():
            print("Missing root:", root)
            continue

        for case_dir in sorted(root.glob("Case *")):
            if not case_dir.is_dir():
                continue

            for plot_dir in iter_plot_dirs(case_dir):
                root_name = root.name
                case_name = case_dir.name
                plot_name = plot_dir.name

                image_files = iter_image_files(plot_dir)
                n_images = len(image_files)

                if n_images != expected_images_per_plot:
                    print(f"Warning: {root_name} / {case_name} / {plot_name} has {n_images} images (expected {expected_images_per_plot}).")

                print("processing", root_name, case_name, plot_name)

                try:
                    res = hemipy.process(
                        img_dir=str(plot_dir),
                        zenith=zenith,
                        azimuth=azimuth,
                        date=date,
                        lat=lat,
                        direction=direction,
                        down_factor=down_factor,
                        min_zenith=min_zenith,
                        max_zenith=max_zenith,
                        zenith_bin=zenith_bin,
                        azimuth_bin=azimuth_bin,
                        fcover_zenith=fcover_zenith,
                        use_miller_rings=use_miller_rings,
                        ignore_zeros=ignore_zeros,
                        pre_process_raw=False,  # your sims are PNGs; keep RAW pipeline off
                    )

                    writer.writerow(
                        {
                            "Root": root_name,
                            "Case": case_name,
                            "Plot": plot_name,
                            "Direction": direction,
                            "N_Images": n_images,
                            "PAIe_Hinge": res["paie_hinge"],
                            "PAI_Hinge": res["pai_hinge"],
                            "Clumping_Hinge": res["clumping_hinge"],
                            "PAIe_Miller": res["paie_miller"],
                            "PAI_Miller": res["pai_miller"],
                            "Clumping_Miller": res["clumping_miller"],
                            "FIPAR": res["fipar"],
                            "FCOVER": res["fcover"],
                        }
                    )
                    f_out.flush()

                except Exception as e:
                    err_writer.writerow(
                        {
                            "Root": root_name,
                            "Case": case_name,
                            "Plot": plot_name,
                            "Direction": direction,
                            "Error": repr(e),
                        }
                    )
                    f_err.flush()
                    print("  ERROR:", repr(e))

print("Wrote", out_path)
print("Wrote", err_path)