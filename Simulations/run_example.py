import glob
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hemipy

img_size = np.array([4000, 4000])
opt_cen  = np.array([2000, 2000])
cal_fun  = np.array([0, 0, 0.09])  # adjust if you get better calibration

lat = 51.7734
date = "2024-01-01"
down_factor = 1

zenith = hemipy.zenith(img_size, opt_cen, cal_fun, down_factor=down_factor)
azimuth = hemipy.azimuth(img_size, opt_cen, down_factor=down_factor)

roots = [
    REPO_ROOT / r"Simulations\DHP - ERECT - 4000x4000\DHP - ERECT - 4000x4000",
    REPO_ROOT / r"Simulations\DHP - PLANO - 4000x4000\DHP - PLANO - 4000x4000",
    REPO_ROOT / r"Simulations\DHP - RND - 4000x4000\DHP - RND - 4000x4000",
]

out_path = HERE / "simulations_output.csv"
with open(out_path, "w") as f:
    f.write("Root,Case,Plot,Direction,PAIe_Hinge,PAI_Hinge,Clumping_Hinge,PAIe_Miller,PAI_Miller,Clumping_Miller,FIPAR,FCOVER\n")
    for root in roots:
        for case in sorted(glob.glob(root + r"\Case *")):
            for plot in sorted(glob.glob(case + r"\*")):
                direction = "up"
                print("processing", root, case, plot)
                res = hemipy.process(
                    img_dir=plot,
                    zenith=zenith,
                    azimuth=azimuth,
                    date=date,
                    lat=lat,
                    direction=direction,
                    down_factor=down_factor,
                )
                f.write(
                    f"{root.split('\\')[-1]},{case.split('\\')[-1]},{plot.split('\\')[-1]},"
                    f"{direction},{res['paie_hinge']},{res['pai_hinge']},{res['clumping_hinge']},"
                    f"{res['paie_miller']},{res['pai_miller']},{res['clumping_miller']},"
                    f"{res['fipar']},{res['fcover']}\n"
                )
                f.flush()

print("Wrote", out_path)