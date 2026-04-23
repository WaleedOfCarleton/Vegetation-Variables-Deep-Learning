from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path
import sys

import pandas as pd

# Make repo root importable for ml.* modules
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse baseline runner
from ml.tools.run_baseline_hemipy_folder import DEFAULT_LAT, run_baseline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run hemipy baseline on Sunny Hemiphotos cases and attach truth values.")
    p.add_argument("--sunny-root", default="Simulations/Sunny Hemiphotos/RND", help="Root folder containing Case */ directories")
    p.add_argument("--truth-csv", default="shared/truth_join/truth_joined_to_hemipy.csv", help="Truth + baseline CSV")
    p.add_argument("--date", default=_dt.date.today().isoformat(), help="Acquisition date (default: today)")
    p.add_argument("--lat", type=float, default=DEFAULT_LAT, help=f"Latitude (default: {DEFAULT_LAT})")
    p.add_argument("--direction", choices=["up", "down"], default="up")
    p.add_argument("--out-csv", default="Simulations/Sunny Hemiphotos/baseline_sunny_with_truth.csv", help="Output CSV path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.sunny_root)
    truth_df = pd.read_csv(args.truth_csv)

    rows = []
    for case_dir in sorted(p for p in root.glob("Case */") if p.is_dir()):
        case_name = case_dir.name.strip()
        try:
            res = run_baseline(case_dir, date=args.date, lat=float(args.lat), direction=args.direction)
        except Exception as exc:  # pragma: no cover
            print(f"Skipping {case_name} due to error: {exc}")
            continue

        truth_row = truth_df[truth_df.get("case_norm") == case_name]
        truth_pai = truth_row["truth_PAI"].iloc[0] if not truth_row.empty else None
        truth_clump = truth_row["truth_Clumping"].iloc[0] if not truth_row.empty else None

        rows.append(
            {
                "case": case_name,
                **res,
                "truth_PAI": truth_pai,
                "truth_Clumping": truth_clump,
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote sunny baseline + truth to: {out_path} (rows={len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
