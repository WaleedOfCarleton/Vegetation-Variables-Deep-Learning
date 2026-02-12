from __future__ import annotations

import argparse
import re
from pathlib import Path
import csv

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
DEFAULT_SIM_ROOT = REPO_ROOT / "Simulations"
DEFAULT_TRUTH = REPO_ROOT / "shared" / "truth_join" / "truth_joined_to_hemipy.csv"
DEFAULT_OUT = HERE / "image_dataset_index.csv"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _parse_pm_value(x):
    """Parse values like '1.66+/-0.06' -> 1.66.

    Falls back to pandas numeric coercion when possible.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NA
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return pd.NA

    if "+/-" in s:
        left = s.split("+/-", 1)[0].strip()
        try:
            return float(left)
        except Exception:
            return pd.NA

    # Generic numeric parse (handles plain strings)
    v = pd.to_numeric(s, errors="coerce")
    if pd.isna(v):
        return pd.NA
    return float(v)


def _normalize_case(case_part: str) -> str | None:
    m = re.search(r"case\s*(\d+)", case_part, flags=re.IGNORECASE)
    if not m:
        return None
    return f"Case {int(m.group(1)):03d}"


def _extract_orientation(path: Path) -> str | None:
    up = " / ".join([p.upper() for p in path.parts])
    if "ERECT" in up:
        return "ERECT"
    if "PLANO" in up or "PLANAR" in up:
        return "PLANO"
    if "RND" in up or "RANDOM" in up:
        return "RND"
    return None


def _extract_plot(path: Path) -> str | None:
    for part in path.parts:
        if re.fullmatch(r"Plot\d+", part, flags=re.IGNORECASE):
            return part
    return None


def _extract_case_norm(path: Path) -> str | None:
    for part in path.parts:
        if re.search(r"case\s*\d+", part, flags=re.IGNORECASE):
            return _normalize_case(part)
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sim-root", type=Path, default=DEFAULT_SIM_ROOT)
    p.add_argument("--truth-csv", type=Path, default=DEFAULT_TRUTH)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--only-simulation-set",
        action="append",
        default=None,
        help=(
            "Only (re)index these top-level simulation_set folders under --sim-root. "
            "Can be passed multiple times (e.g. --only-simulation-set RND). "
            "When used with --update-existing, this updates just those sets without rescanning everything."
        ),
    )
    p.add_argument(
        "--update-existing",
        type=Path,
        default=None,
        help=(
            "Path to an existing image_dataset_index.csv to update in-place. "
            "Requires --only-simulation-set. Rows for the selected simulation_set(s) are replaced."
        ),
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--include-rnd",
        action="store_true",
        default=None,
        help="Include RND images (default: include). Kept for backward-compatibility.",
    )
    g.add_argument(
        "--exclude-rnd",
        action="store_true",
        default=None,
        help="Exclude RND images.",
    )
    args = p.parse_args()

    if args.update_existing is not None and not args.only_simulation_set:
        raise ValueError("--update-existing requires --only-simulation-set (one or more).")

    include_rnd = True
    if args.include_rnd is True:
        include_rnd = True
    if args.exclude_rnd is True:
        include_rnd = False

    sim_root = args.sim_root.resolve()
    if not sim_root.exists():
        raise FileNotFoundError(f"Sim root not found: {sim_root}")

    scan_roots: list[Path]
    if args.only_simulation_set:
        scan_roots = [(sim_root / s) for s in args.only_simulation_set]
        missing = [str(p) for p in scan_roots if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Some --only-simulation-set folders were not found under --sim-root: " + ", ".join(missing)
            )
    else:
        scan_roots = [sim_root]

    rows: list[dict] = []
    for scan_root in scan_roots:
        for fp in scan_root.rglob("*"):
            if not fp.is_file():
                continue
            if fp.name.lower() == "thumbs.db":
                continue
            if fp.suffix.lower() not in IMG_EXTS:
                continue

            try:
                rel = fp.resolve().relative_to(sim_root)
                simulation_set = rel.parts[0] if rel.parts else None
            except Exception:
                simulation_set = None

            orientation = _extract_orientation(fp)
            case_norm = _extract_case_norm(fp)
            plot = _extract_plot(fp)

            if not include_rnd and orientation == "RND":
                continue

            rows.append(
                {
                    "image_path": fp.resolve().relative_to(REPO_ROOT).as_posix(),
                    "filename": fp.name,
                    "simulation_set": simulation_set,
                    "orientation": orientation,
                    "case_norm": case_norm,
                    "plot": plot,
                }
            )

    index_df = pd.DataFrame(rows)
    if index_df.empty:
        raise RuntimeError("No images found. Check --sim-root and extensions.")

    # Join truth labels
    # For PAI (true PAI), the value is invariant across orientations.
    # We join truth by case_norm only so that RND images can be labeled too.
    truth = pd.read_csv(args.truth_csv)
    need_cols = {"case_norm", "truth_PAI"}
    missing = need_cols - set(truth.columns)
    if missing:
        raise ValueError(f"Truth CSV missing columns: {sorted(missing)}")

    keep_cols = [
        c
        for c in [
            "case_norm",
            "orientation",
            "truth_PAI",
            "truth_LAI",
            "truth_Clumping",
            "truth_sim_id",
            "PAIe_Hinge",
            "Clumping_Hinge",
        ]
        if c in truth.columns
    ]
    truth_small = truth[keep_cols].copy()
    for c in ["truth_PAI", "truth_LAI", "truth_Clumping", "PAIe_Hinge", "Clumping_Hinge"]:
        if c in truth_small.columns:
            # Some hinge columns are strings like '1.66+/-0.06'
            if c in {"PAIe_Hinge", "Clumping_Hinge"}:
                truth_small[c] = truth_small[c].map(_parse_pm_value)
            else:
                truth_small[c] = pd.to_numeric(truth_small[c], errors="coerce")

    def _first_nonnull(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else pd.NA

    agg = {"truth_PAI": "mean"}
    if "truth_LAI" in truth_small.columns:
        agg["truth_LAI"] = "mean"
    if "truth_Clumping" in truth_small.columns:
        agg["truth_Clumping"] = "mean"
    if "truth_sim_id" in truth_small.columns:
        agg["truth_sim_id"] = _first_nonnull

    truth_case = truth_small.groupby("case_norm", as_index=False).agg(agg)
    out = index_df.merge(truth_case, on=["case_norm"], how="left", validate="many_to_one")

    # For hinge-region PAIe and clumping (VZA ~57.3°), the values are orientation-specific.
    # Join on (case_norm, orientation) so ERECT/PLANO/RND keep distinct labels.
    if {"case_norm", "orientation", "PAIe_Hinge"}.issubset(set(truth_small.columns)):
        hinge_cols = [c for c in ["case_norm", "orientation", "PAIe_Hinge", "Clumping_Hinge"] if c in truth_small.columns]
        hinge = truth_small[hinge_cols].copy()
        hinge = hinge.dropna(subset=["case_norm", "orientation"], how="any")
        hinge_agg = {"PAIe_Hinge": "mean"}
        if "Clumping_Hinge" in hinge.columns:
            hinge_agg["Clumping_Hinge"] = "mean"
        hinge = hinge.groupby(["case_norm", "orientation"], as_index=False).agg(hinge_agg)
        hinge = hinge.rename(
            columns={
                "PAIe_Hinge": "truth_PAIe_hinge",
                "Clumping_Hinge": "truth_Clumping_hinge",
            }
        )
        out = out.merge(hinge, on=["case_norm", "orientation"], how="left", validate="many_to_one")

    out.to_csv(args.out, index=False)

    # Optional incremental update: replace selected simulation_set rows in an existing index.
    if args.update_existing is not None:
        replace_sets = set(str(s) for s in (args.only_simulation_set or []))
        existing_path = args.update_existing.resolve()
        if not existing_path.exists():
            raise FileNotFoundError(f"Existing index not found: {existing_path}")

        tmp_out = Path(str(existing_path) + ".tmp")
        with existing_path.open("r", newline="", encoding="utf-8") as f_in:
            reader = csv.DictReader(f_in)
            existing_fields = list(reader.fieldnames or [])
            new_fields = list(out.columns)
            # Preserve existing column order and append any new columns at the end.
            fieldnames = existing_fields + [c for c in new_fields if c not in existing_fields]

            with tmp_out.open("w", newline="", encoding="utf-8") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                writer.writeheader()

                # Keep all existing rows except those in the sets we're replacing.
                for row in reader:
                    if (row.get("simulation_set") or "") in replace_sets:
                        continue
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

                # Append updated rows for the selected sets.
                for row in out.to_dict(orient="records"):
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

        tmp_out.replace(existing_path)
        print("Updated existing index:", existing_path)

    print("Wrote:", args.out)
    print("Images:", len(out))
    print("Missing truth_PAI:", int(out["truth_PAI"].isna().sum()))
    if "truth_PAIe_hinge" in out.columns:
        print("Missing truth_PAIe_hinge:", int(out["truth_PAIe_hinge"].isna().sum()))
    print("Orientations:\n", out["orientation"].value_counts(dropna=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())