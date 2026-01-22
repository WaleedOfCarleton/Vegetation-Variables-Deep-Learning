from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_SIM_ROOT = REPO_ROOT / "Simulations"
DEFAULT_TRUTH = REPO_ROOT / "truth_join" / "truth_joined_to_hemipy.csv"
DEFAULT_OUT = HERE / "image_dataset_index.csv"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


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
        "--include-rnd",
        action="store_true",
        help="Include RND images too (truth_PAI will likely be missing for them).",
    )
    args = p.parse_args()

    sim_root = args.sim_root.resolve()
    if not sim_root.exists():
        raise FileNotFoundError(f"Sim root not found: {sim_root}")

    rows: list[dict] = []
    for fp in sim_root.rglob("*"):
        if not fp.is_file():
            continue
        if fp.name.lower() == "thumbs.db":
            continue
        if fp.suffix.lower() not in IMG_EXTS:
            continue

        orientation = _extract_orientation(fp)
        case_norm = _extract_case_norm(fp)
        plot = _extract_plot(fp)

        if not args.include_rnd and orientation == "RND":
            continue

        rows.append(
            {
                "image_path": fp.resolve().relative_to(HERE).as_posix(),
                "filename": fp.name,
                "simulation_set": next((p for p in fp.parts if "DHP -" in p), None),
                "orientation": orientation,
                "case_norm": case_norm,
                "plot": plot,
            }
        )

    index_df = pd.DataFrame(rows)
    if index_df.empty:
        raise RuntimeError("No images found. Check --sim-root and extensions.")

    # Join truth labels (many images -> one truth per case_norm+orientation)
    truth = pd.read_csv(args.truth_csv)
    need_cols = {"case_norm", "orientation", "truth_PAI"}
    missing = need_cols - set(truth.columns)
    if missing:
        raise ValueError(f"Truth CSV missing columns: {sorted(missing)}")

    truth_small = truth[["case_norm", "orientation", "truth_PAI", "truth_LAI", "truth_Clumping", "truth_sim_id"]].copy()
    out = index_df.merge(
        truth_small,
        on=["case_norm", "orientation"],
        how="left",
        validate="many_to_one",
    )

    out.to_csv(args.out, index=False)
    print("Wrote:", args.out)
    print("Images:", len(out))
    print("Missing truth_PAI:", int(out["truth_PAI"].isna().sum()))
    print("Orientations:\n", out["orientation"].value_counts(dropna=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())