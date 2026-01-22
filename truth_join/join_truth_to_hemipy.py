from __future__ import annotations

import itertools
import re
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
TRUTH_XLSX = REPO_ROOT / "Inputs Cases LAI.xlsx"
HEMIPY_CSV = REPO_ROOT / "Simulations" / "simulations_output.csv"
OUT_CSV = HERE / "truth_joined_to_hemipy.csv"


def parse_u(s: str):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None, None
    s = str(s).strip()
    m = re.match(
        r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\+/-\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$",
        s,
    )
    if not m:
        try:
            return float(s), None
        except Exception:
            return None, None
    return float(m.group(1)), float(m.group(2))


def normalize_case(x):
    s = str(x).strip()
    digits = re.findall(r"\d+", s)
    if not digits:
        return None
    n = int(digits[0])
    return f"Case {n:03d}"


def normalize_orientation(x):
    s = str(x).strip().upper()
    if "ERECT" in s:
        return "ERECT"
    if "PLANO" in s or "PLANAR" in s:
        return "PLANO"
    if "RND" in s or "RANDOM" in s:
        return "RND"
    return s


def _find_col(df: pd.DataFrame, *names_lower: str) -> str | None:
    for c in df.columns:
        if str(c).strip().lower() in names_lower:
            return c
    return None


def _find_cols_containing(df: pd.DataFrame, token_lower: str) -> list[str]:
    return [c for c in df.columns if token_lower in str(c).strip().lower()]


def _safe_pearson(a: pd.Series, b: pd.Series) -> float | None:
    tmp = pd.concat([a, b], axis=1).dropna()
    if len(tmp) < 5:
        return None
    return float(tmp.corr().iloc[0, 1])


def _coerce_sim_id(series: pd.Series) -> pd.Series:
    sid = pd.to_numeric(series, errors="coerce")
    if sid.isna().any():
        extracted = series.astype(str).str.extract(r"(\d+)", expand=False)
        sid = sid.fillna(pd.to_numeric(extracted, errors="coerce"))
    return sid


def _map_sim_id_long(sim_id: pd.Series, n_cases: int, orientations: list[str], mode: str) -> pd.DataFrame:
    sid = _coerce_sim_id(sim_id)
    if sid.isna().any():
        raise ValueError("Some Sim ID values are not numeric; cannot map reliably.")
    sid = sid.astype(int)

    sid0 = sid - 1
    n_oris = len(orientations)

    if mode == "cycle":
        case_num = (sid0 // n_oris) + 1
        ori_idx = (sid0 % n_oris)
    elif mode == "block":
        case_num = (sid0 % n_cases) + 1
        ori_idx = (sid0 // n_cases)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return pd.DataFrame(
        {
            "case_norm": case_num.map(lambda n: f"Case {int(n):03d}"),
            "orientation": ori_idx.map(lambda i: orientations[int(i)] if 0 <= int(i) < n_oris else None),
        }
    )


# --- Load HemiPy output ---
hemi = pd.read_csv(HEMIPY_CSV)
hemi["orientation"] = (
    hemi["Root"]
    .astype(str)
    .str.extract(r"DHP\s*-\s*([A-Z]+)\s*-", expand=False)
    .fillna("")
    .map(normalize_orientation)
)
hemi["case_norm"] = hemi["Case"].map(normalize_case)

# Exclude RND entirely (no truth PAI column for it yet)
hemi = hemi[hemi["orientation"].isin(["ERECT", "PLANO"])].copy()

for col in ["PAI_Hinge", "Clumping_Hinge", "PAI_Miller", "Clumping_Miller", "FIPAR", "FCOVER"]:
    vals = hemi[col].apply(parse_u)
    hemi[f"{col}_value"] = vals.apply(lambda t: t[0])
    hemi[f"{col}_stderr"] = vals.apply(lambda t: t[1])

hemi_oris = sorted([o for o in hemi["orientation"].dropna().unique().tolist() if str(o).strip() != ""])
n_cases = hemi["case_norm"].nunique()
n_oris = len(hemi_oris)

print("Hemi cases:", n_cases, "Hemi orientations:", hemi_oris)

# --- Load Truth Excel ---
if not TRUTH_XLSX.exists():
    raise FileNotFoundError(f"Missing {TRUTH_XLSX}")

truth = pd.read_excel(TRUTH_XLSX, sheet_name=0).dropna(axis=1, how="all")
print("Truth columns:", list(truth.columns))

sim_id_col = _find_col(truth, "sim id", "simid", "sim_id", "id")
lai_col = _find_col(truth, "lai")
wai_col = _find_col(truth, "wai")
clump_col = next((c for c in truth.columns if "clump" in str(c).strip().lower()), None)
pai_cols = _find_cols_containing(truth, "pai")  # expects ['PAI', 'PAI.1'] etc

if sim_id_col is None:
    raise ValueError("Could not find a 'Sim ID' column in the truth sheet.")

print(
    "Detected truth cols:",
    {"sim_id_col": sim_id_col, "lai_col": lai_col, "wai_col": wai_col, "pai_cols": pai_cols, "clump_col": clump_col},
)

# Clean Sim ID and drop junk/footer rows
truth["_sim_id_num"] = _coerce_sim_id(truth[sim_id_col])
before = len(truth)
truth = truth.dropna(subset=["_sim_id_num"]).copy()
truth["_sim_id_num"] = truth["_sim_id_num"].astype(int)
after = len(truth)
if after != before:
    print(f"Dropped {before - after} non-data rows (non-numeric Sim ID).")

expected_long = n_cases * max(n_oris, 1)

# Decide whether truth is WIDE (≈75 rows) or LONG (≈150/225 rows)
is_wide = len(truth) <= (n_cases + 2) and len(truth) >= max(1, n_cases - 5)
is_long = abs(len(truth) - expected_long) <= 2

print("Truth rows:", len(truth), "Expected long rows:", expected_long, "Detected format:", "WIDE" if is_wide else ("LONG" if is_long else "UNKNOWN"))

# Build a long-format truth table with columns: case_norm, orientation, truth_*
if is_wide:
    # One row per case; expand to one row per case×orientation.
    truth_base = truth.copy()
    truth_base["case_norm"] = truth_base["_sim_id_num"].map(lambda n: f"Case {int(n):03d}")
    truth_base["truth_sim_id"] = truth_base["_sim_id_num"]

    # Choose which PAI column maps to which orientation (try permutations if ambiguous)
    pai_cols = [c for c in pai_cols if c not in ("LAI:", "Leaf area index")]
    if len(hemi_oris) == 2 and len(pai_cols) >= 2:
        # score both assignments using correlation within each orientation
        assignments = [
            {hemi_oris[0]: pai_cols[0], hemi_oris[1]: pai_cols[1]},
            {hemi_oris[0]: pai_cols[1], hemi_oris[1]: pai_cols[0]},
        ]

        best = None
        for a in assignments:
            parts = []
            for ori in hemi_oris:
                t_part = truth_base[["case_norm", "truth_sim_id"]].copy()
                t_part["orientation"] = ori
                t_part["truth_PAI"] = pd.to_numeric(truth_base[a[ori]], errors="coerce")
                if lai_col is not None:
                    t_part["truth_LAI"] = pd.to_numeric(truth_base[lai_col], errors="coerce")
                if wai_col is not None:
                    t_part["truth_WAI"] = pd.to_numeric(truth_base[wai_col], errors="coerce")
                if clump_col is not None:
                    t_part["truth_Clumping"] = pd.to_numeric(truth_base[clump_col], errors="coerce")
                t_part["truth_PAI_sourcecol"] = a[ori]
                parts.append(t_part)
            t_long = pd.concat(parts, ignore_index=True)

            j = hemi.merge(t_long, on=["case_norm", "orientation"], how="left")
            missing = int(j["truth_PAI"].isna().sum())
            corr = _safe_pearson(j["PAI_Hinge_value"], j["truth_PAI"])
            score = (missing, -(abs(corr) if corr is not None else -1.0))
            if best is None or score < best[0]:
                best = (score, a, corr, missing)

        assert best is not None
        _, chosen_assignment, chosen_corr, chosen_missing = best
        print("Chosen wide PAI mapping:", chosen_assignment, "missing:", chosen_missing, "corr:", chosen_corr)

        truth_long_parts = []
        for ori in hemi_oris:
            t_part = truth_base[["case_norm", "truth_sim_id"]].copy()
            t_part["orientation"] = ori
            t_part["truth_PAI"] = pd.to_numeric(truth_base[chosen_assignment[ori]], errors="coerce")
            if lai_col is not None:
                t_part["truth_LAI"] = pd.to_numeric(truth_base[lai_col], errors="coerce")
            if wai_col is not None:
                t_part["truth_WAI"] = pd.to_numeric(truth_base[wai_col], errors="coerce")
            if clump_col is not None:
                t_part["truth_Clumping"] = pd.to_numeric(truth_base[clump_col], errors="coerce")
            t_part["truth_PAI_sourcecol"] = chosen_assignment[ori]
            truth_long_parts.append(t_part)

        truth_long = pd.concat(truth_long_parts, ignore_index=True)

    else:
        raise ValueError(
            f"Truth looks wide but cannot map PAI columns to orientations. hemi_oris={hemi_oris}, pai_cols={pai_cols}"
        )

elif is_long:
    # One row per case×orientation; infer mapping from Sim ID ordering
    candidates = []
    for mode in ["cycle", "block"]:
        for ori_order in itertools.permutations(hemi_oris):
            mapped = _map_sim_id_long(truth["_sim_id_num"], n_cases=n_cases, orientations=list(ori_order), mode=mode)
            t2 = pd.concat([truth.reset_index(drop=True), mapped], axis=1)

            # pick a "truth PAI" column for scoring (use first PAI-like column)
            pai_for_score = pai_cols[0] if pai_cols else None
            joined = hemi.merge(t2, on=["case_norm", "orientation"], how="left")

            missing = 0
            if pai_for_score is not None:
                missing = int(pd.to_numeric(joined[pai_for_score], errors="coerce").isna().sum())

            corr = None
            if pai_for_score is not None:
                corr = _safe_pearson(joined["PAI_Hinge_value"], pd.to_numeric(joined[pai_for_score], errors="coerce"))

            candidates.append((missing, -(abs(corr) if corr is not None else -1.0), mode, list(ori_order), corr))

    candidates.sort()
    missing, _, best_mode, best_order, best_corr = candidates[0]
    print("Chosen long SimID mapping:", {"mode": best_mode, "orientation_order": best_order, "missing": missing, "corr": best_corr})

    mapped = _map_sim_id_long(truth["_sim_id_num"], n_cases=n_cases, orientations=best_order, mode=best_mode)
    truth2 = pd.concat([truth.reset_index(drop=True), mapped], axis=1)

    truth_long = pd.DataFrame(
        {
            "case_norm": truth2["case_norm"],
            "orientation": truth2["orientation"],
            "truth_sim_id": truth2["_sim_id_num"],
        }
    )
    if lai_col is not None:
        truth_long["truth_LAI"] = pd.to_numeric(truth2[lai_col], errors="coerce")
    if wai_col is not None:
        truth_long["truth_WAI"] = pd.to_numeric(truth2[wai_col], errors="coerce")
    if clump_col is not None:
        truth_long["truth_Clumping"] = pd.to_numeric(truth2[clump_col], errors="coerce")

    # keep all PAI columns with distinct names
    for pc in pai_cols:
        truth_long[f"truth_{pc}"] = pd.to_numeric(truth2[pc], errors="coerce")

else:
    raise ValueError(
        f"Truth format unclear: truth rows={len(truth)}, expected long={expected_long}. "
        "If this is supposed to be 75 rows (wide), make sure the sheet contains only cases 1..75. "
        "If it's supposed to be long, it should be ~cases×orientations rows."
    )

# --- Join ---
joined = hemi.merge(truth_long, on=["case_norm", "orientation"], how="left")

print("Rows:", len(joined))
if "truth_LAI" in joined.columns:
    print("Missing truth_LAI:", int(joined["truth_LAI"].isna().sum()))
if "truth_PAI" in joined.columns:
    print("Missing truth_PAI:", int(joined["truth_PAI"].isna().sum()))

joined.to_csv(OUT_CSV, index=False, float_format="%.2f")
print("Wrote", OUT_CSV)