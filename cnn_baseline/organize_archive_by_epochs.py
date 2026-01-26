from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


EPOCH_RE = re.compile(r"(?:^|_)epochs(?P<epochs>\d+)(?:_|\.|$)")


@dataclass(frozen=True)
class EpochInfo:
    label: str  # e.g. "epochs12" or "epochs_unknown"
    epochs: int | None


def _epoch_from_name(path: Path) -> int | None:
    m = EPOCH_RE.search(path.name)
    if not m:
        return None
    try:
        return int(m.group("epochs"))
    except Exception:
        return None


def _infer_epochs_from_progress_csv(path: Path) -> int | None:
    """Infer the number of epochs from a progress CSV by taking max(epoch) over epoch_end rows."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if df.empty or "epoch" not in df.columns:
        return None

    if "event" in df.columns:
        df = df[df["event"].astype(str) == "epoch_end"].copy()

    if df.empty:
        return None

    s = pd.to_numeric(df["epoch"], errors="coerce").dropna()
    if s.empty:
        return None

    return int(s.max())


def _safe_move(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        # Avoid overwriting: add suffix.
        i = 1
        while True:
            candidate = dest_dir / f"{src.stem}__dup{i}{src.suffix}"
            if not candidate.exists():
                dest = candidate
                break
            i += 1
    shutil.move(str(src), str(dest))
    return dest


def organize_one_folder(folder: Path) -> dict:
    """Organize files directly inside `folder` into epochsXX subfolders."""
    moved = []

    files = [p for p in folder.iterdir() if p.is_file()]
    if not files:
        return {"folder": str(folder), "moved": moved}

    # Identify "untagged" progress CSVs (no epochsNN in name) to infer epochs.
    untagged_progress = [
        p
        for p in files
        if p.suffix.lower() == ".csv"
        and p.name.startswith("cnn_training_progress")
        and _epoch_from_name(p) is None
    ]

    inferred_epochs: int | None = None
    if len(untagged_progress) == 1:
        inferred_epochs = _infer_epochs_from_progress_csv(untagged_progress[0])

    for f in files:
        # Skip anything already organized (we only process files at this folder level).
        epochs_from_name = _epoch_from_name(f)
        if epochs_from_name is not None:
            epoch_info = EpochInfo(label=f"epochs{epochs_from_name}", epochs=epochs_from_name)
        else:
            # If there is exactly one untagged progress CSV, we treat untagged outputs
            # in that folder as belonging to the same run and infer epochs from it.
            if inferred_epochs is not None:
                epoch_info = EpochInfo(label=f"epochs{inferred_epochs}", epochs=inferred_epochs)
            else:
                epoch_info = EpochInfo(label="epochs_unknown", epochs=None)

        dest = _safe_move(f, folder / epoch_info.label)
        moved.append((str(f), str(dest)))

    return {"folder": str(folder), "moved": moved}


def main() -> int:
    p = argparse.ArgumentParser(description="Organize cnn_baseline/archive outputs into epochsXX subfolders.")
    p.add_argument(
        "--archive",
        type=Path,
        default=Path(__file__).resolve().parent / "archive",
        help="Path to cnn_baseline/archive",
    )
    args = p.parse_args()

    archive = args.archive
    if not archive.exists():
        raise SystemExit(f"Archive folder not found: {archive}")

    # Organize each immediate child folder (e.g., manual_YYYYmmdd_HHMMSS).
    targets = [p for p in archive.iterdir() if p.is_dir()]
    if not targets:
        print("No subfolders found under archive; nothing to organize.")
        return 0

    for folder in sorted(targets):
        result = organize_one_folder(folder)
        if result["moved"]:
            print(f"Organized: {folder}")
            print(f"  moved {len(result['moved'])} files")
        else:
            print(f"Skipped (no files): {folder}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
