from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class CropSpec:
    center_x: float
    center_y: float
    radius: float

    @property
    def size(self) -> int:
        return int(round(2 * self.radius))


def _iter_images(root: Path, *, recursive: bool) -> list[Path]:
    it = root.rglob("*") if recursive else root.glob("*")
    out: list[Path] = []
    for p in sorted(it):
        if not p.is_file():
            continue
        if p.name.lower() == "thumbs.db":
            continue
        if p.suffix.lower() not in _IMAGE_EXTS:
            continue
        out.append(p)
    return out


def _crop_to_square(img: Image.Image, spec: CropSpec) -> Image.Image:
    cx = float(spec.center_x)
    cy = float(spec.center_y)
    r = float(spec.radius)
    size = spec.size
    if size <= 0:
        raise ValueError(f"Invalid radius: {spec.radius}")

    left = int(round(cx - r))
    top = int(round(cy - r))
    right = left + size
    bottom = top + size

    w, h = img.size
    crop_left = max(0, left)
    crop_top = max(0, top)
    crop_right = min(w, right)
    crop_bottom = min(h, bottom)

    region = img.crop((crop_left, crop_top, crop_right, crop_bottom))
    out = Image.new("RGB", (size, size), (0, 0, 0))
    paste_x = max(0, -left)
    paste_y = max(0, -top)
    out.paste(region, (paste_x, paste_y))
    return out


def _mask_outside_circle(img: Image.Image, radius: float) -> Image.Image:
    size = img.size[0]
    if img.size[0] != img.size[1]:
        raise ValueError(f"Expected square image, got: {img.size}")

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    # The crop is defined so that the circle center is always at (r, r) in output coords.
    # Using bbox (0, 0, size, size) gives a circle that fills the square.
    draw.ellipse((0, 0, size, size), fill=255)

    black = Image.new("RGB", (size, size), (0, 0, 0))
    return Image.composite(img, black, mask)


def _suffix_for_format(fmt: str, original_suffix: str) -> str:
    fmt = fmt.lower()
    if fmt == "keep":
        return original_suffix
    if fmt in {"jpg", "jpeg"}:
        return ".jpg"
    if fmt in {"tif", "tiff"}:
        return ".tif"
    if fmt == "png":
        return ".png"
    raise ValueError(f"Unsupported format: {fmt}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Crop fisheye images in a folder to a square around a given center/radius, "
            "optionally masking outside the circle to black (recommended for the CNN)."
        )
    )

    p.add_argument("--in-dir", required=True, help="Input folder containing images (optionally nested)")
    p.add_argument("--out-dir", required=True, help="Output folder to write cropped images")

    # Defaults from Thompson / 1930 camera note.
    p.add_argument("--center-x", type=float, default=1305.0)
    p.add_argument("--center-y", type=float, default=1006.0)
    p.add_argument("--radius", type=float, default=708.0)

    p.add_argument("--no-mask", action="store_true", help="Do not mask outside-circle pixels to black")
    p.add_argument("--non-recursive", action="store_true", help="Only process direct children of in-dir")

    p.add_argument(
        "--format",
        default="png",
        choices=["png", "jpg", "tif", "keep"],
        help="Output image format (default: png). 'keep' preserves original extension.",
    )
    p.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality when --format jpg (default: 95)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output files if they already exist")
    p.add_argument("--dry-run", action="store_true", help="List what would be done without writing files")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not in_dir.exists():
        raise FileNotFoundError(f"in-dir not found: {in_dir}")
    if not in_dir.is_dir():
        raise NotADirectoryError(f"in-dir is not a directory: {in_dir}")

    recursive = not bool(args.non_recursive)
    spec = CropSpec(center_x=args.center_x, center_y=args.center_y, radius=args.radius)

    files = _iter_images(in_dir, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No images found under {in_dir} (looked for: {sorted(_IMAGE_EXTS)})")

    print(f"Discovered {len(files)} image(s) under: {in_dir}")
    print(f"Crop center=({spec.center_x:.1f}, {spec.center_y:.1f}) radius={spec.radius:.1f} => out_size={spec.size}x{spec.size}")
    print(f"Mask outside circle: {'no' if args.no_mask else 'yes'}")

    n_written = 0
    for i, src in enumerate(files, start=1):
        rel = src.relative_to(in_dir)
        out_suffix = _suffix_for_format(args.format, src.suffix.lower())
        dst = (out_dir / rel).with_suffix(out_suffix)

        if dst.exists() and not args.overwrite:
            continue

        if args.dry_run:
            print(f"[{i:>5}/{len(files)}] {src} -> {dst}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(src).convert("RGB")
        cropped = _crop_to_square(img, spec)
        if not args.no_mask:
            cropped = _mask_outside_circle(cropped, spec.radius)

        save_kwargs = {}
        if args.format == "jpg":
            save_kwargs = {"quality": int(args.jpeg_quality), "subsampling": 0}

        cropped.save(dst, **save_kwargs)
        n_written += 1

        if n_written % 50 == 0:
            print(f"Wrote {n_written} images...")

    if args.dry_run:
        print("Dry-run complete (no files written).")
    else:
        print(f"Done. Wrote {n_written} cropped image(s) to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
