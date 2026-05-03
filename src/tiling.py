from pathlib import Path
import argparse
import json
import csv

import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform
from PIL import Image


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    rgb input: shape (3, H, W)
    output: shape (H, W, 3), uint8
    """
    rgb = np.transpose(rgb, (1, 2, 0))

    if rgb.dtype == np.uint8:
        return rgb

    rgb = rgb.astype(np.float32)
    lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def generate_windows(width: int, height: int, tile_size: int, overlap: int):
    step = tile_size - overlap

    if step <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    def positions(length: int):
        if length <= tile_size:
            return [0]

        starts = list(range(0, length - tile_size + 1, step))
        last_start = length - tile_size

        if starts[-1] != last_start:
            starts.append(last_start)

        return starts

    for y in positions(height):
        tile_h = min(tile_size, height - y)

        for x in positions(width):
            tile_w = min(tile_size, width - x)
            yield Window(
                col_off=x,
                row_off=y,
                width=tile_w,
                height=tile_h,
            )


def tile_geotiff(
    input_path: str,
    output_dir: str,
    tile_size: int = 1024,
    overlap: int = 128,
    min_valid_ratio: float = 0.05,
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    image_dir = output_dir / "images"
    meta_dir = output_dir / "metadata"
    mask_dir = output_dir / "valid_masks"

    image_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    records = []

    with rasterio.open(input_path) as src:
        if src.count < 3:
            raise ValueError("GeoTIFF must have at least RGB bands")

        has_alpha = src.count >= 4
        source_name = input_path.stem

        print("=" * 60)
        print(f"Source: {input_path}")
        print(f"Size: {src.width} x {src.height}")
        print(f"Bands: {src.count}")
        print(f"CRS: {src.crs}")
        print(f"Tile size: {tile_size}")
        print(f"Overlap: {overlap}")
        print(f"Has alpha mask: {has_alpha}")
        print("=" * 60)

        tile_id = 0
        skipped = 0

        for window in generate_windows(src.width, src.height, tile_size, overlap):
            rgb = src.read([1, 2, 3], window=window)
            rgb_img = normalize_rgb(rgb)

            if has_alpha:
                alpha = src.read(4, window=window)
                valid_mask = alpha > 0
                valid_ratio = float(np.count_nonzero(valid_mask) / valid_mask.size)

                if valid_ratio < min_valid_ratio:
                    skipped += 1
                    continue
            else:
                valid_mask = np.ones((int(window.height), int(window.width)), dtype=bool)
                valid_ratio = 1.0

            tile_name = f"{source_name}_tile_{tile_id:05d}"
            image_path = image_dir / f"{tile_name}.png"
            mask_path = mask_dir / f"{tile_name}_valid.png"
            meta_path = meta_dir / f"{tile_name}.json"

            Image.fromarray(rgb_img).save(image_path)
            Image.fromarray((valid_mask.astype(np.uint8) * 255)).save(mask_path)

            t_transform = window_transform(window, src.transform)

            meta = {
                "tile_id": tile_id,
                "tile_name": tile_name,
                "source": input_path.name,
                "source_path": str(input_path),
                "image_path": str(image_path),
                "valid_mask_path": str(mask_path),
                "crs": src.crs.to_string() if src.crs else None,
                "window": {
                    "col_off": int(window.col_off),
                    "row_off": int(window.row_off),
                    "width": int(window.width),
                    "height": int(window.height),
                },
                "transform": list(t_transform)[:6],
                "valid_ratio": valid_ratio,
                "tile_size": tile_size,
                "overlap": overlap,
                "bounds": rasterio.windows.bounds(window, src.transform),
            }

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            records.append(
                {
                    "tile_id": tile_id,
                    "tile_name": tile_name,
                    "image_path": str(image_path),
                    "metadata_path": str(meta_path),
                    "valid_mask_path": str(mask_path),
                    "source": input_path.name,
                    "crs": src.crs.to_string() if src.crs else None,
                    "col_off": int(window.col_off),
                    "row_off": int(window.row_off),
                    "width": int(window.width),
                    "height": int(window.height),
                    "valid_ratio": round(valid_ratio, 5),
                }
            )

            tile_id += 1

    index_path = output_dir / "tile_index.csv"

    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys() if records else [])
        writer.writeheader()
        writer.writerows(records)

    print(f"Created tiles: {len(records)}")
    print(f"Skipped empty/invalid tiles: {skipped}")
    print(f"Tile index: {index_path}")

    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to GeoTIFF")
    parser.add_argument("--output", default="data/tiles", help="Output tile directory")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--min-valid-ratio", type=float, default=0.05)

    args = parser.parse_args()

    tile_geotiff(
        input_path=args.input,
        output_dir=args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_valid_ratio=args.min_valid_ratio,
    )


if __name__ == "__main__":
    main()
