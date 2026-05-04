from pathlib import Path
import argparse
import json

import cv2
import numpy as np
from PIL import Image


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path):
    return np.array(Image.open(path).convert("L")) > 0



def make_masks(rgb, valid_mask):
    rgb_f = rgb.astype(np.float32)
    r, g, b = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    brightness = (r + g + b) / 3.0
    exg = 2 * g - r - b

    # Shadow is QA only. Do not let it delete vegetation too early.
    shadow = (
        (brightness < 42) &
        valid_mask
    )

    green_rgb = (
        (g > r * 1.02) &
        (g > b * 1.02) &
        (exg > 4) &
        (s > 18)
    )

    green_hsv = (
        (h >= 32) & (h <= 100) &
        (s > 22) &
        (v > 35)
    )

    dark_green = (
        (h >= 35) & (h <= 95) &
        (s > 28) &
        (v > 22) &
        (brightness > 25)
    )

    vegetation_raw = green_rgb | green_hsv | dark_green

    vegetation = (
        vegetation_raw &
        valid_mask
    )

    grayness = (
        (np.abs(r - g) < 28) &
        (np.abs(g - b) < 28) &
        (np.abs(r - b) < 28)
    )

    reddish_roof = (
        (r > g * 1.12) &
        (r > b * 1.10) &
        (brightness > 55) &
        (brightness < 210) &
        (s > 22)
    )

    blue_or_colored_roof = (
        (s > 55) &
        (brightness > 70) &
        (brightness < 230) &
        ~((h >= 35) & (h <= 100))
    )


    gray_roof = (
        grayness &
        (brightness > 165) &
        (brightness < 225) &
        (s < 18)
    )

    dark_gray_roof = (
        grayness &
        (brightness > 125) &
        (brightness < 155) &
        (s < 15) &
        valid_mask &
        (~vegetation) &
        (~shadow)
    )

    rooftop = (
        (reddish_roof | blue_or_colored_roof | gray_roof | dark_gray_roof) &
        valid_mask &
        (~shadow) &
        (~vegetation)
    )

    paved_like = (
        grayness &
        (brightness > 55) &
        (brightness < 215) &
        (s < 45) &
        valid_mask &
        (~shadow) &
        (~vegetation) &
        (~rooftop)
    )

    bare_soil = (
        ((h >= 8) & (h <= 28)) &
        (s > 35) &
        (s < 170) &
        (brightness > 70) &
        (brightness < 205) &
        (r > g) &
        (g >= b * 0.9) &
        valid_mask &
        (~vegetation) &
        (~shadow) &
        (~rooftop)
    )

    # Keep the older strong roof heuristics, but merge them into the modern
    # stable class name. This preserves bright/gray roof extraction while
    # still exporting one robust built-up surface class.
    impervious = (paved_like | rooftop) & (~bare_soil)

    # Shadow only where we did not already find vegetation or emergency classes.
    shadow = shadow & (~vegetation)

    return {
        "vegetation": clean_mask(vegetation, 12, "vegetation"),
        "impervious_surface": clean_mask(impervious, 120, "impervious_surface"),
        "bare_soil": clean_mask(bare_soil, 90, "bare_soil"),
        "shadow_ignore": clean_mask(shadow, 200, "shadow_ignore"),
    }


def clean_mask(mask, min_area, class_name):
    mask = mask.astype(np.uint8)

    if class_name == "vegetation":
        # Gentle cleaning: keep small tree crowns.
        close_k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

        open_k = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    else:
        open_k = np.ones((5, 5), np.uint8)
        close_k = np.ones((9, 9), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    clean = np.zeros_like(mask)

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 1

    return clean.astype(bool)


def save_masks(tile_image_path, tile_meta_path, output_dir):
    tile_image_path = Path(tile_image_path)
    tile_meta_path = Path(tile_meta_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(tile_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    rgb = load_rgb(tile_image_path)
    valid_mask = load_mask(meta["valid_mask_path"])

    masks = make_masks(rgb, valid_mask)

    saved = []

    for class_name, mask in masks.items():
        out_path = output_dir / f"{meta['tile_name']}_{class_name}.png"
        Image.fromarray((mask.astype(np.uint8) * 255)).save(out_path)

        saved.append({
            "tile_name": meta["tile_name"],
            "class": class_name,
            "mask_path": str(out_path),
            "metadata_path": str(tile_meta_path),
        })

    return saved


def process_tile_folder(tile_root, output_dir):
    tile_root = Path(tile_root)
    output_dir = Path(output_dir)

    image_dir = tile_root / "images"
    meta_dir = tile_root / "metadata"

    all_records = []

    for image_path in sorted(image_dir.glob("*.png")):
        meta_path = meta_dir / f"{image_path.stem}.json"

        if not meta_path.exists():
            print(f"Missing metadata for {image_path.name}")
            continue

        records = save_masks(image_path, meta_path, output_dir)
        all_records.extend(records)

    index_path = output_dir / "mask_index.json"

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

    print(f"Processed tiles: {len(list(image_dir.glob('*.png')))}")
    print(f"Created masks: {len(all_records)}")
    print(f"Mask index: {index_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tile_root", help="Tile folder containing images/metadata/valid_masks")
    parser.add_argument("--output", default="outputs/masks", help="Output mask folder")
    args = parser.parse_args()

    process_tile_folder(args.tile_root, args.output)


if __name__ == "__main__":
    main()
