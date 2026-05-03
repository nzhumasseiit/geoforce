from pathlib import Path
import argparse
import json

import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import slic
from skimage.color import rgb2lab


CLASS_ID = {
    "background": 0,
    "vegetation": 1,
    "ground": 2,
    "rooftop": 3,
    "shadow_ignore": 4,
}

ID_CLASS = {v: k for k, v in CLASS_ID.items()}


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path):
    return np.array(Image.open(path).convert("L")) > 0


def save_mask(mask, path):
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def majority_class(labels, weak_class_map, valid_mask):
    out = np.zeros_like(weak_class_map, dtype=np.uint8)

    for seg_id in np.unique(labels):
        region = labels == seg_id
        region_valid = region & valid_mask

        if region_valid.sum() < 20:
            continue

        values = weak_class_map[region_valid]
        values = values[values != CLASS_ID["background"]]

        if len(values) == 0:
            continue

        counts = np.bincount(values, minlength=len(CLASS_ID))

        # Vegetation boost: if enough pixels in a superpixel are vegetation,
        # keep it as vegetation even if paved/rooftop slightly wins.
        veg_ratio = counts[CLASS_ID["vegetation"]] / region_valid.sum()
        if veg_ratio >= 0.12:
            out[region_valid] = CLASS_ID["vegetation"]
            continue

        cls = counts.argmax()
        confidence = counts[cls] / region_valid.sum()

        threshold_by_class = {
            CLASS_ID["vegetation"]: 0.12,
            CLASS_ID["rooftop"]: 0.20,
            CLASS_ID["ground"]: 0.42,
            CLASS_ID["shadow_ignore"]: 0.45,
        }

        threshold = threshold_by_class.get(cls, 0.25)

        if confidence >= threshold:
            out[region_valid] = cls

    return out


def clean_class_mask(mask, min_area_px, cls):
    mask = mask.astype(np.uint8)

    if cls == "vegetation":
        # Tree crowns can be fragmented, so close gaps gently.
        close_k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

        # Light opening removes isolated noise without killing small trees.
        open_k = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    else:
        close_k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            clean[labels == i] = 1

    return clean.astype(bool)


def process_tile(tile_image_path, tile_meta_path, weak_masks_by_class, out_dir):
    rgb = load_rgb(tile_image_path)

    with open(tile_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    valid_mask = load_mask(meta["valid_mask_path"])
    weak_class_map = np.zeros(valid_mask.shape, dtype=np.uint8)

    # Priority order:
    # broad/uncertain first, stronger semantic classes later.
    # Vegetation last = vegetation can rescue trees from shadow/paved confusion.
    priority = ["shadow_ignore", "ground", "rooftop", "vegetation"]

    for cls in priority:
        if cls not in weak_masks_by_class:
            continue

        m = load_mask(weak_masks_by_class[cls])
        weak_class_map[m & valid_mask] = CLASS_ID[cls]

    # Shadow should not erase vegetation.
    if "shadow_ignore" in weak_masks_by_class and "vegetation" in weak_masks_by_class:
        shadow = load_mask(weak_masks_by_class["shadow_ignore"])
        vegetation = load_mask(weak_masks_by_class["vegetation"])
        weak_class_map[shadow & vegetation & valid_mask] = CLASS_ID["vegetation"]

    lab = rgb2lab(rgb)

    labels = slic(
        lab,
        n_segments=1200,
        compactness=10,
        sigma=1,
        start_label=1,
        channel_axis=-1,
    )

    object_class_map = majority_class(labels, weak_class_map, valid_mask)

    records = []

    min_area_by_class = {
        "vegetation": 35,
        "ground": 180,
        "rooftop": 100,
        "shadow_ignore": 250,
    }

    for cls, cls_id in CLASS_ID.items():
        if cls == "background":
            continue

        mask = object_class_map == cls_id
        mask = clean_class_mask(mask, min_area_by_class[cls], cls)

        out_path = out_dir / f"{meta['tile_name']}_{cls}.png"
        save_mask(mask, out_path)

        records.append({
            "tile_name": meta["tile_name"],
            "class": cls,
            "mask_path": str(out_path),
            "metadata_path": str(tile_meta_path),
        })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weak_mask_index")
    parser.add_argument("--output", default="outputs/masks_obia")
    args = parser.parse_args()

    weak_mask_index = Path(args.weak_mask_index)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = json.load(open(weak_mask_index, "r", encoding="utf-8"))

    by_tile = {}

    for r in records:
        tile = r["tile_name"]
        by_tile.setdefault(tile, {
            "metadata_path": r["metadata_path"],
            "masks": {},
        })
        by_tile[tile]["masks"][r["class"]] = r["mask_path"]

    all_out = []

    for tile, item in by_tile.items():
        meta = json.load(open(item["metadata_path"], "r", encoding="utf-8"))

        if "tile_image_path" in meta:
            image_path = meta["tile_image_path"]
        else:
            image_path = str(
                Path(item["metadata_path"])
                .parents[1] / "images" / f"{tile}.png"
            )

        all_out.extend(
            process_tile(
                image_path,
                item["metadata_path"],
                item["masks"],
                out_dir,
            )
        )

    index_path = out_dir / "mask_index.json"

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2)

    print(f"Processed tiles: {len(by_tile)}")
    print(f"Saved OBIA masks: {len(all_out)}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
