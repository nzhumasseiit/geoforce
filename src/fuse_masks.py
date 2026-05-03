from pathlib import Path
import argparse
import json

import cv2
import numpy as np
from PIL import Image


def load_mask(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def save_mask(mask: np.ndarray, path: Path):
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def ensure_record(records_by_class, cls: str, tile_name: str, metadata_path: str, output_dir: Path):
    if cls not in records_by_class:
        records_by_class[cls] = {
            "tile_name": tile_name,
            "class": cls,
            "mask_path": str(output_dir / f"{tile_name}_{cls}.png"),
            "metadata_path": metadata_path,
        }
    return records_by_class[cls]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rule_mask_index")
    parser.add_argument("yolo_mask_index")
    parser.add_argument("--output", default="outputs/masks_fused")
    parser.add_argument("--yolo-min-area", type=int, default=40)
    parser.add_argument("--roof-dilate", type=int, default=5)
    args = parser.parse_args()

    rule_records = json.load(open(args.rule_mask_index, "r", encoding="utf-8"))
    yolo_records = json.load(open(args.yolo_mask_index, "r", encoding="utf-8"))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_tile = {}

    for record in rule_records:
        tile = record["tile_name"]
        by_tile.setdefault(tile, {
            "metadata_path": record["metadata_path"],
            "rules": {},
            "yolo": {},
        })
        by_tile[tile]["rules"][record["class"]] = record

    for record in yolo_records:
        tile = record["tile_name"]
        by_tile.setdefault(tile, {
            "metadata_path": record["metadata_path"],
            "rules": {},
            "yolo": {},
        })
        by_tile[tile]["yolo"][record["class"]] = record

    fused_records = []
    kernel = np.ones((args.roof_dilate, args.roof_dilate), np.uint8)

    for tile_name, item in by_tile.items():
        records_by_class = {}
        metadata_path = item["metadata_path"]

        vegetation = None
        ground = None
        shadow = None
        rooftop_rule = None

        if "vegetation" in item["rules"]:
            vegetation = load_mask(item["rules"]["vegetation"]["mask_path"])
        if "ground" in item["rules"]:
            ground = load_mask(item["rules"]["ground"]["mask_path"])
        if "shadow_ignore" in item["rules"]:
            shadow = load_mask(item["rules"]["shadow_ignore"]["mask_path"])
        if "rooftop" in item["rules"]:
            rooftop_rule = load_mask(item["rules"]["rooftop"]["mask_path"])

        shape = None
        for mask in [vegetation, ground, shadow, rooftop_rule]:
            if mask is not None:
                shape = mask.shape
                break

        if shape is None and "rooftop" in item["yolo"]:
            shape = load_mask(item["yolo"]["rooftop"]["mask_path"]).shape

        if shape is None:
            continue

        if vegetation is None:
            vegetation = np.zeros(shape, dtype=bool)
        if ground is None:
            ground = np.zeros(shape, dtype=bool)
        if shadow is None:
            shadow = np.zeros(shape, dtype=bool)
        if rooftop_rule is None:
            rooftop_rule = np.zeros(shape, dtype=bool)

        rooftop_yolo = np.zeros(shape, dtype=bool)
        if "rooftop" in item["yolo"]:
            rooftop_yolo = load_mask(item["yolo"]["rooftop"]["mask_path"])
            rooftop_yolo = cv2.morphologyEx(
                rooftop_yolo.astype(np.uint8),
                cv2.MORPH_CLOSE,
                kernel,
            ) > 0

            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                rooftop_yolo.astype(np.uint8),
                connectivity=8,
            )
            clean = np.zeros_like(rooftop_yolo, dtype=bool)
            for idx in range(1, count):
                if stats[idx, cv2.CC_STAT_AREA] >= args.yolo_min_area:
                    clean[labels == idx] = True
            rooftop_yolo = clean

        # Let strong YOLO roof evidence override rule ground/shadow,
        # but keep vegetation where there is overwhelming evidence.
        rooftop = rooftop_rule | (rooftop_yolo & ~vegetation)
        ground = ground & ~rooftop
        shadow = shadow & ~rooftop

        class_masks = {
            "vegetation": vegetation,
            "ground": ground,
            "shadow_ignore": shadow,
            "rooftop": rooftop,
        }

        for cls, mask in class_masks.items():
            record = ensure_record(records_by_class, cls, tile_name, metadata_path, out_dir)
            out_path = Path(record["mask_path"])
            save_mask(mask, out_path)
            fused_records.append(record)

    index_path = out_dir / "mask_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(fused_records, f, indent=2)

    print(f"Saved fused masks: {len(fused_records)}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
