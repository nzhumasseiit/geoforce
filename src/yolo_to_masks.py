from pathlib import Path
import argparse
import json

import cv2
import numpy as np
from PIL import Image


def load_meta(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_label_line(line: str):
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    class_id = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:5])
    conf = float(parts[5]) if len(parts) >= 6 else None

    return {
        "class_id": class_id,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "conf": conf,
    }


def yolo_box_to_pixels(det, width: int, height: int):
    cx = det["cx"] * width
    cy = det["cy"] * height
    box_w = det["w"] * width
    box_h = det["h"] * height

    x1 = max(0, int(round(cx - box_w / 2.0)))
    y1 = max(0, int(round(cy - box_h / 2.0)))
    x2 = min(width, int(round(cx + box_w / 2.0)))
    y2 = min(height, int(round(cy + box_h / 2.0)))

    return x1, y1, x2, y2


def draw_detections_mask(label_path: Path, image_path: Path, min_conf: float, expand_px: int):
    image = Image.open(image_path)
    width, height = image.size
    mask = np.zeros((height, width), dtype=np.uint8)
    confidences = []

    if not label_path.exists():
        return mask.astype(bool), confidences

    with open(label_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            det = parse_label_line(raw_line)
            if det is None:
                continue

            conf = det["conf"] if det["conf"] is not None else 1.0
            if conf < min_conf:
                continue

            x1, y1, x2, y2 = yolo_box_to_pixels(det, width, height)
            x1 = max(0, x1 - expand_px)
            y1 = max(0, y1 - expand_px)
            x2 = min(width, x2 + expand_px)
            y2 = min(height, y2 + expand_px)

            if x2 <= x1 or y2 <= y1:
                continue

            mask[y1:y2, x1:x2] = 255
            confidences.append(conf)

    return mask > 0, confidences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tile_root", help="Tile folder containing images/metadata/valid_masks")
    parser.add_argument("yolo_run_dir", help="YOLO output dir, e.g. outputs/yolo/streamlit_preds")
    parser.add_argument("--output", default="outputs/yolo_masks", help="Output mask folder")
    parser.add_argument("--class-name", default="rooftop")
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--expand-px", type=int, default=8)
    args = parser.parse_args()

    tile_root = Path(args.tile_root)
    yolo_run_dir = Path(args.yolo_run_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = yolo_run_dir / "labels"
    image_dir = tile_root / "images"
    meta_dir = tile_root / "metadata"

    records = []

    for image_path in sorted(image_dir.glob("*.png")):
        tile_name = image_path.stem
        meta_path = meta_dir / f"{tile_name}.json"
        label_path = labels_dir / f"{tile_name}.txt"

        if not meta_path.exists():
            continue

        mask, confidences = draw_detections_mask(
            label_path=label_path,
            image_path=image_path,
            min_conf=args.min_conf,
            expand_px=args.expand_px,
        )

        out_path = out_dir / f"{tile_name}_{args.class_name}.png"
        Image.fromarray((mask.astype(np.uint8) * 255)).save(out_path)

        records.append({
            "tile_name": tile_name,
            "class": args.class_name,
            "mask_path": str(out_path),
            "metadata_path": str(meta_path),
            "source_method": "yolo",
            "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
            "detections": len(confidences),
        })

    index_path = out_dir / "mask_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Saved YOLO-derived masks: {len(records)}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
