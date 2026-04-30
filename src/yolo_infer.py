from pathlib import Path
import argparse
import json
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir")
    parser.add_argument("--weights", default="models/roof_yolo_best.pt")
    parser.add_argument("--output", default="outputs/yolo")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    results = model.predict(
        source=str(images_dir),
        conf=args.conf,
        save=True,
        save_txt=True,
        project=str(out_dir),
        name="preds",
        exist_ok=True
    )

    print(f"Saved YOLO predictions to {out_dir}/preds")

if __name__ == "__main__":
    main()