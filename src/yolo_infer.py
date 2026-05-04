from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir")
    parser.add_argument("--weights", default="models/emergency.pt")
    parser.add_argument("--output", default="outputs/yolo")
    parser.add_argument("--conf", type=float, default=0.10)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.output)

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is required for the optional emergency YOLO module. "
            "Install dependencies from requirements.txt and add weights to models/emergency.pt."
        ) from exc

    model = YOLO(args.weights)

    results = model.predict(
        source=str(images_dir / "images"),  # IMPORTANT
        conf=args.conf,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(out_dir),
        name="streamlit_preds",  # avoid overwriting
        exist_ok=True
    )

    print(f"Saved YOLO predictions to {out_dir}/streamlit_preds")

if __name__ == "__main__":
    main()
