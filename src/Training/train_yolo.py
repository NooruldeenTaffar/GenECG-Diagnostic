"""Custom YOLOv8 training entrypoint that wires GenECG labels from the processed folder."""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
    from ultralytics.data import utils as data_utils
except ModuleNotFoundError as exc:  # pragma: no cover - checked at runtime
    raise RuntimeError(
        "Ultralytics is required. Install it with `pip install ultralytics` before running this script."
    ) from exc

DEFAULT_DATA = Path("data_A.yaml")
DEFAULT_LABEL_ROOT = Path("data/Processed/YOLO_Labels/Dataset_A")


def override_label_resolution(labels_root: Path):
    """Patch YOLO's label discovery so every image maps to a label under labels_root."""
    original_func = data_utils.img2label_paths

    def _custom_img2label_paths(img_paths):
        resolved = []
        for img in img_paths:
            img_path = Path(img)
            label_path = labels_root / (img_path.stem + ".txt")
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {img_path} at {label_path}")
            resolved.append(str(label_path))
        return resolved

    data_utils.img2label_paths = _custom_img2label_paths
    return original_func


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on GenECG with processed labels.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset YAML (defaults to data_A.yaml).")
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=DEFAULT_LABEL_ROOT,
        help="Directory containing YOLO-formatted label .txt files.",
    )
    parser.add_argument("--weights", default="yolov8n.pt", help="Base model weights to fine-tune (default: yolov8n.pt).")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50).")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size used for training (default: 640).")
    parser.add_argument("--device", default=None, help="Torch device spec, e.g., 0 for GPU or 'cpu'.")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = args.data.resolve()
    labels_root = args.labels_root.resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_root}")

    original_resolver = override_label_resolution(labels_root)

    try:
        model = YOLO(args.weights)
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
        )
    finally:
        data_utils.img2label_paths = original_resolver


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"Training failed: {err}")
        sys.exit(1)
