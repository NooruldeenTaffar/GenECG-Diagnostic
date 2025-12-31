import argparse
import sys
from pathlib import Path
from tqdm import tqdm

try:
    from src.Data_pipeline.yolo_labels import generate_yolo_labels
except ModuleNotFoundError:
    # Allow running this script directly by making src.Data_pipeline importable
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parents[1]
    for path in (project_root, current_dir):
        str_path = str(path)
        if str_path not in sys.path:
            sys.path.append(str_path)
    from src.Data_pipeline.yolo_labels import generate_yolo_labels

DEFAULT_RAW_DIR = Path("/Volumes/Noori SSD/Applications/GenECG-Diagnostic/data/Raw/GenECG/Dataset_B_ECGs_with_imperfections")
DEFAULT_OUTPUT_DIR = Path("data/Processed/YOLO_Labels")


def run_mass_labeling(raw_data_dir: Path, output_dir: Path, limit: int | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Targeting directory: {raw_data_dir}")
    if not raw_data_dir.exists():
        print("❌ ERROR: Path does not exist. Check your folder names!")
        return

    all_images = sorted(raw_data_dir.rglob("*.[pP][nN][gG]"))

    if limit is not None and limit > 0:
        process_list = all_images[:limit]
    else:
        process_list = all_images

    print(f"Found {len(all_images)} images. Processing {len(process_list)}...")

    success_count = 0
    for img_path in tqdm(process_list):
        try:
            generate_yolo_labels(str(img_path), str(output_dir))
            success_count += 1
        except Exception as e:
            print(f"Error on {img_path.name}: {e}")

    print(f"\n✅ Success! Generated {success_count} labels in {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate YOLO labels for GenECG datasets.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Root directory containing ECG PNGs (defaults to Dataset B).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where YOLO label .txt files will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process (processes all when omitted or <=0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_mass_labeling(args.raw_dir, args.output_dir, args.limit)
