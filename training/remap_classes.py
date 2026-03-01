"""
Remap a 47-class fruit/vegetable dataset to the project's 14-class system.

Reads YOLO label files from the source dataset, keeps only bounding boxes
belonging to the 14 target classes, remaps their class IDs, and writes
the filtered labels + copies matching images to the destination.

For "bell pepper", a colour-based HSV split determines whether each box
is bell_pepper_green (class 2) or bell_pepper_red (class 3).

Usage:
    python -m training.remap_classes \
        --src "C:/Users/Oriol/OneDrive/Escritorio/Combined Vegetables - Fruits.v8-2025-06-05-12-15am.yolov8" \
        --dst dataset
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

# ── Source dataset class IDs (from data.yaml, 47 classes) ──────────────
# Only the 13 source classes that map to our 14 targets.
# "bell pepper" (source 7) is handled specially → green or red.
SOURCE_TO_TARGET = {
    1:  0,   # apple       → apple
    4:  1,   # banana      → banana
    # 7: bell pepper → handled by colour split (2 or 3)
    13: 4,   # carrot      → carrot
    18: 5,   # cucumber    → cucumber
    22: 6,   # grape       → grape
    27: 7,   # lemon       → lemon
    32: 8,   # onion       → onion
    33: 9,   # orange      → orange
    36: 10,  # peach       → peach
    39: 11,  # potato      → potato
    43: 12,  # strawberry  → strawberry
    44: 13,  # tomato      → tomato
}

BELL_PEPPER_SOURCE_ID = 7
BELL_PEPPER_GREEN_TARGET = 2
BELL_PEPPER_RED_TARGET = 3

# IDs we care about (for quick filtering)
KEEP_IDS = set(SOURCE_TO_TARGET.keys()) | {BELL_PEPPER_SOURCE_ID}


def _is_red_pepper(image: np.ndarray, bbox: tuple[float, float, float, float]) -> bool:
    """
    Decide if a bell pepper crop is red or green using HSV colour analysis.

    Args:
        image: Full image (BGR, as loaded by cv2).
        bbox: (x_center, y_center, width, height) in normalised [0,1] coords.

    Returns:
        True if red, False if green.
    """
    h, w = image.shape[:2]
    xc, yc, bw, bh = bbox
    x1 = max(0, int((xc - bw / 2) * w))
    y1 = max(0, int((yc - bh / 2) * h))
    x2 = min(w, int((xc + bw / 2) * w))
    y2 = min(h, int((yc + bh / 2) * h))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Red hue wraps around 0/180 in OpenCV HSV
    mask_low = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask_high = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    red_pixels = cv2.countNonZero(mask_low) + cv2.countNonZero(mask_high)
    total_pixels = crop.shape[0] * crop.shape[1]

    return (red_pixels / total_pixels) > 0.40


def remap_split(src_root: Path, dst_root: Path, split: str, stats: dict):
    """Remap one split (train / valid / test)."""
    src_images = src_root / split / "images"
    src_labels = src_root / split / "labels"
    # Normalise "valid" → "val" for Ultralytics convention
    dst_split = "val" if split == "valid" else split
    dst_images = dst_root / dst_split / "images"
    dst_labels = dst_root / dst_split / "labels"

    if not src_labels.exists():
        print(f"  Skipping {split}: {src_labels} not found")
        return

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    label_files = sorted(src_labels.glob("*.txt"))
    kept = 0
    skipped = 0

    for lbl_file in label_files:
        lines = lbl_file.read_text().strip().splitlines()
        if not lines:
            continue

        # Find the matching image
        img_file = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            candidate = src_images / (lbl_file.stem + ext)
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            skipped += 1
            continue

        # Load image lazily (only if we have bell peppers)
        image = None
        new_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            src_id = int(parts[0])
            if src_id not in KEEP_IDS:
                continue

            coords = [float(v) for v in parts[1:5]]

            if src_id == BELL_PEPPER_SOURCE_ID:
                if image is None:
                    image = cv2.imread(str(img_file))
                    if image is None:
                        break
                if _is_red_pepper(image, tuple(coords)):
                    target_id = BELL_PEPPER_RED_TARGET
                    stats["bell_pepper_red"] = stats.get("bell_pepper_red", 0) + 1
                else:
                    target_id = BELL_PEPPER_GREEN_TARGET
                    stats["bell_pepper_green"] = stats.get("bell_pepper_green", 0) + 1
            else:
                target_id = SOURCE_TO_TARGET[src_id]

            new_lines.append(f"{target_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}")
            class_name = [
                "apple", "banana", "bell_pepper_green", "bell_pepper_red",
                "carrot", "cucumber", "grape", "lemon",
                "onion", "orange", "peach", "potato",
                "strawberry", "tomato",
            ][target_id]
            stats[class_name] = stats.get(class_name, 0) + 1

        if new_lines:
            # Write filtered label
            (dst_labels / lbl_file.name).write_text("\n".join(new_lines) + "\n")
            # Copy image
            shutil.copy2(img_file, dst_images / img_file.name)
            kept += 1
        else:
            skipped += 1

    print(f"  {split}: kept {kept} images, skipped {skipped} (no target classes)")


def main():
    parser = argparse.ArgumentParser(
        description="Remap 47-class dataset to 14-class system"
    )
    parser.add_argument(
        "--src", required=True,
        help="Source dataset root (with train/valid/test subdirs)"
    )
    parser.add_argument(
        "--dst", default="dataset",
        help="Destination 14-class dataset root (default: dataset)"
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    print(f"Source:      {src}")
    print(f"Destination: {dst}")
    print()

    stats: dict[str, int] = {}

    for split in ("train", "valid", "test"):
        print(f"Processing {split}...")
        remap_split(src, dst, split, stats)

    print("\n" + "=" * 50)
    print("CLASS DISTRIBUTION (bounding boxes)")
    print("=" * 50)
    for class_name in sorted(stats.keys()):
        print(f"  {class_name:<25} {stats[class_name]:>6}")
    print(f"  {'TOTAL':<25} {sum(stats.values()):>6}")


if __name__ == "__main__":
    main()
