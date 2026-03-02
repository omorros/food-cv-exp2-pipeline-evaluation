"""
Generate degraded image variants for robustness testing.

Applies three degradations to the clean test images:
  D1: Gaussian blur    (kernel=15, sigma=6.0)  — out-of-focus camera
  D2: Gaussian noise   (mean=0, sigma=50)      — low-light sensor noise
  D3: JPEG compression (quality=5)             — messaging app compression

Usage:
    python -m evaluation.generate_degradations \
        --src dataset_exp2/images \
        --dst dataset_exp2
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from config import set_reproducibility_seed


def apply_blur(image: np.ndarray) -> np.ndarray:
    """D1: Gaussian blur (kernel=15, sigma=6.0)."""
    return cv2.GaussianBlur(image, (15, 15), 6.0)


def apply_noise(image: np.ndarray) -> np.ndarray:
    """D2: Gaussian noise (mean=0, sigma=50)."""
    noise = np.random.normal(0, 50, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def apply_jpeg(image: np.ndarray, quality: int = 5) -> np.ndarray:
    """D3: JPEG compression (quality=5)."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode(".jpg", image, encode_params)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEGRADATIONS = {
    "images_d1_blur":  ("D1: Gaussian blur (kernel=15, sigma=6.0)", apply_blur),
    "images_d2_noise": ("D2: Gaussian noise (sigma=50)", apply_noise),
    "images_d3_jpeg":  ("D3: JPEG compression (quality=5)", apply_jpeg),
}


def generate_degradations(src_dir: str, dst_root: str):
    """
    Generate all degraded variants.

    Args:
        src_dir: Directory with clean images.
        dst_root: Root directory for output (degraded folders created here).
    """
    set_reproducibility_seed()

    src = Path(src_dir)
    dst = Path(dst_root)
    images = sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)

    print(f"Source: {src} ({len(images)} images)")
    print()

    for folder_name, (description, fn) in DEGRADATIONS.items():
        out_dir = dst / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"{description}")
        print(f"  Output: {out_dir}")

        for i, img_path in enumerate(images):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  WARNING: Could not read {img_path.name}, skipping")
                continue

            degraded = fn(image)

            out_path = out_dir / (img_path.stem + ".jpg")
            cv2.imwrite(str(out_path), degraded)

        count = len(list(out_dir.glob("*.jpg")))
        print(f"  Generated: {count} images")
        print()

    print("Done. Labels are shared — no changes needed for labels/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate degraded image variants for robustness testing"
    )
    parser.add_argument("--src", default="dataset_exp2/images",
                        help="Source clean images directory")
    parser.add_argument("--dst", default="dataset_exp2",
                        help="Destination root (degraded folders created here)")
    args = parser.parse_args()

    generate_degradations(args.src, args.dst)
