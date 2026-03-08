import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from harris.harris import harris_response
from harris.nms import nms


def count_manual(img: np.ndarray, k: float, threshold_ratio: float) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    response = harris_response(gray, k=k)
    corners = nms(response, threshold_ratio=threshold_ratio)
    return int(corners.sum())


def count_opencv_threshold_only(
    img: np.ndarray, k: float, block_size: int, sobel_ksize: int, threshold_ratio: float
) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    response = cv2.cornerHarris(gray, blockSize=block_size, ksize=sobel_ksize, k=k)
    response = cv2.dilate(response, None)
    corners = response > (threshold_ratio * response.max())
    return int(corners.sum())


def count_opencv_with_nms(
    img: np.ndarray, k: float, block_size: int, sobel_ksize: int, threshold_ratio: float
) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    response = cv2.cornerHarris(gray, blockSize=block_size, ksize=sobel_ksize, k=k)
    corners = nms(response, threshold_ratio=threshold_ratio)
    return int(corners.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="统计不同图像的 Harris 角点数量")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-csv", type=Path, default=Path("data/output/stats/corner_counts.csv"))
    parser.add_argument("--k", type=float, default=0.04)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--sobel-ksize", type=int, default=3)
    args = parser.parse_args()

    image_paths = sorted([p for p in args.data_dir.glob("*") if p.is_file()])
    if not image_paths:
        raise FileNotFoundError(f"no images found in {args.data_dir}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"skip unreadable image -> {image_path}")
            continue

        manual_count = count_manual(img, k=args.k, threshold_ratio=args.threshold)
        opencv_count = count_opencv_threshold_only(
            img,
            k=args.k,
            block_size=args.block_size,
            sobel_ksize=args.sobel_ksize,
            threshold_ratio=args.threshold,
        )
        opencv_nms_count = count_opencv_with_nms(
            img,
            k=args.k,
            block_size=args.block_size,
            sobel_ksize=args.sobel_ksize,
            threshold_ratio=args.threshold,
        )

        rows.append(
            {
                "image": image_path.name,
                "manual_nms": manual_count,
                "opencv_threshold_only": opencv_count,
                "opencv_nms": opencv_nms_count,
            }
        )

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "manual_nms", "opencv_threshold_only", "opencv_nms"]
        )
        writer.writeheader()
        writer.writerows(rows)

    manual_vals = np.array([r["manual_nms"] for r in rows], dtype=np.float64)
    opencv_vals = np.array([r["opencv_threshold_only"] for r in rows], dtype=np.float64)
    opencv_nms_vals = np.array([r["opencv_nms"] for r in rows], dtype=np.float64)

    print(f"saved -> {args.out_csv}")
    print(
        "manual_nms: mean={:.2f}, min={}, max={}".format(
            manual_vals.mean(), int(manual_vals.min()), int(manual_vals.max())
        )
    )
    print(
        "opencv_threshold_only: mean={:.2f}, min={}, max={}".format(
            opencv_vals.mean(), int(opencv_vals.min()), int(opencv_vals.max())
        )
    )
    print(
        "opencv_nms: mean={:.2f}, min={}, max={}".format(
            opencv_nms_vals.mean(), int(opencv_nms_vals.min()), int(opencv_nms_vals.max())
        )
    )


if __name__ == "__main__":
    main()
