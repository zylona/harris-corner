from pathlib import Path

import cv2
import numpy as np

from harris.harris import harris_response
from harris.nms import nms


DATA_DIR = Path("data/raw")
MANUAL_OUT_DIR = Path("data/output/manual")
OPENCV_OUT_DIR = Path("data/output/opencv")
MERGED_OUT_DIR = Path("data/output/compare")


def detect_manual(img: np.ndarray, k: float = 0.04, threshold_ratio: float = 0.01) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    response = harris_response(gray, k=k)
    corners = nms(response, threshold_ratio=threshold_ratio)

    result = img.copy()
    ys, xs = np.where(corners)
    for x, y in zip(xs, ys):
        cv2.circle(result, (x, y), 3, (0, 0, 255), 1)
    return result


def detect_opencv(
    img: np.ndarray,
    k: float = 0.04,
    block_size: int = 2,
    sobel_ksize: int = 3,
    threshold_ratio: float = 0.01,
) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    response = cv2.cornerHarris(gray, blockSize=block_size, ksize=sobel_ksize, k=k)
    response = cv2.dilate(response, None)
    corners = response > (threshold_ratio * response.max())

    result = img.copy()
    ys, xs = np.where(corners)
    for x, y in zip(xs, ys):
        cv2.circle(result, (x, y), 3, (0, 0, 255), 1)
    return result


def label_image(img: np.ndarray, label: str) -> np.ndarray:
    canvas = img.copy()
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 20), 2, cv2.LINE_AA)
    return canvas


def run_one(image_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"skip unreadable image -> {image_path}")
        return

    manual = detect_manual(img)
    opencv = detect_opencv(img)

    manual_path = MANUAL_OUT_DIR / image_path.name
    opencv_path = OPENCV_OUT_DIR / image_path.name
    merged_path = MERGED_OUT_DIR / image_path.name

    cv2.imwrite(str(manual_path), manual)
    cv2.imwrite(str(opencv_path), opencv)

    merged = np.hstack(
        [
            label_image(img, "original"),
            label_image(manual, "manual"),
            label_image(opencv, "opencv"),
        ]
    )
    cv2.imwrite(str(merged_path), merged)
    print(f"saved -> {manual_path}")
    print(f"saved -> {opencv_path}")
    print(f"saved -> {merged_path}")


def main() -> None:
    MANUAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    OPENCV_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(DATA_DIR.glob("*")):
        if image_path.is_file():
            run_one(image_path)


if __name__ == "__main__":
    main()
