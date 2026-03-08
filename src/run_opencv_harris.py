import cv2
import numpy as np
from pathlib import Path


DATA_DIR = Path("data/raw")
OUT_DIR = Path("data/output/opencv")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def run(image_path):

    img = cv2.imread(str(image_path))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # OpenCV Harris
    R = cv2.cornerHarris(
        gray,
        blockSize=2,
        ksize=3,
        k=0.04
    )

    # 膨胀增强角点
    R = cv2.dilate(R, None)

    result = img.copy()

    # 阈值
    threshold = 0.01 * R.max()

    ys, xs = np.where(R > threshold)

    for x, y in zip(xs, ys):
        cv2.circle(result, (x, y), 3, (0, 0, 255), 1)

    out = OUT_DIR / image_path.name

    cv2.imwrite(str(out), result)

    print(f"saved -> {out}")


def main():

    for img in DATA_DIR.glob("*"):
        run(img)


if __name__ == "__main__":
    main()