import argparse
from pathlib import Path

import cv2
import numpy as np


DATA_DIR = Path("data/raw")
DEFAULT_IMAGE = Path("data/raw/chessboard.png")
OUT_DIR = Path("data/output/sensitivity")


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


def parse_list(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def build_row(img: np.ndarray, title: str) -> np.ndarray:
    top_pad = 36
    h, w = img.shape[:2]
    canvas = np.zeros((h + top_pad, w, 3), dtype=np.uint8)
    canvas[top_pad:, :, :] = img
    cv2.putText(canvas, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2, cv2.LINE_AA)
    return canvas


def merge_grid(rows: list[np.ndarray], cols: int = 3) -> np.ndarray:
    if not rows:
        raise ValueError("rows must not be empty")

    padded = rows[:]
    empty = np.zeros_like(rows[0])
    while len(padded) % cols != 0:
        padded.append(empty)

    grid_rows = []
    for i in range(0, len(padded), cols):
        grid_rows.append(np.hstack(padded[i : i + cols]))
    return np.vstack(grid_rows)


def run_sweep(
    image: np.ndarray,
    parameter_name: str,
    values: list,
    fixed_k: float,
    fixed_block_size: int,
    fixed_sobel_ksize: int,
    fixed_threshold: float,
) -> np.ndarray:
    rows = []
    for value in values:
        k = fixed_k
        block_size = fixed_block_size
        sobel_ksize = fixed_sobel_ksize
        threshold = fixed_threshold

        if parameter_name == "k":
            k = float(value)
        elif parameter_name == "blockSize":
            block_size = int(value)
        elif parameter_name == "sobel_ksize":
            sobel_ksize = int(value)
        elif parameter_name == "threshold":
            threshold = float(value)
        else:
            raise ValueError(f"unsupported parameter: {parameter_name}")

        result = detect_opencv(
            image,
            k=k,
            block_size=block_size,
            sobel_ksize=sobel_ksize,
            threshold_ratio=threshold,
        )
        rows.append(build_row(result, f"{parameter_name}={value}"))
    return merge_grid(rows, cols=3)


def run_for_one_image(
    image_path: Path,
    k_values: list[float],
    block_values: list[int],
    sobel_values: list[int],
    threshold_values: list[float],
    fixed_k: float,
    fixed_block_size: int,
    fixed_sobel_ksize: int,
    fixed_threshold: float,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"skip unreadable image -> {image_path}")
        return

    image_out_dir = OUT_DIR / image_path.stem
    image_out_dir.mkdir(parents=True, exist_ok=True)

    sweeps = [
        ("k", k_values, "k_sweep.jpg"),
        ("blockSize", block_values, "block_size_sweep.jpg"),
        ("sobel_ksize", sobel_values, "sobel_ksize_sweep.jpg"),
        ("threshold", threshold_values, "threshold_sweep.jpg"),
    ]

    for parameter_name, values, out_name in sweeps:
        merged = run_sweep(
            image=image,
            parameter_name=parameter_name,
            values=values,
            fixed_k=fixed_k,
            fixed_block_size=fixed_block_size,
            fixed_sobel_ksize=fixed_sobel_ksize,
            fixed_threshold=fixed_threshold,
        )
        out_path = image_out_dir / out_name
        cv2.imwrite(str(out_path), merged)
        print(f"saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Harris 参数敏感性实验（OpenCV cornerHarris）")
    parser.add_argument("--image", type=Path, default=None, help="仅对单张图像执行（默认对 data/raw 全部图像执行）")
    parser.add_argument("--k-values", default="0.02,0.04,0.06,0.08")
    parser.add_argument("--block-values", default="2,3,5,7")
    parser.add_argument("--sobel-values", default="3,5,7")
    parser.add_argument("--threshold-values", default="0.005,0.01,0.02,0.05")
    parser.add_argument("--fixed-k", type=float, default=0.04)
    parser.add_argument("--fixed-block-size", type=int, default=2)
    parser.add_argument("--fixed-sobel-ksize", type=int, default=3)
    parser.add_argument("--fixed-threshold", type=float, default=0.01)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    k_values = parse_list(args.k_values, float)
    block_values = parse_list(args.block_values, int)
    sobel_values = parse_list(args.sobel_values, int)
    threshold_values = parse_list(args.threshold_values, float)

    if args.image is not None:
        image_paths = [args.image]
    else:
        image_paths = sorted([p for p in DATA_DIR.glob("*") if p.is_file()])
        if not image_paths:
            image_paths = [DEFAULT_IMAGE]

    for image_path in image_paths:
        run_for_one_image(
            image_path=image_path,
            k_values=k_values,
            block_values=block_values,
            sobel_values=sobel_values,
            threshold_values=threshold_values,
            fixed_k=args.fixed_k,
            fixed_block_size=args.fixed_block_size,
            fixed_sobel_ksize=args.fixed_sobel_ksize,
            fixed_threshold=args.fixed_threshold,
        )


if __name__ == "__main__":
    main()
