import os
from pathlib import Path
import cv2
import numpy as np


def load_images_from_folder(folder_path, extensions=None, min_count=1):
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    images = []
    for file in sorted(folder.iterdir()):
        if file.suffix.lower() in extensions:
            img = cv2.imread(str(file))
            if img is None:
                continue
            images.append((file.name, img))

    if len(images) < min_count:
        raise ValueError(f"Expected at least {min_count} images, but found {len(images)}")

    return images


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def translate(image, tx, ty):
    rows, cols = image.shape[:2]
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, matrix, (cols, rows))


def rotate(image, angle, center=None, scale=1.0):
    rows, cols = image.shape[:2]
    if center is None:
        center = (cols // 2, rows // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, matrix, (cols, rows))


def scale(image, sx, sy=None):
    if sy is None:
        sy = sx
    rows, cols = image.shape[:2]
    new_width = max(1, int(cols * sx))
    new_height = max(1, int(rows * sy))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def shear(image, shx=0.0, shy=0.0):
    rows, cols = image.shape[:2]
    matrix = np.float32([[1, shx, 0], [shy, 1, 0]])
    new_width = int(cols + abs(shx * rows))
    new_height = int(rows + abs(shy * cols))
    return cv2.warpAffine(image, matrix, (new_width, new_height))


def reflect(image, axis="x"):
    if axis == "x":
        return cv2.flip(image, 0)
    if axis == "y":
        return cv2.flip(image, 1)
    if axis in {"xy", "yx", "both"}:
        return cv2.flip(image, -1)
    raise ValueError("axis must be 'x', 'y', or 'xy'")


def affine_transform(image, src_points, dst_points, output_size=None):
    if len(src_points) != 3 or len(dst_points) != 3:
        raise ValueError("Affine transform requires exactly 3 source and 3 destination points")

    rows, cols = image.shape[:2]
    if output_size is None:
        output_size = (cols, rows)

    src = np.float32(src_points)
    dst = np.float32(dst_points)
    matrix = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(image, matrix, output_size)


def perspective_transform(image, src_points, dst_points, output_size=None):
    if len(src_points) != 4 or len(dst_points) != 4:
        raise ValueError("Perspective transform requires exactly 4 source and 4 destination points")

    rows, cols = image.shape[:2]
    if output_size is None:
        output_size = (cols, rows)

    src = np.float32(src_points)
    dst = np.float32(dst_points)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, output_size)


def ensure_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def save_image(output_folder, base_name, suffix, image):
    output_folder = ensure_folder(output_folder)
    name = f"{Path(base_name).stem}_{suffix}{Path(base_name).suffix}"
    cv2.imwrite(str(output_folder / name), image)
    return output_folder / name


def process_dataset(dataset_folder, result_folder):
    images = load_images_from_folder(dataset_folder, min_count=20)

    for filename, image in images:
        gray = convert_to_grayscale(image)
        save_image(result_folder / "Grayscale Result", filename, "gray", gray)

        translated = translate(gray, tx=50, ty=30)
        save_image(result_folder / "Translation Result", filename, "translated", translated)

        rotated = rotate(gray, angle=30)
        save_image(result_folder / "Rotation Result", filename, "rotated", rotated)

        scaled = scale(gray, sx=1.2, sy=1.2)
        save_image(result_folder / "Scaling Result", filename, "scaled", scaled)

        sheared = shear(gray, shx=0.3, shy=0.0)
        save_image(result_folder / "Shearing Result", filename, "sheared", sheared)

        reflected = reflect(gray, axis="y")
        save_image(result_folder / "Reflection Result", filename, "reflected", reflected)

        rows, cols = gray.shape[:2]
        src_tri = [(0, 0), (cols - 1, 0), (0, rows - 1)]
        dst_tri = [(0, int(rows * 0.33)), (int(cols * 0.65), int(rows * 0.35)), (int(cols * 0.15), int(rows * 0.6))]
        affine = affine_transform(gray, src_tri, dst_tri)
        save_image(result_folder / "Affine Transformation Result", filename, "affine", affine)

        src_quad = [(0, 0), (cols - 1, 0), (cols - 1, rows - 1), (0, rows - 1)]
        dst_quad = [(0, 0), (cols - 1, int(rows * 0.1)), (int(cols * 0.85), rows - 1), (int(cols * 0.2), int(rows * 0.9))]
        perspective = perspective_transform(gray, src_quad, dst_quad)
        save_image(result_folder / "Perspetive Transform Result", filename, "perspective", perspective)

    print(f"Processed {len(images)} images from {dataset_folder}")
    print(f"Results saved in: {result_folder}")


if __name__ == "__main__":
    dataset_folder = Path("Data Building Photo")
    result_folder = Path("Result")
    process_dataset(dataset_folder, result_folder)
