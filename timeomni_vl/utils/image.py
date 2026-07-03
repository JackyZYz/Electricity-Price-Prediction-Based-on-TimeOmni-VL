from typing import Tuple

import numpy as np
from PIL import Image


def normalize_image(image: Image.Image) -> np.ndarray:
    arr = np.array(image).astype(np.float32) / 255.0
    return arr


def denormalize_image(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def resize_image(
    image: Image.Image,
    size: Tuple[int, int],
) -> Image.Image:
    return image.resize(size, Image.BILINEAR)
