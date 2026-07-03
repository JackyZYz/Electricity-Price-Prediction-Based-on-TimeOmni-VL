from typing import Tuple

import numpy as np
from PIL import Image


class ImageRenderer:
    @staticmethod
    def grayscale_to_rgb(
        grid: np.ndarray,
        color: Tuple[int, int, int],
    ) -> np.ndarray:
        grid = np.clip(grid, -1.0, 1.0)
        gray = ((grid + 1.0) / 2.0 * 255.0).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
        for i in range(3):
            rgb[..., i] = (rgb[..., i] * color[i] / 255.0).astype(np.uint8)
        return rgb

    @staticmethod
    def resize_band(
        band: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        img = Image.fromarray(band)
        img_resized = img.resize((width, height), Image.BILINEAR)
        return np.array(img_resized)

    @staticmethod
    def assemble_image(
        bands: np.ndarray,
        separator: int = 1,
        bg_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> Image.Image:
        n_bands, h, w, c = bands.shape
        total_h = n_bands * h + (n_bands + 1) * separator
        canvas = np.zeros((total_h, w, c), dtype=np.uint8)
        canvas[:, :] = bg_color
        for i in range(n_bands):
            y = separator + i * (h + separator)
            canvas[y : y + h, :] = bands[i]
        return Image.fromarray(canvas)

    @staticmethod
    def normalize_grid(
        grid: np.ndarray,
        vmin: float = -1.0,
        vmax: float = 1.0,
    ) -> np.ndarray:
        grid = np.clip(grid, vmin, vmax)
        return ((grid - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
