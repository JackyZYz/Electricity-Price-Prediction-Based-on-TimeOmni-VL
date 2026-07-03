from typing import Dict

import numpy as np
from PIL import Image

from timeomni_vl.bitsi.renderer import ImageRenderer
from timeomni_vl.exceptions import BiTSIError
from timeomni_vl.logger import get_logger

logger = get_logger(__name__)


class I2TSConverter:
    def __init__(
        self,
        frequency: int,
        image_size: int,
    ):
        self.frequency = frequency
        self.image_size = image_size
        self.renderer = ImageRenderer()

    def convert(
        self,
        image: Image.Image,
        n_vars: int,
        target_length: int,
        stats: Dict[str, np.ndarray],
        task: str = "forecasting",
    ) -> np.ndarray:
        image_array = np.array(image.convert("RGB"))
        bands = self._extract_bands(image_array, n_vars)

        sequences = []
        for i in range(n_vars):
            grid = self._resize_to_grid(bands[i])
            seq = self._unfold_grid(grid, target_length)
            sequences.append(seq)

        x_hat = np.stack(sequences, axis=1) if n_vars > 1 else sequences[0].reshape(-1, 1)
        return x_hat

    def _extract_bands(
        self,
        image_array: np.ndarray,
        n_vars: int,
    ) -> np.ndarray:
        h, w, _ = image_array.shape
        available_h = h - (n_vars + 1)
        band_h = max(1, available_h // n_vars)
        separator = max(1, (h - n_vars * band_h) // (n_vars + 1))

        bands = []
        for i in range(n_vars):
            y_start = separator + i * (band_h + separator)
            y_end = y_start + band_h
            bands.append(image_array[y_start:y_end, :])
        return np.stack(bands, axis=0)

    def _resize_to_grid(
        self,
        band: np.ndarray,
    ) -> np.ndarray:
        img = Image.fromarray(band)
        resized = img.resize((self.image_size, self.frequency), Image.BILINEAR)
        gray = np.array(resized.convert("L"))
        grid = gray.astype(np.float32) / 255.0 * 2.0 - 1.0
        return grid

    def _unfold_grid(
        self,
        grid: np.ndarray,
        target_length: int,
    ) -> np.ndarray:
        n_cycles = grid.shape[0]
        seq = grid.reshape(-1)
        seq = seq[:target_length]
        if len(seq) < target_length:
            seq = np.pad(seq, (0, target_length - len(seq)), mode="edge")
        return seq
