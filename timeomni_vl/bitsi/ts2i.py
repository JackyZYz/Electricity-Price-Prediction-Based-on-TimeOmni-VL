from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from timeomni_vl.bitsi.renderer import ImageRenderer
from timeomni_vl.exceptions import BiTSIError
from timeomni_vl.logger import get_logger

logger = get_logger(__name__)

DEFAULT_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 165, 0),
    (128, 0, 128),
    (255, 192, 203),
    (128, 128, 128),
]


class TS2IConverter:
    def __init__(
        self,
        frequency: int,
        image_size: int,
        color_map: Optional[Dict[str, str]] = None,
    ):
        self.frequency = frequency
        self.image_size = image_size
        self.renderer = ImageRenderer()
        self.color_map = color_map or {}

    def convert(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        task: str = "forecasting",
        variable_names: Optional[List[str]] = None,
    ) -> Image.Image:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        n_vars = x.shape[1]
        grids = self._fold_to_grid(x)
        band_h = max(1, self.image_size // (n_vars + 1))
        separator = max(1, band_h // 8)
        available_h = self.image_size - (n_vars + 1) * separator
        band_h = max(1, available_h // n_vars)
        band_w = self.image_size

        bands = []
        for i in range(n_vars):
            color = self._get_color(i, variable_names)
            grid = grids[i]
            resized = self.renderer.resize_band(grid, band_h, band_w)
            rgb = self.renderer.grayscale_to_rgb(resized.astype(np.float32) / 255.0 * 2.0 - 1.0, color)
            bands.append(rgb)
        bands = np.stack(bands, axis=0)

        image_array = self.renderer.assemble_image(bands, separator=separator)
        image_array = np.array(image_array)

        if mask is not None:
            image_array = self._apply_mask(image_array, mask, task, n_vars, band_h, separator)

        return Image.fromarray(image_array)

    def _fold_to_grid(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        n_vars = x.shape[1]
        n_points = x.shape[0]
        n_cycles = int(np.ceil(n_points / self.frequency))
        padded_len = n_cycles * self.frequency
        pad = np.zeros((padded_len - n_points, n_vars), dtype=np.float32)
        x_padded = np.concatenate([x, pad], axis=0)

        grids = x_padded.reshape(n_vars, n_cycles, self.frequency)
        return grids

    def _get_color(
        self,
        idx: int,
        variable_names: Optional[List[str]],
    ) -> Tuple[int, int, int]:
        if not variable_names:
            return DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

        var_name = variable_names[idx]

        # Exact match
        if var_name in self.color_map:
            hex_color = self.color_map[var_name]
            return self._hex_to_rgb(hex_color)

        # Partial match: key is a substring of variable name
        for key, hex_color in self.color_map.items():
            if key in var_name:
                return self._hex_to_rgb(hex_color)

        return DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        return tuple(int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

    def _apply_mask(
        self,
        image_array: np.ndarray,
        mask: np.ndarray,
        task: str,
        n_vars: int,
        band_h: int,
        separator: int,
    ) -> np.ndarray:
        if mask.ndim == 1:
            mask = np.repeat(mask[:, np.newaxis], n_vars, axis=1)

        mask_color = (0, 0, 0)
        for var_idx in range(n_vars):
            y_start = separator + var_idx * (band_h + separator)
            y_end = y_start + band_h
            var_mask = mask[:, var_idx]
            if task == "forecasting":
                masked_positions = np.where(var_mask)[0]
                if len(masked_positions) > 0:
                    x_start = int(masked_positions[0] / len(var_mask) * image_array.shape[1])
                    image_array[y_start:y_end, x_start:] = mask_color
            else:
                for t in range(len(var_mask)):
                    if var_mask[t]:
                        x1 = int(t / len(var_mask) * image_array.shape[1])
                        x2 = int((t + 1) / len(var_mask) * image_array.shape[1])
                        image_array[y_start:y_end, x1:x2] = mask_color
        return image_array
