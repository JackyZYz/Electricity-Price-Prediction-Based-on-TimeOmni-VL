from typing import Dict, Tuple

import numpy as np


class RobustFidelityNormalizer:
    def __init__(
        self,
        alpha: float = 0.5,
        c_mad: float = 0.6745,
        kappa: float = 4.0,
    ):
        self.alpha = alpha
        self.c_mad = c_mad
        self.kappa = kappa

    def fit_transform(
        self,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        mu = np.nanmedian(x, axis=0, keepdims=True)
        sigma = self._compute_sigma(x, mu)
        sigma = np.where(sigma == 0, 1e-6, sigma)

        x_norm = (x - mu) / sigma
        x_norm = np.tanh(self.kappa * x_norm / self.alpha)

        stats = {
            "mu": mu.squeeze(),
            "sigma": sigma.squeeze(),
        }
        return x_norm, stats

    def inverse_transform(
        self,
        x_norm: np.ndarray,
        stats: Dict[str, np.ndarray],
    ) -> np.ndarray:
        mu = stats["mu"]
        sigma = stats["sigma"]

        if mu.ndim == 0:
            mu = mu.reshape(1)
            sigma = sigma.reshape(1)

        x_scaled = np.arctanh(np.clip(x_norm, -0.999, 0.999)) * self.alpha / self.kappa
        x = x_scaled * sigma + mu
        return x

    def _compute_sigma(self, x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        mad = np.nanmedian(np.abs(x - mu), axis=0, keepdims=True)
        sigma_mad = mad / self.c_mad
        sigma_std = np.nanstd(x, axis=0, keepdims=True)
        sigma = self.alpha * sigma_std + (1 - self.alpha) * sigma_mad
        return np.where(sigma == 0, 1e-6, sigma)
