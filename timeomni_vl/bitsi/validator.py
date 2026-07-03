from typing import Dict

import numpy as np

from timeomni_vl.bitsi.i2ts import I2TSConverter
from timeomni_vl.bitsi.rfn import RobustFidelityNormalizer
from timeomni_vl.bitsi.ts2i import TS2IConverter


class BiTSIValidator:
    def __init__(
        self,
        rfn: RobustFidelityNormalizer,
        ts2i: TS2IConverter,
        i2ts: I2TSConverter,
    ):
        self.rfn = rfn
        self.ts2i = ts2i
        self.i2ts = i2ts

    def validate(
        self,
        x: np.ndarray,
        task: str = "forecasting",
    ) -> Dict[str, float]:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        x_norm, stats = self.rfn.fit_transform(x)
        image = self.ts2i.convert(x_norm, task=task)
        x_hat_norm = self.i2ts.convert(
            image,
            n_vars=x.shape[1],
            target_length=x.shape[0],
            stats=stats,
            task=task,
        )
        x_hat = self.rfn.inverse_transform(x_hat_norm, stats)

        mae = float(np.nanmean(np.abs(x - x_hat)))
        rmse = float(np.sqrt(np.nanmean((x - x_hat) ** 2)))
        max_error = float(np.nanmax(np.abs(x - x_hat)))
        return {"mae": mae, "rmse": rmse, "max_error": max_error}
