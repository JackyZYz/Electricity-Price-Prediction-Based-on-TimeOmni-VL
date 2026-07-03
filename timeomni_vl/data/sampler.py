from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from timeomni_vl.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Sample:
    context: np.ndarray
    target: np.ndarray
    metadata: dict
    task: str


class RollingWindowSampler:
    def __init__(
        self,
        context_length: int,
        target_length: int,
        stride: int = 1,
        task: str = "forecasting",
        imputation_mask_ratio: Tuple[float, float] = (0.1, 0.5),
    ):
        self.context_length = context_length
        self.target_length = target_length
        self.stride = stride
        self.task = task
        self.imputation_mask_ratio = imputation_mask_ratio

    def sample(
        self,
        df: pd.DataFrame,
        target_var: str,
    ) -> List[Sample]:
        values = df.values.astype(np.float32)
        col_names = list(df.columns)
        target_var_idx = self._find_target_index(col_names, target_var)

        total_length = self.context_length + self.target_length
        samples = []
        for start_idx in range(0, len(values) - total_length + 1, self.stride):
            if self.task == "forecasting":
                sample = self._create_forecast_sample(values, start_idx, target_var_idx)
            elif self.task == "imputation":
                sample = self._create_imputation_sample(values, start_idx, target_var_idx)
            else:
                raise ValueError(f"Unknown task: {self.task}")
            samples.append(sample)
        return samples

    def _find_target_index(self, col_names: List[str], target_var: str) -> int:
        if target_var in col_names:
            return col_names.index(target_var)
        for i, name in enumerate(col_names):
            if target_var in name:
                logger.info(f"Resolved target variable '{target_var}' to column '{name}'")
                return i
        logger.warning(f"Target variable '{target_var}' not found, defaulting to first column")
        return 0

    def _create_forecast_sample(
        self,
        values: np.ndarray,
        start_idx: int,
        target_var_idx: int,
    ) -> Sample:
        context = values[start_idx : start_idx + self.context_length]
        target = values[
            start_idx + self.context_length : start_idx + self.context_length + self.target_length
        ]
        return Sample(
            context=context,
            target=target,
            metadata={"start_idx": start_idx, "target_var_idx": target_var_idx},
            task="forecasting",
        )

    def _create_imputation_sample(
        self,
        values: np.ndarray,
        start_idx: int,
        target_var_idx: int,
    ) -> Sample:
        total_length = self.context_length + self.target_length
        seq = values[start_idx : start_idx + total_length].copy()
        mask_ratio = np.random.uniform(*self.imputation_mask_ratio)
        n_mask = max(1, int(total_length * mask_ratio))
        mask_start = np.random.randint(0, total_length - n_mask + 1)
        seq[mask_start : mask_start + n_mask] = np.nan
        context = seq
        target = values[start_idx : start_idx + total_length]
        return Sample(
            context=context,
            target=target,
            metadata={
                "start_idx": start_idx,
                "target_var_idx": target_var_idx,
                "mask_start": mask_start,
                "mask_length": n_mask,
            },
            task="imputation",
        )
