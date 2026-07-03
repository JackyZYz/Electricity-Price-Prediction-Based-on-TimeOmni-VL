from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


class GenerationTaskGenerator:
    def __init__(
        self,
        frequency: int,
        context_days: int,
        forecast_days: int,
    ):
        self.frequency = frequency
        self.context_days = context_days
        self.forecast_days = forecast_days

    def generate_cot(
        self,
        understanding_qas: List[Dict[str, Any]],
    ) -> str:
        lines = []
        for qa in understanding_qas:
            lines.append(f"{qa['task_id']}: {qa['cot']}")
        lines.append(
            f"Forecast: predict the next {self.forecast_days} day(s) "
            f"({self.forecast_days * self.frequency} points) of electricity price."
        )
        return "\n".join(lines)

    def build_interleaved_sequence(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
        instruction: str,
        cot: str,
    ) -> Dict[str, Any]:
        return {
            "system": "You are a time series understanding and generation assistant.",
            "image_src": source_image,
            "instruction": instruction,
            "cot": cot,
            "image_tgt": target_image,
        }


class GenerationEvaluator:
    def __init__(self, metrics: List[str] = None):
        self.metrics = metrics or ["mae", "rmse", "nmape", "direction"]

    def evaluate(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
    ) -> Dict[str, float]:
        predictions = np.asarray(predictions)
        references = np.asarray(references)
        result = {}
        for metric in self.metrics:
            if metric == "mae":
                result["mae"] = self.mae(predictions, references)
            elif metric == "rmse":
                result["rmse"] = self.rmse(predictions, references)
            elif metric == "nmape":
                result["nmape"] = self.nmape(predictions, references)
            elif metric == "direction":
                result["direction"] = self.direction_accuracy(predictions, references)
            elif metric == "nmae":
                result["nmae"] = self.nmae(predictions, references)
        return result

    def mae(self, pred: np.ndarray, ref: np.ndarray) -> float:
        return float(np.nanmean(np.abs(pred - ref)))

    def rmse(self, pred: np.ndarray, ref: np.ndarray) -> float:
        return float(np.sqrt(np.nanmean((pred - ref) ** 2)))

    def nmape(self, pred: np.ndarray, ref: np.ndarray) -> float:
        scale = np.nanmean(np.abs(ref))
        if scale == 0:
            scale = 1e-6
        return float(np.nanmean(np.abs(pred - ref) / scale))

    def nmae(self, pred: np.ndarray, ref: np.ndarray) -> float:
        scale = np.nanmean(np.abs(ref))
        if scale == 0:
            scale = 1e-6
        return float(np.nanmean(np.abs(pred - ref)) / scale)

    def direction_accuracy(self, pred: np.ndarray, ref: np.ndarray) -> float:
        pred_diff = np.diff(pred, axis=0)
        ref_diff = np.diff(ref, axis=0)
        matches = (pred_diff > 0) == (ref_diff > 0)
        return float(np.nanmean(matches))
