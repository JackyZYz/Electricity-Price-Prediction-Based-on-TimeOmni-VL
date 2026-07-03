from typing import Any, Dict, List

import numpy as np
from PIL import Image

from timeomni_vl.logger import get_logger

logger = get_logger(__name__)


class UnderstandingTaskGenerator:
    def __init__(
        self,
        frequency: int,
        image_size: int,
    ):
        self.frequency = frequency
        self.image_size = image_size

    def generate_all(
        self,
        sample,
    ) -> List[Dict[str, Any]]:
        n_vars = sample.context.shape[1]
        total_cycles = sample.context.shape[0] // self.frequency
        var_idx = sample.metadata.get("target_var_idx", 0)

        qas = [
            self.qa1_variable_counting(n_vars),
            self.qa2_variable_y_range(var_idx, n_vars),
            self.qa3_cycle_bounding_box(var_idx, 0, total_cycles),
        ]

        values = sample.context[:, var_idx]
        if total_cycles >= 2:
            qas.append(self.qa4_mean_comparison(var_idx, 0, 1, values))
        qas.append(self.qa5_anomaly_detection(var_idx, values))
        qas.append(self.qa6_trend_analysis(var_idx, 0, values))
        return qas

    def qa1_variable_counting(
        self,
        n_vars: int,
    ) -> Dict[str, Any]:
        return {
            "task_id": "qa1",
            "question": "How many variables are encoded in the TS-image?",
            "cot": f"The TS-image contains {n_vars} horizontal color bands, each representing one variable.",
            "answer": n_vars,
        }

    def qa2_variable_y_range(
        self,
        var_idx: int,
        n_vars: int,
    ) -> Dict[str, Any]:
        band_height = self.image_size // n_vars
        y_min = var_idx * band_height
        y_max = (var_idx + 1) * band_height
        return {
            "task_id": "qa2",
            "question": f"What is the vertical range of variable {var_idx}?",
            "cot": f"Variable {var_idx} occupies band {var_idx} from y={y_min} to y={y_max}.",
            "answer": [y_min, y_max],
        }

    def qa3_cycle_bounding_box(
        self,
        var_idx: int,
        cycle_idx: int,
        total_cycles: int,
    ) -> Dict[str, Any]:
        band_height = self.image_size // max(1, total_cycles)
        x_min = cycle_idx * band_height
        x_max = (cycle_idx + 1) * band_height
        return {
            "task_id": "qa3",
            "question": f"What is the bounding box of cycle {cycle_idx} for variable {var_idx}?",
            "cot": f"Cycle {cycle_idx} spans x={x_min} to x={x_max}.",
            "answer": [x_min, x_max],
        }

    def qa4_mean_comparison(
        self,
        var_idx: int,
        cycle_a: int,
        cycle_b: int,
        values: np.ndarray,
    ) -> Dict[str, Any]:
        mean_a = float(np.nanmean(values[cycle_a * self.frequency : (cycle_a + 1) * self.frequency]))
        mean_b = float(np.nanmean(values[cycle_b * self.frequency : (cycle_b + 1) * self.frequency]))
        brighter = cycle_a if mean_a > mean_b else cycle_b
        return {
            "task_id": "qa4",
            "question": f"Which cycle has a higher average value, cycle {cycle_a} or cycle {cycle_b}?",
            "cot": f"Cycle {cycle_a} mean={mean_a:.2f}, cycle {cycle_b} mean={mean_b:.2f}.",
            "answer": brighter,
        }

    def qa5_anomaly_detection(
        self,
        var_idx: int,
        values: np.ndarray,
        threshold: float = 18.0,
    ) -> Dict[str, Any]:
        anomalies = np.where(values > threshold)[0].tolist()
        return {
            "task_id": "qa5",
            "question": f"Which time points in variable {var_idx} exceed {threshold}?",
            "cot": f"Detected {len(anomalies)} points above threshold {threshold}.",
            "answer": anomalies,
        }

    def qa6_trend_analysis(
        self,
        var_idx: int,
        cycle_idx: int,
        values: np.ndarray,
    ) -> Dict[str, Any]:
        cycle_values = values[cycle_idx * self.frequency : (cycle_idx + 1) * self.frequency]
        peak_idx = int(np.nanargmax(cycle_values))
        trough_idx = int(np.nanargmin(cycle_values))
        description = f"Peak at point {peak_idx}, trough at point {trough_idx}."
        return {
            "task_id": "qa6",
            "question": f"Describe the trend of variable {var_idx} in cycle {cycle_idx}.",
            "cot": description,
            "answer": {
                "peak": peak_idx,
                "trough": trough_idx,
                "description": description,
            },
        }


class UnderstandingEvaluator:
    def __init__(self):
        self.scorers = {
            "qa1": self.exact_match,
            "qa2": self.iou_score,
            "qa3": self.iou_score,
            "qa4": self.exact_match,
            "qa5": self.weighted_accuracy,
            "qa6": self.composite_qa6,
        }

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        scores = {}
        grouped = {"qa1": [], "qa2": [], "qa3": [], "qa4": [], "qa5": [], "qa6": []}
        for pred, ref in zip(predictions, references):
            task_id = ref.get("task_id", "qa1")
            score = self.scorers[task_id](pred.get("answer"), ref.get("answer"))
            grouped[task_id].append(score)

        for task_id, task_scores in grouped.items():
            if task_scores:
                scores[task_id] = float(np.mean(task_scores))
        if scores:
            scores["average"] = float(np.mean(list(scores.values())))
        return scores

    def exact_match(self, pred: Any, ref: Any) -> float:
        return 1.0 if pred == ref else 0.0

    def iou_score(self, pred: Any, ref: Any) -> float:
        try:
            p = list(pred)
            r = list(ref)
            inter = max(0, min(p[1], r[1]) - max(p[0], r[0]))
            union = max(p[1], r[1]) - min(p[0], r[0])
            return inter / union if union > 0 else 0.0
        except Exception:
            return 0.0

    def weighted_accuracy(self, pred: Any, ref: Any) -> float:
        try:
            p = set(pred)
            r = set(ref)
            if not r:
                return 1.0 if not p else 0.0
            return len(p & r) / len(r)
        except Exception:
            return 0.0

    def composite_qa6(self, pred: Any, ref: Any) -> float:
        try:
            if not isinstance(pred, dict) or not isinstance(ref, dict):
                return 0.0
            score = 0.0
            if pred.get("peak") == ref.get("peak"):
                score += 0.5
            if pred.get("trough") == ref.get("trough"):
                score += 0.5
            return score
        except Exception:
            return 0.0
