from typing import Any, Dict, List


class UnderstandingMetrics:
    @staticmethod
    def exact_match(pred: Any, ref: Any) -> float:
        return 1.0 if pred == ref else 0.0

    @staticmethod
    def iou(pred_box: List[int], ref_box: List[int]) -> float:
        try:
            inter = max(0, min(pred_box[1], ref_box[1]) - max(pred_box[0], ref_box[0]))
            union = max(pred_box[1], ref_box[1]) - min(pred_box[0], ref_box[0])
            return inter / union if union > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def weighted_accuracy(pred: Dict, ref: Dict) -> float:
        try:
            p = set(pred)
            r = set(ref)
            if not r:
                return 1.0 if not p else 0.0
            return len(p & r) / len(r)
        except Exception:
            return 0.0

    @staticmethod
    def bert_score(pred_text: str, ref_text: str) -> float:
        raise NotImplementedError("BERTScore requires external dependency")
