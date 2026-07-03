import numpy as np


class GenerationMetrics:
    @staticmethod
    def mae(pred: np.ndarray, ref: np.ndarray) -> float:
        return float(np.nanmean(np.abs(pred - ref)))

    @staticmethod
    def rmse(pred: np.ndarray, ref: np.ndarray) -> float:
        return float(np.sqrt(np.nanmean((pred - ref) ** 2)))

    @staticmethod
    def nmape(pred: np.ndarray, ref: np.ndarray) -> float:
        scale = np.nanmean(np.abs(ref))
        if scale == 0:
            scale = 1e-6
        return float(np.nanmean(np.abs(pred - ref) / scale))

    @staticmethod
    def nmae(pred: np.ndarray, ref: np.ndarray) -> float:
        scale = np.nanmean(np.abs(ref))
        if scale == 0:
            scale = 1e-6
        return float(np.nanmean(np.abs(pred - ref)) / scale)

    @staticmethod
    def mase(pred: np.ndarray, ref: np.ndarray) -> float:
        naive_errors = np.abs(ref[1:] - ref[:-1])
        scale = np.nanmean(naive_errors)
        if scale == 0:
            scale = 1e-6
        return float(np.nanmean(np.abs(pred - ref)) / scale)

    @staticmethod
    def direction_accuracy(pred: np.ndarray, ref: np.ndarray) -> float:
        pred = np.asarray(pred).reshape(-1)
        ref = np.asarray(ref).reshape(-1)
        if len(pred) < 2 or len(ref) < 2:
            return 0.0
        pred_diff = np.diff(pred)
        ref_diff = np.diff(ref)
        matches = (pred_diff > 0) == (ref_diff > 0)
        return float(np.nanmean(matches))

    @staticmethod
    def mape(pred: np.ndarray, ref: np.ndarray) -> float:
        mask = ref != 0
        return float(np.nanmean(np.abs((pred[mask] - ref[mask]) / ref[mask])))

    @staticmethod
    def skill_score(pred: np.ndarray, ref: np.ndarray, baseline: np.ndarray) -> float:
        model_mae = GenerationMetrics.mae(pred, ref)
        baseline_mae = GenerationMetrics.mae(baseline, ref)
        if baseline_mae == 0:
            return 0.0
        return 1.0 - model_mae / baseline_mae
