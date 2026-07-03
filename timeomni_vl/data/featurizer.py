from typing import List

import numpy as np
import pandas as pd

from timeomni_vl.logger import get_logger

logger = get_logger(__name__)


class Featurizer:
    def __init__(self, frequency: int = 96):
        self.frequency = frequency

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.add_calendar_features(df)
        df = self.add_lag_features(df)
        df = self.add_rolling_stats(df)
        df = self.add_diff_features(df)
        return df

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["hour"] = df.index.hour
        df["minute"] = df.index.minute
        df["dayofweek"] = df.index.dayofweek
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
        return df

    def add_lag_features(
        self,
        df: pd.DataFrame,
        lags: List[int] = None,
    ) -> pd.DataFrame:
        if lags is None:
            lags = [1, 2, 7]
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["hour", "minute", "dayofweek", "is_weekend"]]
        for col in numeric_cols:
            for lag in lags:
                df[f"{col}_lag{lag}d"] = df[col].shift(lag * self.frequency)
        return df

    def add_rolling_stats(
        self,
        df: pd.DataFrame,
        windows: List[int] = None,
    ) -> pd.DataFrame:
        if windows is None:
            windows = [96, 96 * 7]
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if not c.endswith(("_lag1d", "_lag2d", "_lag7d"))]
        for col in numeric_cols:
            for window in windows:
                df[f"{col}_roll{window}_mean"] = df[col].rolling(window=window, min_periods=1).mean()
                df[f"{col}_roll{window}_std"] = df[col].rolling(window=window, min_periods=1).std()
        return df

    def add_diff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if not c.endswith(("_mean", "_std"))]
        for col in numeric_cols[:10]:
            df[f"{col}_diff1"] = df[col].diff(1)
            df[f"{col}_diff{self.frequency}"] = df[col].diff(self.frequency)
        return df
