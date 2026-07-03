from typing import Dict, List

import numpy as np
import pandas as pd

from timeomni_vl.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    def __init__(
        self,
        missing_threshold: float = 0.5,
        method: str = "mixed",
    ):
        self.missing_threshold = missing_threshold
        self.method = method

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.drop_high_missing_variables(df)
        df = self.fill_missing(df, self.method)
        df = df.dropna(how="all", axis=0)
        return df

    def drop_high_missing_variables(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        missing_ratio = df.isna().mean()
        keep_cols = missing_ratio[missing_ratio <= self.missing_threshold].index.tolist()
        dropped = [c for c in df.columns if c not in keep_cols]
        if dropped:
            logger.info(f"Dropped variables with high missing ratio: {dropped}")
        return df[keep_cols]

    def fill_missing(
        self,
        df: pd.DataFrame,
        method: str = "mixed",
    ) -> pd.DataFrame:
        if method == "forward":
            return df.ffill().bfill()

        if method == "linear":
            return df.interpolate(method="linear", limit_direction="both")

        if method == "daily":
            return self._fill_daily(df)

        if method == "mixed":
            df = df.ffill(limit=4)
            df = df.interpolate(method="linear", limit=8, limit_direction="both")
            df = self._fill_daily(df)
            df = df.ffill().bfill()
            return df

        raise ValueError(f"Unknown fill method: {method}")

    def _fill_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        filled = df.copy()
        for col in df.columns:
            series = df[col]
            nan_mask = series.isna()
            if not nan_mask.any():
                continue
            group = series.index.dayofweek
            daily_mean = series.groupby(group).transform("mean")
            filled[col] = series.fillna(daily_mean)
        return filled

    def detect_outliers(
        self,
        series: pd.Series,
        method: str = "iqr",
    ) -> pd.Series:
        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return (series < lower) | (series > upper)

        if method == "zscore":
            mean = series.mean()
            std = series.std()
            return np.abs(series - mean) > 3 * std

        raise ValueError(f"Unknown outlier method: {method}")

    def clip_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
    ) -> pd.DataFrame:
        df = df.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            mask = self.detect_outliers(df[col], method)
            if mask.any():
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                df[col] = df[col].clip(lower, upper)
        return df
