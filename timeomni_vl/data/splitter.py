from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class TimeSeriesSplitter:
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(self, df: pd.DataFrame) -> SplitResult:
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:].copy()
        return SplitResult(train=train, val=val, test=test)

    def split_by_date(
        self,
        df: pd.DataFrame,
        train_end: str,
        val_end: str,
    ) -> SplitResult:
        train = df[df.index < train_end].copy()
        val = df[(df.index >= train_end) & (df.index < val_end)].copy()
        test = df[df.index >= val_end].copy()
        return SplitResult(train=train, val=val, test=test)
