from pathlib import Path
from typing import Dict, List

import pandas as pd

from timeomni_vl.exceptions import DataLoadError
from timeomni_vl.logger import get_logger

logger = get_logger(__name__)


class CSVLoader:
    def __init__(self, data_dir: str, encoding: str = "utf-8"):
        self.data_dir = Path(data_dir)
        self.encoding = encoding

    def load_all(self) -> Dict[str, pd.DataFrame]:
        files = sorted(self.data_dir.rglob("*.csv"))
        if not files:
            raise DataLoadError(f"No CSV files found in {self.data_dir}")

        loaded = {}
        for file_path in files:
            key = str(file_path.relative_to(self.data_dir))
            try:
                loaded[key] = self.load_single(file_path)
            except Exception as e:
                logger.warning(f"Failed to load {key}: {e}")
        return loaded

    def load_single(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(file_path, encoding=self.encoding)
        self._validate_columns(df, file_path)
        return df

    def _validate_columns(self, df: pd.DataFrame, file_path: Path) -> None:
        if len(df.columns) < 10:
            raise DataLoadError(f"{file_path}: expected at least 10 columns, got {len(df.columns)}")
        meta_cols = list(df.columns[:9])
        expected_meta = ["ID", "父ID", "数据类型", "数据地区", "数据所属菜单", "数据来源", "数据描述", "日期", "更新时间"]
        if meta_cols != expected_meta:
            logger.debug(f"{file_path}: metadata columns differ from standard header")
