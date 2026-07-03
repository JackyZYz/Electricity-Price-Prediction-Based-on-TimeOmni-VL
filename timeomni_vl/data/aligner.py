from pathlib import Path
from typing import Dict, List

import pandas as pd

from timeomni_vl.exceptions import DataAlignmentError
from timeomni_vl.logger import get_logger

logger = get_logger(__name__)

META_COLS = ["ID", "父ID", "数据类型", "数据地区", "数据所属菜单", "数据来源", "数据描述", "日期", "更新时间"]


class DateAligner:
    def __init__(self, frequency: int = 96):
        self.frequency = frequency

    def align(
        self,
        data_dict: Dict[str, pd.DataFrame],
        target_key: str,
    ) -> pd.DataFrame:
        if target_key not in data_dict:
            target_key = self._find_target_key(data_dict, target_key)

        target_df = data_dict[target_key]
        target_dates = self._parse_date_column(target_df).index.normalize().unique()
        logger.info(f"Aligning to target {target_key} with {len(target_dates)} dates")

        long_list = []
        for key, df in data_dict.items():
            var_name = self._extract_variable_name(df, key)
            long_df = self._reshape_to_long(df, var_name)
            if long_df is None or long_df.empty:
                continue
            long_df = long_df[long_df.index.normalize().isin(target_dates)]
            long_list.append(long_df)

        if not long_list:
            raise DataAlignmentError("No variables could be aligned")

        merged = long_list[0]
        for right in long_list[1:]:
            merged = merged.join(right, how="outer")

        merged = merged.sort_index()
        return merged

    def _find_target_key(self, data_dict: Dict[str, pd.DataFrame], target_name: str) -> str:
        for key, df in data_dict.items():
            if target_name in key or (df.shape[1] > 3 and target_name in str(df.iloc[0, 2])):
                return key
        raise DataAlignmentError(f"Target variable {target_name} not found")

    def _extract_variable_name(self, df: pd.DataFrame, key: str) -> str:
        base_name = Path(key).stem
        if df.shape[1] > 3 and pd.notna(df.iloc[0, 2]):
            data_type = str(df.iloc[0, 2]).strip()
            if data_type and data_type not in base_name:
                return f"{base_name}_{data_type}"
        return base_name

    def _parse_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df = df.dropna(subset=["日期"])
        df = df.set_index("日期")
        return df

    def _reshape_to_long(
        self,
        df: pd.DataFrame,
        var_name: str,
    ) -> pd.DataFrame:
        df = self._parse_date_column(df)
        point_cols = [c for c in df.columns if c not in META_COLS]
        point_cols = self._drop_duplicate_midnight(point_cols)
        if len(point_cols) != self.frequency:
            logger.warning(f"{var_name}: expected {self.frequency} point columns, got {len(point_cols)}")

        value_rows = []
        for date, row in df.iterrows():
            for i, col in enumerate(point_cols[: self.frequency]):
                value_rows.append(
                    {
                        "date": date,
                        "time_idx": i,
                        var_name: row.get(col),
                    }
                )
        long_df = pd.DataFrame(value_rows)
        if long_df.empty:
            return pd.DataFrame()
        long_df["datetime"] = long_df["date"] + pd.to_timedelta(long_df["time_idx"] * 15, unit="m")
        long_df = long_df.set_index("datetime")[[var_name]]
        long_df = long_df[~long_df.index.duplicated(keep="first")]
        return long_df

    def _drop_duplicate_midnight(self, point_cols: List[str]) -> List[str]:
        if "24:00" in point_cols and "00:00" in point_cols:
            point_cols = [c for c in point_cols if c != "24:00"]
        return point_cols
