from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from timeomni_vl.config import DataConfig
from timeomni_vl.data.aligner import DateAligner
from timeomni_vl.data.cleaner import DataCleaner
from timeomni_vl.data.featurizer import Featurizer
from timeomni_vl.data.loader import CSVLoader
from timeomni_vl.data.sampler import RollingWindowSampler, Sample
from timeomni_vl.data.splitter import SplitResult, TimeSeriesSplitter
from timeomni_vl.logger import get_logger
from timeomni_vl.utils.io import ensure_dir, save_json

logger = get_logger(__name__)


class DataPipeline:
    def __init__(self, config: DataConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.raw_data_dir = Path(config.raw_data_dir)

    def run(self) -> Dict[str, List[Sample]]:
        logger.info("Loading CSV files...")
        loader = CSVLoader(self.raw_data_dir)
        data_dict = loader.load_all()
        logger.info(f"Loaded {len(data_dict)} CSV files")

        logger.info("Aligning dates...")
        aligner = DateAligner(self.config.frequency)
        aligned_df = aligner.align(data_dict, self.config.target_variable)
        logger.info(f"Aligned dataframe shape: {aligned_df.shape}")

        logger.info("Cleaning data...")
        cleaner = DataCleaner(missing_threshold=self.config.missing_threshold)
        cleaned_df = cleaner.clean(aligned_df)
        logger.info(f"Cleaned dataframe shape: {cleaned_df.shape}")

        if self.config.calendar_features:
            logger.info("Adding calendar features...")
            featurizer = Featurizer(self.config.frequency)
            cleaned_df = featurizer.add_calendar_features(cleaned_df)

        logger.info("Splitting data...")
        splitter = TimeSeriesSplitter(
            self.config.train_ratio,
            self.config.val_ratio,
            self.config.test_ratio,
        )
        split_result = splitter.split(cleaned_df)

        logger.info("Sampling windows...")
        context_length = self.config.context_days * self.config.points_per_day
        target_length = self.config.forecast_days * self.config.points_per_day
        sampler = RollingWindowSampler(
            context_length=context_length,
            target_length=target_length,
            stride=self.config.points_per_day,
            task="forecasting",
        )

        train_samples = sampler.sample(split_result.train, self.config.target_variable)
        val_samples = sampler.sample(split_result.val, self.config.target_variable)
        test_samples = sampler.sample(split_result.test, self.config.target_variable)
        logger.info(
            f"Generated samples: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
        )

        self._save_outputs(cleaned_df, split_result, train_samples, val_samples, test_samples)
        return {"train": train_samples, "val": val_samples, "test": test_samples}

    def _save_outputs(
        self,
        df: pd.DataFrame,
        split_result: SplitResult,
        train_samples: List[Sample],
        val_samples: List[Sample],
        test_samples: List[Sample],
    ) -> None:
        ensure_dir(self.output_dir)
        try:
            df.to_parquet(self.output_dir / "aligned_cleaned.parquet")
            split_result.train.to_parquet(self.output_dir / "train.parquet")
            split_result.val.to_parquet(self.output_dir / "val.parquet")
            split_result.test.to_parquet(self.output_dir / "test.parquet")
        except ImportError:
            logger.warning("pyarrow not installed, falling back to CSV output")
            df.to_csv(self.output_dir / "aligned_cleaned.csv")
            split_result.train.to_csv(self.output_dir / "train.csv")
            split_result.val.to_csv(self.output_dir / "val.csv")
            split_result.test.to_csv(self.output_dir / "test.csv")

        metadata = {
            "n_variables": df.shape[1],
            "variable_names": list(df.columns),
            "n_train": len(split_result.train),
            "n_val": len(split_result.val),
            "n_test": len(split_result.test),
            "n_train_samples": len(train_samples),
            "n_val_samples": len(val_samples),
            "n_test_samples": len(test_samples),
        }
        save_json(metadata, self.output_dir / "metadata.json")


def build_data_pipeline(config: DataConfig) -> DataPipeline:
    return DataPipeline(config)
