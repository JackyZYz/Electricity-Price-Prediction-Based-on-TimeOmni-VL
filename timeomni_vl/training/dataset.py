from typing import Any, Dict, List

import numpy as np
from PIL import Image

from timeomni_vl.bitsi.i2ts import I2TSConverter
from timeomni_vl.bitsi.rfn import RobustFidelityNormalizer
from timeomni_vl.bitsi.ts2i import TS2IConverter
from timeomni_vl.config import BiTSIConfig, DataConfig
from timeomni_vl.data.sampler import Sample
from timeomni_vl.tasks.generation import GenerationTaskGenerator
from timeomni_vl.tasks.understanding import UnderstandingTaskGenerator


class TSUMMDatasetBuilder:
    def __init__(
        self,
        data_config: DataConfig,
        bitsi_config: BiTSIConfig,
    ):
        self.data_config = data_config
        self.bitsi_config = bitsi_config
        self.rfn = RobustFidelityNormalizer(
            alpha=bitsi_config.alpha,
            c_mad=bitsi_config.c_mad,
            kappa=bitsi_config.kappa,
        )
        self.ts2i = TS2IConverter(
            frequency=data_config.frequency,
            image_size=bitsi_config.image_size,
            color_map=bitsi_config.color_map,
        )
        self.i2ts = I2TSConverter(
            frequency=data_config.frequency,
            image_size=bitsi_config.image_size,
        )
        self.understanding_generator = UnderstandingTaskGenerator(
            frequency=data_config.frequency,
            image_size=bitsi_config.image_size,
        )
        self.generation_generator = GenerationTaskGenerator(
            frequency=data_config.frequency,
            context_days=data_config.context_days,
            forecast_days=data_config.forecast_days,
        )

    def build(
        self,
        samples: List[Sample],
        variable_names: List[str] = None,
    ) -> List[Dict[str, Any]]:
        dataset = []
        for sample in samples:
            ctx_norm, stats = self.rfn.fit_transform(sample.context)
            tgt_norm, _ = self.rfn.fit_transform(sample.target)

            src_image = self.ts2i.convert(ctx_norm, variable_names=variable_names)
            tgt_image = self.ts2i.convert(tgt_norm, variable_names=variable_names)

            qas = self.understanding_generator.generate_all(sample)
            cot = self.generation_generator.generate_cot(qas)
            instruction = "Predict the next day's electricity price based on the historical TS-image."

            dataset.append(
                {
                    "image_src": src_image,
                    "image_tgt": tgt_image,
                    "instruction": instruction,
                    "cot": cot,
                    "answer": "",
                    "task": sample.task,
                    "stats": stats,
                }
            )
        return dataset
