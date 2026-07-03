from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from timeomni_vl.bitsi.i2ts import I2TSConverter
from timeomni_vl.models.adapter import BackboneAdapter
from timeomni_vl.utils.text import extract_between_tags


class GenerationInferencer:
    def __init__(
        self,
        adapter: BackboneAdapter,
        i2ts_converter: I2TSConverter,
        use_cot: bool = True,
    ):
        self.adapter = adapter
        self.i2ts = i2ts_converter
        self.use_cot = use_cot

    def forecast(
        self,
        source_image: Image.Image,
        instruction: str,
        n_vars: int,
        target_length: int,
        stats: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        cot = ""
        if self.use_cot:
            raw_cot = self.adapter.understand(
                source_image,
                "Analyze this electricity price TS-image and summarize key patterns.",
            )
            cot = extract_between_tags(raw_cot, "think")

        generated_image = self.adapter.generate(
            source_image,
            instruction,
            cot=cot,
            image_size=(source_image.width, source_image.height),
        )
        predicted_values = self.i2ts.convert(
            generated_image,
            n_vars=n_vars,
            target_length=target_length,
            stats=stats,
            task="forecasting",
        )
        return {
            "cot": cot,
            "generated_image": generated_image,
            "predicted_values": predicted_values,
        }

    def impute(
        self,
        source_image: Image.Image,
        instruction: str,
        n_vars: int,
        target_length: int,
        stats: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        cot = ""
        if self.use_cot:
            raw_cot = self.adapter.understand(
                source_image,
                "Identify the masked regions in this TS-image.",
            )
            cot = extract_between_tags(raw_cot, "think")

        generated_image = self.adapter.generate(
            source_image,
            instruction,
            cot=cot,
            image_size=(source_image.width, source_image.height),
        )
        predicted_values = self.i2ts.convert(
            generated_image,
            n_vars=n_vars,
            target_length=target_length,
            stats=stats,
            task="imputation",
        )
        return {
            "cot": cot,
            "generated_image": generated_image,
            "predicted_values": predicted_values,
        }
