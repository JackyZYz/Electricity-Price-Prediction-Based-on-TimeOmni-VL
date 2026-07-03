import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from timeomni_vl.bitsi.i2ts import I2TSConverter
from timeomni_vl.config import ConfigManager
from timeomni_vl.inference.generation import GenerationInferencer
from timeomni_vl.inference.understanding import UnderstandingInferencer
from timeomni_vl.models import build_adapter
from timeomni_vl.utils.io import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--task", choices=["understand", "forecast", "impute"], required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="outputs/prediction.npy")
    args = parser.parse_args()

    cfg = ConfigManager(args.config)
    model_cfg = cfg.get_model_config()
    bitsi_cfg = cfg.get_bitsi_config()
    data_cfg = cfg.get_data_config()

    adapter = build_adapter(model_cfg.backbone)
    adapter.load(model_cfg.model_path, model_cfg.device)

    image = Image.open(args.image)

    if args.task == "understand":
        inferencer = UnderstandingInferencer(adapter)
        result = inferencer.infer(image, args.prompt)
        print(result)
    else:
        i2ts = I2TSConverter(data_cfg.frequency, bitsi_cfg.image_size)
        inferencer = GenerationInferencer(adapter, i2ts)
        if args.task == "forecast":
            result = inferencer.forecast(
                image,
                args.prompt,
                n_vars=1,
                target_length=data_cfg.forecast_days * data_cfg.points_per_day,
                stats={"mu": np.array([0.0]), "sigma": np.array([1.0])},
            )
        else:
            result = inferencer.impute(
                image,
                args.prompt,
                n_vars=1,
                target_length=data_cfg.forecast_days * data_cfg.points_per_day,
                stats={"mu": np.array([0.0]), "sigma": np.array([1.0])},
            )
        output_path = Path(args.output)
        ensure_dir(output_path.parent)
        np.save(output_path, result["predicted_values"])
        print(f"Saved prediction to {output_path}")


if __name__ == "__main__":
    main()
