from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    raw_data_dir: str
    output_dir: str
    frequency: int = 96
    points_per_day: int = 96
    context_days: int = 7
    forecast_days: int = 1
    long_context_days: int = 14
    long_forecast_days: int = 3
    target_variable: str = "统一结算点电价临时结果"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    missing_threshold: float = 0.5
    selected_variables: Optional[List[str]] = None
    calendar_features: bool = True


@dataclass
class BiTSIConfig:
    image_size: int = 896
    alpha: float = 0.5
    c_mad: float = 0.6745
    kappa: float = 4.0
    color_map: Dict[str, str] = field(default_factory=dict)
    max_variables: Optional[int] = None


@dataclass
class ModelConfig:
    backbone: str = "mock"
    model_path: Optional[str] = None
    device: str = "auto"
    dtype: str = "bfloat16"


@dataclass
class TrainingConfig:
    lr: float = 3e-5
    batch_size: int = 1
    num_epochs: int = 100
    warmup_ratio: float = 0.05
    lambda_und: float = 1.0
    lambda_gen: float = 1.0
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "bf16"
    max_grad_norm: float = 1.0
    save_every: int = 500
    eval_every: int = 100
    lora_enabled: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class EvalConfig:
    metrics: List[str] = field(default_factory=lambda: ["mae", "rmse", "nmape", "direction"])
    visualize: bool = True
    num_samples: int = 10


class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.raw = self._load_yaml()

    def _load_yaml(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_data_config(self) -> DataConfig:
        return DataConfig(**self.raw.get("data", {}))

    def get_bitsi_config(self) -> BiTSIConfig:
        return BiTSIConfig(**self.raw.get("bitsi", {}))

    def get_model_config(self) -> ModelConfig:
        return ModelConfig(**self.raw.get("model", {}))

    def get_training_config(self) -> TrainingConfig:
        return TrainingConfig(**self.raw.get("training", {}))

    def get_eval_config(self) -> EvalConfig:
        return EvalConfig(**self.raw.get("eval", {}))
