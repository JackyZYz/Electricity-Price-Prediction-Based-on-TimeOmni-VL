import argparse

import torch
from torch.utils.data import DataLoader

from timeomni_vl.config import ConfigManager
from timeomni_vl.data.pipeline import build_data_pipeline
from timeomni_vl.logger import get_logger
from timeomni_vl.models import build_adapter
from timeomni_vl.training.collator import TSUMMCollator
from timeomni_vl.training.dataset import TSUMMDatasetBuilder
from timeomni_vl.training.trainer import TSUMMTrainer
from timeomni_vl.utils.io import ensure_dir

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = ConfigManager(args.config)
    model_cfg = cfg.get_model_config()
    training_cfg = cfg.get_training_config()
    data_cfg = cfg.get_data_config()
    bitsi_cfg = cfg.get_bitsi_config()

    adapter = build_adapter(model_cfg.backbone)
    adapter.load(model_cfg.model_path, model_cfg.device)

    # Setup optimizer/LoRA for real backbones (Janus/Bagel)
    if hasattr(adapter, "setup_optimizer"):
        adapter.setup_optimizer(
            lr=training_cfg.lr,
            lora_enabled=training_cfg.lora_enabled,
            lora_rank=training_cfg.lora_rank,
            lora_alpha=training_cfg.lora_alpha,
            lora_target_modules=training_cfg.lora_target_modules,
            max_grad_norm=training_cfg.max_grad_norm,
        )
        trainable = adapter.get_trainable_params()
        logger.info(f"Trainable params: {trainable}")

    data_pipeline = build_data_pipeline(data_cfg)
    samples = data_pipeline.run()

    dataset_builder = TSUMMDatasetBuilder(data_cfg, bitsi_cfg)
    train_dataset = dataset_builder.build(samples["train"])
    val_dataset = dataset_builder.build(samples["val"]) if samples.get("val") else []

    collator = TSUMMCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_cfg.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

    output_dir = ensure_dir(data_cfg.output_dir) / "training"
    trainer = TSUMMTrainer(
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_cfg,
        output_dir=str(output_dir),
    )
    trainer.train()


if __name__ == "__main__":
    main()
