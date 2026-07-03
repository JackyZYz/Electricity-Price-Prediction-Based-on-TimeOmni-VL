from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from timeomni_vl.config import TrainingConfig
from timeomni_vl.logger import get_logger
from timeomni_vl.models.adapter import BackboneAdapter
from timeomni_vl.utils.io import ensure_dir, save_json

logger = get_logger(__name__)


class TSUMMTrainer:
    def __init__(
        self,
        adapter: BackboneAdapter,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainingConfig,
        output_dir: str,
    ):
        self.adapter = adapter
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.global_step = 0

    def train(self) -> Dict[str, Any]:
        ensure_dir(self.output_dir)
        history = {"train_loss": [], "val_loss": []}
        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = self._train_epoch(epoch)
            history["train_loss"].append(train_metrics.get("loss_total", 0.0))
            logger.info(f"Epoch {epoch} train metrics: {train_metrics}")

            if self.val_loader and epoch % max(1, self.config.eval_every) == 0:
                val_metrics = self._validate()
                history["val_loss"].append(val_metrics.get("loss_total", 0.0))
                logger.info(f"Epoch {epoch} val metrics: {val_metrics}")

            if epoch % max(1, self.config.save_every) == 0:
                self._save_checkpoint(epoch)

        save_json(history, self.output_dir / "training_history.json")
        return history

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        total_loss = 0.0
        n_batches = 0
        for batch in self.train_loader:
            metrics = self.adapter.train_step(batch)
            total_loss += metrics.get("loss_total", 0.0)
            n_batches += 1
            self.global_step += 1
        return {"loss_total": total_loss / max(1, n_batches)}

    def _validate(self) -> Dict[str, float]:
        total_loss = 0.0
        n_batches = 0
        for batch in self.val_loader:
            metrics = self.adapter.train_step(batch)
            total_loss += metrics.get("loss_total", 0.0)
            n_batches += 1
        return {"loss_total": total_loss / max(1, n_batches)}

    def _save_checkpoint(self, epoch: int) -> None:
        path = self.output_dir / f"checkpoint_epoch_{epoch}"
        self.adapter.save_checkpoint(str(path))
        logger.info(f"Saved checkpoint to {path}")

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        logger.info(f"Step {self.global_step}: {metrics}")
