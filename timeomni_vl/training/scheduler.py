import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            if self.warmup_steps == 0:
                return [base_lr for base_lr in self.base_lrs]
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]

        progress = (self.last_epoch - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine
            for base_lr in self.base_lrs
        ]
