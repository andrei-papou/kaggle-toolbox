from torch.optim import Optimizer
from transformers.optimization import get_cosine_schedule_with_warmup

from kaggle_toolbox.lr_scheduling import LRScheduler


def create_cosine_scheduler_with_warmup(
        optimizer: Optimizer,
        num_training_steps: int,
        warmup_steps_ratio: float,
        num_cycles: float = 0.5) -> LRScheduler:
    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(warmup_steps_ratio * num_training_steps),
        num_training_steps=num_training_steps,
        num_cycles=num_cycles)
