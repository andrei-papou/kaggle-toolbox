import typing as t

from torch.utils.tensorboard.writer import SummaryWriter

from .base import Logger


class TensorBoardLogger(Logger):

    def __init__(self, log_dir: str, metric_whitelist: t.Optional[t.Set[str]] = None) -> None:
        self._metric_whitelist = metric_whitelist
        self._writer: SummaryWriter = SummaryWriter(log_dir=log_dir)

    def log_params(self, params: t.Dict[str, t.Any]):
        pass  # TODO: handle hyperparams properly.

    def log_metrics(self, step: int, metrics: t.Dict[str, float]):
        for metric_key, metric_val in metrics.items():
            if self._metric_whitelist is None or metric_key in self._metric_whitelist:
                self._writer.add_scalar(tag=metric_key, scalar_value=metric_val, global_step=step)