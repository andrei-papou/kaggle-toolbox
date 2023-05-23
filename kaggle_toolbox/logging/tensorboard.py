import typing as t

from torch.utils.tensorboard.writer import SummaryWriter

from kaggle_toolbox.metrics import Metric
from .base import Logger


class TensorBoardLogger(Logger):

    def __init__(
            self,
            log_dir: str,
            metric_prefix: t.Optional[str] = None,
            metric_whitelist: t.Optional[t.Set[t.Type[Metric]]] = None) -> None:
        super().__init__(metric_whitelist=metric_whitelist)
        self._writer: SummaryWriter = SummaryWriter(log_dir=log_dir)
        self._metric_prefix = metric_prefix

    def _log_params(self, params: t.Dict[str, t.Any]):
        pass  # TODO: handle hyperparams properly.

    def _log_metrics(self, step: int, metrics: t.Dict[str, float]):
        for metric_key, metric_val in metrics.items():
            self._writer.add_scalar(
                tag=f'{self._metric_prefix}_{metric_key}' if self._metric_prefix is not None else metric_key,
                scalar_value=metric_val,
                global_step=step)
