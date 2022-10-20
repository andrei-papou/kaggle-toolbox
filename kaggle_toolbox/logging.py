from __future__ import annotations

import types
import typing as t

import wandb
from torch.utils.tensorboard.writer import SummaryWriter
from wandb.wandb_run import Run as WAndBRun


class Logger:

    def __enter__(self) -> Logger:
        return self

    def __exit__(
            self,
            exc_type: t.Optional[t.Type[BaseException]],
            exc_value: t.Optional[BaseException],
            traceback: t.Optional[types.TracebackType]):
        pass

    def log_params(self, params: t.Dict[str, t.Any]):
        raise NotImplementedError()

    def log_metrics(self, step: int, metrics: t.Dict[str, float]):
        raise NotImplementedError()


class StdOutLogger(Logger):

    def log_params(self, params: t.Dict[str, t.Any]):
        print('Using params:')
        for param in sorted(params.keys()):
            print(f'\t{param} = {params[param]}')

    def log_metrics(self, step: int, metrics: t.Dict[str, float]):
        print(f'Step {step} metrics:')
        for m in sorted(metrics.keys()):
            print(f'\t{m} = {metrics[m]:.8f}')


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


_WANDB_DEFAULT_START_METHOD = 'thread'
_WANDB_DEFAULT_REINIT = True


class WAndBLogger(Logger):

    def __init__(
            self,
            user_name: str,
            api_key: str,
            project: str,
            run_id: str,
            metric_prefix: t.Optional[str] = None,
            start_method: str = _WANDB_DEFAULT_START_METHOD,
            reinit: bool = _WANDB_DEFAULT_REINIT):
        wandb.login(key=api_key)
        self._user_name = user_name
        self._project = project
        self._run_id = run_id
        self._metric_prefix = metric_prefix
        self._start_method = start_method
        self._run: t.Optional[WAndBRun] = None
        self._reinit = reinit
    
    @property
    def run(self) -> WAndBRun:
        assert self._run is not None
        return self._run

    def __enter__(self) -> WAndBLogger:
        self._run = t.cast(t.Optional[WAndBRun], wandb.init(
            project=self._project,
            entity=self._user_name,
            id=self._run_id,
            settings=wandb.Settings(start_method=self._start_method),
            reinit=self._reinit))
        return self

    def __exit__(
            self,
            exc_type: t.Optional[t.Type[BaseException]],
            exc_value: t.Optional[BaseException],
            traceback: t.Optional[types.TracebackType]):
        wandb.finish(exit_code=1 if exc_value is not None else 0)
    
    def log_params(self, params: t.Dict[str, t.Any]):
        self.run.config.update(params)

    def log_metrics(self, step: int, metrics: t.Dict[str, float]):
        self.run.log(
            step=step,
            data={
                f'{self._metric_prefix}_{k}' if self._metric_prefix is not None else k: v
                for k, v in metrics.items()
            },
            commit=True)
