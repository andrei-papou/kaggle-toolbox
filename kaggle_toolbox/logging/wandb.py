from __future__ import annotations

import types
import typing as t

import wandb
from wandb.wandb_run import Run as WAndBRun

from .base import Logger


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