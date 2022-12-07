from __future__ import annotations

import types
import typing as t

from kaggle_toolbox.data import DatasetKind
from kaggle_toolbox.metrics import PredQualityMetric


class Logger:

    def __init__(self, metric_whitelist: t.Optional[t.Set[t.Type[PredQualityMetric]]] = None) -> None:
        self._metric_whitelist: t.Optional[t.Set[str]]
        if metric_whitelist is not None:
            self._metric_whitelist = set()
            for m in metric_whitelist:
                self._metric_whitelist.add(m.name_for_dataset_kind(DatasetKind.train))
                self._metric_whitelist.add(m.name_for_dataset_kind(DatasetKind.valid))
        else:
            self._metric_whitelist = None

    def __enter__(self) -> Logger:
        return self

    def __exit__(
            self,
            exc_type: t.Optional[t.Type[BaseException]],
            exc_value: t.Optional[BaseException],
            traceback: t.Optional[types.TracebackType]):
        pass

    def _get_metrics_to_track(self, metrics: t.Dict[str, float]) -> t.Dict[str, float]:
        return {
            k: v for k, v in metrics.items()
            if self._metric_whitelist is None or k in self._metric_whitelist
        }

    def _log_params(self, params: t.Dict[str, t.Any]):
        raise NotImplementedError()

    def log_params(self, params: t.Dict[str, t.Any]):
        self._log_params(params)

    def _log_metrics(self, step: int, metrics: t.Dict[str, float]):
        raise NotImplementedError()

    def log_metrics(self, step: int, metrics: t.Dict[str, float]):
        self._log_metrics(step, self._get_metrics_to_track(metrics))
