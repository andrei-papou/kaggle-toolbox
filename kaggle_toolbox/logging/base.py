from __future__ import annotations

import types
import typing as t


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
