import typing as t

_T = t.TypeVar('_T')


class Environment:

    def __init__(self, env_type: str):
        self._env_type = env_type

    def param(self, **kwargs: _T) -> _T:
        return kwargs[self._env_type]
