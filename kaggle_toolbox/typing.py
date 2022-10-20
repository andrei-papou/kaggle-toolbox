import typing as t

_T = t.TypeVar('_T')


def pd_row(x: t.Any) -> t.Dict[str, t.Any]:
    return t.cast(t.Dict[str, t.Any], x)


class DynamicDict:

    def __init__(self, inner: t.Mapping[str, t.Any]):
        self._inner = inner

    def get_typed(self, key: str, type: t.Callable[[t.Any], _T]) -> t.Optional[_T]:
        raw = self._inner.get(key)
        return type(raw) if raw is not None else None

    def get_typed_or_raise(self, key: str, type: t.Callable[[t.Any], _T]) -> _T:
        return type(self._inner[key])


def unwrap_opt(x: t.Optional[_T]) -> _T:
    assert x is not None
    return x
