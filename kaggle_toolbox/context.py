import typing as t

_C = t.TypeVar('_C')


class ContextManagerList(t.Generic[_C]):

    def __init__(self, cm_list: t.Iterable[t.ContextManager[_C]]):
        self._cm_list = cm_list

    def __enter__(self) -> t.List[_C]:
        return [cm.__enter__() for cm in self._cm_list]

    def __exit__(self, *args, **kwargs):
        for cm in self._cm_list:
            cm.__exit__(*args, **kwargs)
