import typing as t

_K = t.TypeVar('_K', bound=t.Hashable)
_C = t.TypeVar('_C')


class ContextManagerList(t.Generic[_C]):

    def __init__(self, cm_list: t.Iterable[t.ContextManager[_C]]):
        self._cm_list = cm_list

    def __enter__(self) -> t.List[_C]:
        return [cm.__enter__() for cm in self._cm_list]

    def __exit__(self, *args, **kwargs):
        for cm in self._cm_list:
            cm.__exit__(*args, **kwargs)


class ContextManagerDict(t.Generic[_K, _C]):

    def __init__(self, cm_dict: t.Mapping[_K, t.ContextManager[_C]]) -> None:
        self._cm_dict = cm_dict

    def __enter__(self) -> t.Dict[_K, _C]:
        return {k: cm.__enter__() for k, cm in self._cm_dict.items()}

    def __exit__(self, *args, **kwargs):
        for _, cm in self._cm_dict.items():
            cm.__exit__(*args, **kwargs)
