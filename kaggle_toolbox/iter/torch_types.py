import typing as t

import typing_extensions as t_ext

_T = t.TypeVar('_T', covariant=True)


class DataLoader(t.Iterable[_T], t.Sized, t_ext.Protocol):
    pass
