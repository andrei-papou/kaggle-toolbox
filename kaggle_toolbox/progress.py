import typing as t

import typing_extensions as t_ext
from tqdm import tqdm as ascii_tqdm
from tqdm.notebook import tqdm as notebook_tqdm

_T = t.TypeVar('_T', covariant=True)


class _ProgressBarInstance(t.Iterable[_T], t_ext.Protocol):

    def set_description(self, desc: str):
        ...


class _NoOpProgressBarInstance(t.Iterable[_T]):

    def __init__(self, it: t.Iterable[_T]) -> None:
        self._it = it

    def __iter__(self) -> t.Iterator[_T]:
        return iter(self._it)

    def set_description(self, desc: str):
        pass


class ProgressBar:

    @classmethod
    def attach_to_pandas(cls):
        raise NotImplementedError()

    def __call__(
            self,
            it: t.Iterable[_T],
            desc: t.Optional[str] = None,
            total: t.Optional[int] = None) -> _ProgressBarInstance[_T]:
        raise NotImplementedError()


class NoOpProgressBar(ProgressBar):

    @classmethod
    def attach_to_pandas(cls):
        pass

    def __call__(
            self,
            it: t.Iterable[_T],
            desc: t.Optional[str] = None,
            total: t.Optional[int] = None) -> _ProgressBarInstance[_T]:
        return _NoOpProgressBarInstance(it)


class ASCIIProgressBar(ProgressBar):

    @classmethod
    def attach_to_pandas(cls):
        ascii_tqdm.pandas()

    def __call__(
            self,
            it: t.Iterable[_T],
            desc: t.Optional[str] = None,
            total: t.Optional[int] = None) -> _ProgressBarInstance[_T]:
        return ascii_tqdm(it, desc=desc, total=total)


class NotebookProgressBar(ProgressBar):

    @classmethod
    def attach_to_pandas(cls):
        notebook_tqdm.pandas()

    def __call__(
            self,
            it: t.Iterable[_T],
            desc: t.Optional[str] = None,
            total: t.Optional[int] = None) -> _ProgressBarInstance[_T]:
        return notebook_tqdm(it, desc=desc, total=total)
