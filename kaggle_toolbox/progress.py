import typing as t

import typing_extensions as t_ext
from tqdm import tqdm as ascii_tqdm
from tqdm.notebook import tqdm as notebook_tqdm

_T = t.TypeVar('_T', covariant=True)


class _ProgressBarInstance(t.Iterable[_T], t_ext.Protocol):

    def set_description(self, desc: str):
        ...


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
