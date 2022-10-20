import typing as t

from kaggle_toolbox.iter.sized_iter import SizedIter


class Index(t.NamedTuple):
    global_step: int
    local_step_pos: t.Tuple[int, int]


class _LocalIndexRange:

    def __init__(
            self,
            global_step: int,
            local_step_from: int,
            local_step_to: int,
            total_local_steps: int):
        self._global_step = global_step
        self._local_step_from = local_step_from
        self._local_step_to = local_step_to
        self._total_local_steps = total_local_steps
        self._local_step_current = local_step_from

    def __iter__(self) -> t.Iterator[Index]:
        return self

    def __next__(self) -> Index:
        if self._local_step_current >= self._local_step_to:
            raise StopIteration()
        step = self._local_step_current
        self._local_step_current += 1
        return Index(self._global_step, local_step_pos=(step, self._total_local_steps))


def local_index_range(
        global_step: int,
        local_step_from: int,
        local_step_to: int,
        total_local_steps: int) -> SizedIter[Index]:
    return SizedIter(
        it=_LocalIndexRange(
            global_step=global_step,
            local_step_from=local_step_from,
            local_step_to=local_step_to,
            total_local_steps=total_local_steps),
        n=local_step_to - local_step_from)
