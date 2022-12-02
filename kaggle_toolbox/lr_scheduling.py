import typing_extensions as t_ext


class LRScheduler(t_ext.Protocol):

    def step(self):
        ...


class NoOpLRScheduler:

    def step(self):
        pass


def create_no_op_scheduler() -> LRScheduler:
    return NoOpLRScheduler()
