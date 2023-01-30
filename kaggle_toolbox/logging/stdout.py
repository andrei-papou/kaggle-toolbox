import typing as t

from .base import Logger


class StdOutLogger(Logger):

    def _log_params(self, params: t.Dict[str, t.Any]):
        print('Using params:')
        for param in sorted(params.keys()):
            print(f'\t{param} = {params[param]}')

    def _log_metrics(self, step: int, metrics: t.Dict[str, float]):
        print(f'Step {step} metrics:')
        for m in sorted(metrics.keys()):
            print(f'\t{m} = {metrics[m]:.8f}')
