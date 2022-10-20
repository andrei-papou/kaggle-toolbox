import typing as t
from enum import Enum


class EnvironmentType(str, Enum):
    local = 'local'
    colab = 'colab'

    @classmethod
    def raise_unknown(cls, unk_obj: t.Any) -> t.NoReturn:
        raise ValueError(f'Unknown environment type: {unk_obj}. Valid choices are: {list(cls)}.')
