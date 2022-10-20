import typing as t
from pathlib import Path

import pandas as pd

_T = t.TypeVar('_T')


def pd_dataframe(x: t.Any) -> pd.DataFrame:
    return t.cast(pd.DataFrame, x)


def pd_series(x: t.Any) -> pd.Series:
    return t.cast(pd.Series, x)


def pd_row(x: t.Any) -> t.Dict[str, t.Any]:
    return t.cast(t.Dict[str, t.Any], x)


def unwrap_opt(x: t.Optional[_T]) -> _T:
    assert x is not None
    return x


def read_csv(path: Path) -> pd.DataFrame:
    return pd_dataframe(pd.read_csv(path))
