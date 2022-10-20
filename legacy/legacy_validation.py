import typing as t

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


class ValidationStrategy:
    fold_col = 'fold'

    def __init__(
            self,
            num_folds: int,
            target_list: t.List[str],
            num_bins: t.Optional[int] = None):
        self._num_folds = num_folds
        self._target_list = target_list
        self._num_bins = num_bins

    @property
    def num_folds(self) -> int:
        return self._num_folds

    @staticmethod
    def _target_to_bin_col(target: str) -> str:
        return f'{target}_bin'

    def assign_folds(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.fold_col] = -1

        num_bins = self._num_bins
        if num_bins is None:
            num_bins = int(np.floor(1 + np.log2(len(df))))

        for target in self._target_list:
            df.loc[:, self._target_to_bin_col(target)] = pd.cut(df[target], bins=num_bins, labels=False)
        mskf = MultilabelStratifiedKFold(n_splits=self._num_folds, shuffle=True, random_state=42)
        data_targets = df[[self._target_to_bin_col(target) for target in self._target_list]].values
        for fold_, (_, v_) in enumerate(mskf.split(X=df,y=data_targets)):
            df.loc[v_, 'fold'] = fold_
            
        df = df.drop([self._target_to_bin_col(target) for target in self._target_list], axis=1)
        return df


def analyze_val_strategy(df: pd.DataFrame, target_list: t.List[str], num_folds: int, fold_col: str) -> pd.DataFrame:
    row_list = []
    for fold in range(num_folds):
        row_list.append({
            'fold': fold,
            'num_samples': len(df[df[fold_col] == fold]),
            **{
                f'{t}_mean': df[df[fold_col] == fold][t].mean()
                for t in target_list
            }
        })
    return pd.DataFrame(row_list)


def build_fold_result_df(fold_list: t.List[int], score_list: t.List[float]) -> pd.DataFrame:
    return pd.DataFrame([
        {'fold': fold, 'score': score} for fold, score in zip(fold_list, score_list)
    ])
