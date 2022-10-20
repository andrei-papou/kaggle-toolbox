import typing as t

import pandas as pd


def analyze_val_strategy(
        df: pd.DataFrame,
        target_list: t.List[str],
        num_folds: int,
        fold_col: str = 'fold') -> pd.DataFrame:
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
