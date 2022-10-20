import typing as t
from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, default_collate as default_collate_fn

from fp_ell.tokenization import TokenizerResult, Tokenizer
from fp_ell.typing import pd_row


@dataclass
class DatasetItem:
    id: t.List[str]
    tokenizer_result: TokenizerResult
    target: torch.Tensor


TorchCollator = t.Callable[[t.List[t.Dict[str, torch.Tensor]]], t.Dict[str, torch.Tensor]]


class Collator:

    def __init__(self, tokenizer_result_collator: TorchCollator = default_collate_fn):
        self._tokenizer_result_collator = tokenizer_result_collator

    def __call__(
            self,
            item_list: t.List[DatasetItem]) -> DatasetItem:
        assert len(item_list) > 0
        tokenizer_result_type = type(item_list[0].tokenizer_result)
        return DatasetItem(
            id=sum([item.id for item in item_list], []),
            tokenizer_result=tokenizer_result_type.from_collateable_dict(
                self._tokenizer_result_collator([item.tokenizer_result.to_collatable_dict() for item in item_list])),
            target=default_collate_fn([item.target for item in item_list]))


class Dataset(TorchDataset):

    def __init__(
            self,
            df: pd.DataFrame,
            tokenizer: Tokenizer,
            max_len: int,
            target_list: t.List[str]):
        self._df = df.copy().reset_index(drop=True)
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._target_list = target_list

    def _get_tokenizer_input(self, row: t.Dict[str, t.Any]) -> str:
        (
            full_text,
         ) = (
            str(row['full_text']),
         )

        return full_text

    def sort_by_tokenizer_input_len(self):
        self._df['_tok_input_len'] = self._df.progress_apply(self._get_tokenizer_input, axis=1)
        self._df = self._df.sort_values('_tok_input_len')

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> DatasetItem:
        row = self._df.iloc[idx]

        tokenizer_input = self._get_tokenizer_input(pd_row(row))
        id = str(row['text_id'])

        tokenizer_result = self._tokenizer.tokenize(
            tokenizer_input, max_len=self._max_len)
        target_tensor = torch.tensor(
            [float(row[target]) for target in self._target_list],
            dtype=torch.float32)

        return DatasetItem(
            id=[id],
            tokenizer_result=tokenizer_result,
            target=target_tensor)
