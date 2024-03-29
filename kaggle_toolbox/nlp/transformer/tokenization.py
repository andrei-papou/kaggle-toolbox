from __future__ import annotations

import typing as t
from dataclasses import dataclass

import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.generic import PaddingStrategy

from kaggle_toolbox.device import Device, CPUDevice


@dataclass
class TokenizerResult:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor

    def get_model_input(self) -> t.Dict[str, torch.Tensor]:
        return {
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'token_type_ids': self.token_type_ids,
        }

    def to_collatable_dict(self) -> t.Dict[str, torch.Tensor]:
        return {
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'token_type_ids': self.token_type_ids,
        }

    @classmethod
    def from_collateable_dict(cls, val: t.Dict[str, torch.Tensor]) -> TokenizerResult:
        return cls(**val)

    def __len__(self) -> int:
        return len(self.input_ids)

    @property
    def tensor_dict(self) -> t.Dict[str, torch.Tensor]:
        return self.get_model_input()

    def to_device(self, device: Device) -> TokenizerResult:
        return self.__class__.from_collateable_dict({
            k: v.to(device.as_torch) for k, v in self.to_collatable_dict().items()
        })

    def to_cpu(self) -> TokenizerResult:
        return self.to_device(device=CPUDevice())


class Tokenizer:
    result_type: t.Type[TokenizerResult]

    def __init__(self, inner: PreTrainedTokenizerBase, padding_strategy: PaddingStrategy = PaddingStrategy.MAX_LENGTH):
        self._inner = inner
        self._padding_strategy = padding_strategy
        self._special_token_list: t.List[str] = []

    @classmethod
    def from_checkpoint(
            cls,
            checkpoint: str,
            padding_strategy: PaddingStrategy = PaddingStrategy.MAX_LENGTH) -> Tokenizer:
        return cls(
            AutoTokenizer.from_pretrained(checkpoint),
            padding_strategy=padding_strategy)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._inner

    def __len__(self) -> int:
        return len(self.tokenizer)  # type: ignore

    @property
    def cls_token(self) -> str:
        return self.tokenizer.cls_token

    @property
    def sep_token(self) -> str:
        return self.tokenizer.sep_token

    @property
    def num_special_tokens(self) -> int:
        return len(self._special_token_list)

    def add_special_token_list(self, tok_list: t.List[str]):
        self._special_token_list.extend(tok_list)
        self.tokenizer.add_special_tokens({'additional_special_tokens': tok_list})  # type: ignore

    def tokenize(
            self,
            *texts: str,
            max_len: t.Optional[int] = None,
            add_special_tokens: bool = True) -> TokenizerResult:
        encoding = self.tokenizer(
            *texts,
            truncation=max_len is not None,
            max_length=max_len,
            padding=self._padding_strategy,
            return_attention_mask=True,
            return_token_type_ids=True,
            add_special_tokens=add_special_tokens)  # type: ignore
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(encoding['token_type_ids'], dtype=torch.long)

        return TokenizerResult(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
