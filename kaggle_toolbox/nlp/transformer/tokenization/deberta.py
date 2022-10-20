import typing as t
from dataclasses import dataclass

import torch
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.generic import PaddingStrategy

from .tokenizer import Tokenizer, TokenizerResult


@dataclass
class DebertaTokenizerResult(TokenizerResult):
    token_type_ids: torch.Tensor

    def get_model_input(self) -> t.Dict[str, torch.Tensor]:
        return {
            **super().get_model_input(),
            'token_type_ids': self.token_type_ids,
        }

    def to_collatable_dict(self) -> t.Dict[str, torch.Tensor]:
        return {
            **super().to_collatable_dict(),
            'token_type_ids': self.token_type_ids,
        }


class DebertaTokenizer(Tokenizer):
    result_type = DebertaTokenizerResult

    def __init__(self, checkpoint: str, padding_strategy: PaddingStrategy = PaddingStrategy.MAX_LENGTH):
        super().__init__(padding_strategy=padding_strategy)
        self._tokenizer = DebertaV2TokenizerFast.from_pretrained(checkpoint)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    def _build_result(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor) -> TokenizerResult:
        return DebertaTokenizerResult(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
