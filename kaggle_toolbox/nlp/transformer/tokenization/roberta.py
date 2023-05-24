import typing as t
from dataclasses import dataclass

import torch
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.generic import PaddingStrategy

from .tokenizer import Tokenizer, TokenizerResult


@dataclass
class RobertaTokenizerResult(TokenizerResult):
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


class RobertaTokenizer(Tokenizer):
    result_type = RobertaTokenizerResult

    def __init__(self, checkpoint: str, padding_strategy: PaddingStrategy = PaddingStrategy.MAX_LENGTH):
        super().__init__(padding_strategy=padding_strategy)
        self._tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    def _build_result(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor) -> TokenizerResult:
        return RobertaTokenizerResult(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
