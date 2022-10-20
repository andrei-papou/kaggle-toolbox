import typing as t

from transformers.utils.generic import PaddingStrategy

from .tokenizer import Tokenizer, TokenizerResult
from .deberta import DebertaTokenizer
from .longformer import LongformerTokenizer


_BACKBONE_TO_TOKENIZER_TYPE = {
    'microsoft/deberta-v3-small': DebertaTokenizer,
    'microsoft/deberta-v3-base': DebertaTokenizer,
    'microsoft/deberta-v3-large': DebertaTokenizer,
    'allenai/longformer-base-4096': LongformerTokenizer,
}

def get_tokenizer_for_backbone(
        backbone: str,
        checkpoint: t.Optional[str] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.MAX_LENGTH) -> Tokenizer:
    checkpoint = checkpoint if checkpoint is not None else backbone
    tokenizer_type = _BACKBONE_TO_TOKENIZER_TYPE.get(backbone)
    if tokenizer_type is None:
        raise ValueError(f'Backbone "{backbone}" is not supported.')
    return tokenizer_type(checkpoint, padding_strategy=padding_strategy)
