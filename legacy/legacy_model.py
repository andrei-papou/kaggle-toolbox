import itertools
import typing as t
from enum import Enum

import torch
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout

from .tokenization import ModelInputValue


class Model(torch.nn.Module):

    @property
    def backbone(self) -> torch.nn.Module:
        raise NotImplementedError()

    @property
    def backbone_named_parameters(self) -> t.Iterator[t.Tuple[str, torch.nn.parameter.Parameter]]:
        return self.backbone.named_parameters()

    @property
    def head_named_parameters(self) -> t.Iterator[t.Tuple[str, torch.nn.parameter.Parameter]]:
        raise NotImplementedError()

    def resize_token_embeddings(self, num_tokens: int):
        raise NotImplementedError()

    def forward(self, **inputs: ModelInputValue) -> torch.Tensor:
        raise NotImplementedError()

    def load_backbone(self, from_checkpoint: str):
        self.backbone.load_state_dict(
            torch.load(from_checkpoint, map_location=self.backbone.device))


class ModelBuilder:

    def __init__(
            self,
            backbone_checkpoint: str,
            num_targets: int,
            enable_gradient_checkpointing: bool = False):
        self._backbone_checkpoint = backbone_checkpoint
        self._num_targets = num_targets
        self._enable_gradient_checkpointing = enable_gradient_checkpointing

    def build(self) -> Model:
        raise NotImplementedError()


class ClsTokenPooler(torch.nn.Module):

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return features[:, 0, :]


class AttentionHeadPooler(torch.nn.Module):
    def __init__(self, h_size: int, hidden_dim: t.Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else h_size
        self._attention = torch.nn.Sequential(
            torch.nn.Linear(h_size, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self._attention(features)
        if mask is not None:
            score[mask == 0] = float('-inf')
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector


class MeanPooler(torch.nn.Module):

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = mask.unsqueeze(-1).expand(features.size()).float()
        sum_embeddings = torch.sum(features * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MultiStagedDropout(torch.nn.Module):

    def __init__(
            self,
            inner_model: torch.nn.Module,
            num_stages: int,
            start_prob: float,
            increment: float,
            dropout_cls: t.Callable[[float], torch.nn.Module] = StableDropout):
        super().__init__()
        self._inner_model = inner_model
        self._dropout_list = torch.nn.ModuleList([
            dropout_cls(start_prob + (increment * i)) for i in range(num_stages)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._inner_model(drop(x)) for drop in self._dropout_list], dim=0).mean(dim=0)


class PoolerType(str, Enum):
    cls_token = 'cls_token'
    mean = 'mean'
    attention = 'attention'


class _AutoModel(Model):

    def __init__(self, backbone_checkpoint: str, num_targets: int, pooler_type: PoolerType = PoolerType.mean):
        super().__init__()
        config = AutoConfig.from_pretrained(backbone_checkpoint)
        config.hidden_dropout = 0.
        config.hidden_dropout_prob = 0.
        config.attention_dropout = 0.
        config.attention_probs_dropout_prob = 0.
        config.output_hidden_states = True
        self._transformer = AutoModel.from_pretrained(backbone_checkpoint, config=config)
        if pooler_type == PoolerType.cls_token:
            self._pooler = ClsTokenPooler()
        elif pooler_type == PoolerType.attention:
            self._pooler = AttentionHeadPooler(h_size=config.hidden_size)
        elif pooler_type == PoolerType.mean:
            self._pooler = MeanPooler()
        else:
            raise ValueError(f'Wrong pooler_type = {pooler_type}.')
        # self._regressor = torch.nn.Sequential(
        #     # MultiStagedDropout(
        #     #     classifier=torch.nn.Linear(in_features=config.hidden_size, out_features=num_classes),
        #     #     num_stages=5,
        #     #     start_prob=config.hidden_dropout_prob - 0.02,
        #     #     increment=0.01),
        #     torch.nn.Linear(in_features=config.hidden_size, out_features=num_targets),
        # )
        self._regressor = torch.nn.Linear(in_features=config.hidden_size, out_features=num_targets)
        self._init_weights(self._regressor, config)

    def _init_weights(self, module: torch.nn.Module, config: AutoConfig):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)  # type: ignore
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)  # type: ignore
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @property
    def backbone(self) -> torch.nn.Module:
        return self._transformer

    @property
    def head_named_parameters(self) -> t.Iterator[t.Tuple[str, torch.nn.parameter.Parameter]]:
        return itertools.chain(
            self._pooler.named_parameters(),
            self._regressor.named_parameters())

    def resize_token_embeddings(self, num_tokens: int):
        self._transformer.resize_token_embeddings(num_tokens)

    def forward(self, **inputs: ModelInputValue) -> torch.Tensor:
        transformer_outputs = self._transformer(**inputs)
        x = transformer_outputs.hidden_states[-1]
        x = self._pooler(x, mask=inputs['attention_mask'])
        return self._regressor(x)


class AutoModelBuilder(ModelBuilder):

    def __init__(
            self,
            backbone_checkpoint: str,
            num_targets: int,
            enable_gradient_checkpointing: bool = False,
            pretrained_backbone_checkpoint: t.Optional[str] = None,
            pooler_type: PoolerType = PoolerType.mean):
        super().__init__(
            backbone_checkpoint=backbone_checkpoint,
            num_targets=num_targets,
            enable_gradient_checkpointing=enable_gradient_checkpointing)
        self._pretrained_backbone_checkpoint = pretrained_backbone_checkpoint
        self._pooler_type = pooler_type

    def build(self) -> Model:
        model = _AutoModel(
            self._backbone_checkpoint,
            num_targets=self._num_targets,
            pooler_type=self._pooler_type)
        if self._enable_gradient_checkpointing:
            model.backbone.gradient_checkpointing_enable()  # type: ignore
        if self._pretrained_backbone_checkpoint is not None:
            model.load_backbone(self._pretrained_backbone_checkpoint)
        return model
