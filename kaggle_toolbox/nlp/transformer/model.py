import itertools
import typing as t

import torch
import typing_extensions as t_ext
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout

from kaggle_toolbox.model import Model as BaseModel
from .backbone import Backbone
from .data import TokenizedX

_X = t.TypeVar('_X', bound=TokenizedX)


class MultiStagedDropout(torch.nn.Module):

    def __init__(
            self,
            inner_model: torch.nn.Module,
            num_stages: int,
            start_prob: float,
            increment: float):
        super().__init__()
        self._inner_model = inner_model
        self._dropout_list = torch.nn.ModuleList([
            StableDropout(start_prob + (increment * i)) for i in range(num_stages)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._inner_model(drop(x)) for drop in self._dropout_list], dim=0).mean(dim=0)


class Squeezer(torch.nn.Module):

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return super().__call__(features)


class TakeNthSqueezer(Squeezer):

    def __init__(self, n: int = -1):
        super().__init__()
        self._n = n

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, self._n]


class _LayerIdxBasedSqueezer(Squeezer):

    def __init__(self, layer_idx_list: t.List[int]) -> None:
        super().__init__()
        self._layer_idx_list = layer_idx_list


class ConcatSqueezer(_LayerIdxBasedSqueezer):

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.concat([features[:, idx] for idx in self._layer_idx_list], dim=-1)


class MeanSqueezer(_LayerIdxBasedSqueezer):

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack([features[:, idx] for idx in self._layer_idx_list], dim=0).mean(dim=0)


class SumSqueezer(_LayerIdxBasedSqueezer):

    def __init__(self, layer_idx_list: t.List[int], weight_list: t.Optional[t.List[float]] = None):
        super().__init__(layer_idx_list=layer_idx_list)
        self._weight_list = weight_list if weight_list is not None else [1.0] * len(self._layer_idx_list)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            features[:, idx] * w
            for idx, w in zip(self._layer_idx_list, self._weight_list)
        ], dim=0).sum(dim=0)


class Pooler(torch.nn.Module):

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def __call__(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return super().__call__(features, mask)


class ClsTokenPooler(Pooler):

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return features[:, 0, :]


class AttentionHeadPooler(Pooler):
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


class MeanPooler(Pooler):

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = mask.unsqueeze(-1).expand(features.size()).float()
        sum_embeddings = torch.sum(features * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class DNN(t_ext.Protocol):

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        ...

    def named_parameters(self) -> t.Iterator[t.Tuple[str, torch.nn.parameter.Parameter]]:
        ...


class Model(BaseModel[_X]):

    @property
    def backbone(self) -> Backbone[_X]:
        raise NotImplementedError()

    @property
    def backbone_named_parameters(self) -> t.Iterator[t.Tuple[str, torch.nn.parameter.Parameter]]:
        return self.backbone.named_parameters()

    @property
    def head_named_parameters(self) -> t.Iterator[t.Tuple[str, torch.nn.parameter.Parameter]]:
        raise NotImplementedError()

    def load_backbone(self, from_checkpoint: str):
        self.backbone.load_state_dict(
            torch.load(from_checkpoint, map_location=self.backbone.device))


class StandardModel(Model[_X]):

    def __init__(
            self,
            backbone: Backbone[_X],
            squeezer: Squeezer,
            pooler: Pooler,
            dnn: DNN):
        super().__init__()
        self._backbone = backbone
        self._squeezer = squeezer
        self._pooler = pooler
        self._dnn = dnn

    @property
    def backbone(self) -> Backbone[_X]:
        return self._backbone

    @property
    def head_named_parameters(self) -> t.Iterator[t.Tuple[str, torch.nn.parameter.Parameter]]:
        return itertools.chain(
            self._squeezer.named_parameters(),
            self._pooler.named_parameters(),
            self._dnn.named_parameters())

    def forward(self, x: _X) -> torch.Tensor:
        features = self._backbone(x)
        features = self._squeezer(features)
        features = self._pooler(features, mask=x.attention_mask)
        return self._dnn(features)
