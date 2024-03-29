import typing as t

import torch

from kaggle_toolbox.data import LabeledDatasetItem
from kaggle_toolbox.tabular.activations import t_softmax
from .components import GFLU


class GANDALFBackbone(torch.nn.Module):

    def __init__(
            self,
            input_dim: int,
            num_gflu_stages: int,
            gflu_dropout: float = 0.0,
            gflu_feature_init_sparsity: float = 0.3,
            learnable_sparsity: bool = True,):
        super().__init__()
        self.n_features = input_dim
        self.output_dim = input_dim

        self.gflus = GFLU(
            n_features_in=self.n_features,
            n_stages=num_gflu_stages,
            dropout=gflu_dropout,
            feature_sparsity=gflu_feature_init_sparsity,
            learnable_sparsity=learnable_sparsity,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gflus(x)

    @property
    def feature_importance_(self) -> torch.Tensor:
        return t_softmax(self.gflus.feature_masks)\
            .sum(dim=0).detach().cpu()


class GANDALFHead(torch.nn.Module):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            is_regression: bool):
        super().__init__()
        self._head = self._build_head(input_dim, output_dim)
        self._t0 = torch.nn.parameter.Parameter(
            torch.rand(output_dim),
            requires_grad=True)

        self._is_regression = is_regression

    @classmethod
    def _build_head(cls, in_dim: int, out_dim: int) -> torch.nn.Module:
        return torch.nn.Linear(in_dim, out_dim)

    def init_from_data(self, data_loader: t.Iterable[LabeledDatasetItem[t.Any]]):
        if self._is_regression:
            batch = next(iter(data_loader))
            self._t0.data = torch.mean(batch.y, dim=0)

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        y_hat = self._head(backbone_features)
        if self._is_regression:
            y_hat = y_hat + self._t0
        return y_hat
